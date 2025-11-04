#!/usr/bin/env python3
"""
Single-file solution for the Hull Tactical Market Prediction challenge.

This script trains a blended ensemble of gradient-boosted tree models on the
public training data, calibrates exposures to align with the competition's
adjusted Sharpe metric, and produces a submission CSV for the public test set.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

# Use a non-interactive backend so plots can be generated in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import ElasticNet, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise SystemExit("LightGBM is required. Install it before running the script.") from exc

try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise SystemExit("CatBoost is required. Install it before running the script.") from exc

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit("XGBoost is required. Install it before running the script.") from exc


# Competition constants
MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0
TRADING_DAYS_PER_YEAR = 252
RANDOM_STATE = 42
TARGET_VOL_RATIO = 1.08
VOL_TOLERANCE = 0.015
MAX_REFINE_SCALE = 150.0
MAX_VOL_RATIO_ALLOWED = 1.12
ROLLING_WINDOWS = (5, 21)
ROLLING_CORR_THRESHOLD = 0.01
ROLLING_MAX_FEATURES = 150


class ParticipantVisibleError(Exception):
    pass


def competition_score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    if not pd.api.types.is_numeric_dtype(submission["prediction"]):
        raise ParticipantVisibleError("Predictions must be numeric")

    df = solution.copy()
    df["position"] = submission["prediction"].clip(MIN_INVESTMENT, MAX_INVESTMENT)

    if df["position"].max() > MAX_INVESTMENT or df["position"].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError("Predictions outside [0, 2] bounds")

    df["strategy_returns"] = df["risk_free_rate"] * (1 - df["position"]) + df["position"] * df["forward_returns"]

    strategy_excess_returns = df["strategy_returns"] - df["risk_free_rate"]
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = strategy_excess_cumulative ** (1 / len(df)) - 1
    strategy_std = df["strategy_returns"].std()

    if strategy_std == 0:
        raise ParticipantVisibleError("Division by zero, strategy std is zero")

    sharpe = strategy_mean_excess_return / strategy_std * math.sqrt(TRADING_DAYS_PER_YEAR)
    strategy_volatility = float(strategy_std * math.sqrt(TRADING_DAYS_PER_YEAR) * 100)

    market_excess_returns = df["forward_returns"] - df["risk_free_rate"]
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = market_excess_cumulative ** (1 / len(df)) - 1
    market_std = df["forward_returns"].std()
    if market_std == 0:
        raise ParticipantVisibleError("Division by zero, market std is zero")
    market_volatility = float(market_std * math.sqrt(TRADING_DAYS_PER_YEAR) * 100)

    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    return_gap = max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * TRADING_DAYS_PER_YEAR)
    return_penalty = 1 + (return_gap**2) / 100

    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000.0)

PREFERRED_FEATURES = [
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "E7",
    "E8",
    "E9",
    "E10",
    "E11",
    "E12",
    "E13",
    "E14",
    "E15",
    "E16",
    "E17",
    "E18",
    "E19",
    "E20",
    "I2",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "P9",
    "P10",
    "P11",
    "P12",
    "P13",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S12",
    "D1",
    "D2",
    "D3",
    "D4",
    "M2",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "M10",
    "M11",
    "M12",
    "M13",
    "M14",
    "M15",
    "M16",
    "M17",
    "M18",
    "lagged_forward_returns",
    "lagged_risk_free_rate",
    "lagged_market_forward_excess_returns",
]


@dataclass
class CalibrationResult:
    params: Dict[str, float]
    positions: np.ndarray
    score: float
    volatility_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hull Tactical Market Prediction pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("kaggle/input/hull-tactical-market-prediction"),
        help="Directory containing train.csv and test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where submission and optional artifacts will be saved.",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate diagnostic plots saved under <output-dir>/plots.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of temporal folds for cross-validation.",
    )
    parser.add_argument(
        "--optimizer-trials",
        type=int,
        default=600,
        help="Number of random search trials for blending/calibration optimizer.",
    )
    parser.add_argument(
        "--optimizer-seed",
        type=int,
        default=1337,
        help="Random seed used for optimizer reproducibility.",
    )
    parser.add_argument(
        "--enable-rolling-features",
        action="store_true",
        help="Generate rolling mean/std features for curated columns before training.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence extremely noisy sub-loggers when running with DEBUG verbosity.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def load_datasets(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected train.csv and test.csv inside the data directory.")

    logging.info("Loading train/test data from %s", data_dir)
    train_df = pd.read_csv(train_path, na_values=["null"])
    test_df = pd.read_csv(test_path, na_values=["null"])
    sample_df = pd.read_csv(sample_path) if sample_path.exists() else None

    logging.info("Train shape: %s | Test shape: %s", train_df.shape, test_df.shape)
    return train_df, test_df, sample_df


def select_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
    reserved_cols = {
        "forward_returns",
        "risk_free_rate",
        "market_forward_excess_returns",
    }
    candidate_cols = sorted(set(train_df.columns) & set(test_df.columns))
    candidate_cols = [c for c in candidate_cols if c not in reserved_cols]

    curated = [col for col in PREFERRED_FEATURES if col in candidate_cols]
    if len(curated) >= 30:
        logging.info("Using curated feature subset (%d columns)", len(curated))
        return curated

    logging.info(
        "Curated subset unavailable (%d columns); using all %d candidate columns",
        len(curated),
        len(candidate_cols),
    )
    return candidate_cols


def augment_with_rolling_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_features: List[str],
    windows: Tuple[int, ...] = ROLLING_WINDOWS,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if "date_id" not in train_df.columns:
        return train_df, test_df, []

    combined = pd.concat(
        [
            train_df.assign(_dataset="train"),
            test_df.assign(_dataset="test"),
        ],
        ignore_index=True,
    ).sort_values("date_id")

    rolling_features: List[str] = []
    candidate_cols = [
        col
        for col in base_features
        if col[0] in {"E", "P", "S", "M", "V", "I"}
    ]

    for col in candidate_cols:
        if col not in combined.columns:
            continue
        col_series = pd.to_numeric(combined[col], errors="coerce")
        for window in windows:
            mean_name = f"{col}_mean_{window}"
            std_name = f"{col}_std_{window}"
            combined[mean_name] = col_series.rolling(window, min_periods=1).mean()
            combined[std_name] = col_series.rolling(window, min_periods=1).std().fillna(0.0)
            rolling_features.extend([mean_name, std_name])

    combined = combined.sort_index()
    train_aug = combined[combined["_dataset"] == "train"].drop(columns="_dataset")
    test_aug = combined[combined["_dataset"] == "test"].drop(columns="_dataset")
    if not rolling_features:
        return train_df, test_df, []

    correlations = (
        train_aug[rolling_features]
        .corrwith(pd.to_numeric(train_aug["forward_returns"], errors="coerce"))
        .abs()
        .fillna(0.0)
    )
    selected = correlations[correlations >= ROLLING_CORR_THRESHOLD].sort_values(ascending=False)
    if ROLLING_MAX_FEATURES and len(selected) > ROLLING_MAX_FEATURES:
        selected = selected.head(ROLLING_MAX_FEATURES)
    keep_features = selected.index.tolist()

    drop_features = [feat for feat in rolling_features if feat not in keep_features]
    train_aug = train_aug.drop(columns=drop_features)
    test_aug = test_aug.drop(columns=drop_features)

    logging.info(
        "Rolling features generated=%d | selected=%d (threshold=%.4f)",
        len(rolling_features),
        len(keep_features),
        ROLLING_CORR_THRESHOLD,
    )
    return train_aug, test_aug, keep_features


def preprocess_features(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    features = df.loc[:, list(feature_cols)].copy()
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features = features.fillna(0.0)
    return features.astype(np.float32)


def adjusted_sharpe(solution_df: pd.DataFrame, positions: np.ndarray) -> float:
    if positions.max() > MAX_INVESTMENT + 1e-6 or positions.min() < MIN_INVESTMENT - 1e-6:
        raise ValueError("Positions outside allowable bounds.")

    rf = solution_df["risk_free_rate"].to_numpy()
    forward = solution_df["forward_returns"].to_numpy()

    strategy_returns = rf * (1.0 - positions) + forward * positions
    strategy_excess = strategy_returns - rf
    strategy_excess_cumulative = np.prod(1 + strategy_excess)
    strategy_mean_excess = strategy_excess_cumulative ** (1 / len(solution_df)) - 1
    strategy_std = strategy_returns.std()

    if strategy_std == 0:
        return -math.inf

    sharpe = strategy_mean_excess / strategy_std * math.sqrt(TRADING_DAYS_PER_YEAR)
    strategy_volatility = float(strategy_std * math.sqrt(TRADING_DAYS_PER_YEAR) * 100)

    market_excess_returns = forward - rf
    market_excess_cumulative = np.prod(1 + market_excess_returns)
    market_mean_excess_return = market_excess_cumulative ** (1 / len(solution_df)) - 1
    market_std = forward.std()
    if market_std == 0:
        return -math.inf
    market_volatility = float(market_std * math.sqrt(TRADING_DAYS_PER_YEAR) * 100)

    excess_vol = max(0.0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0.0
    vol_penalty = 1 + excess_vol

    return_gap = max(
        0.0,
        (market_mean_excess_return - strategy_mean_excess) * 100 * TRADING_DAYS_PER_YEAR,
    )
    return_penalty = 1 + (return_gap ** 2) / 100.0

    adjusted = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted), 1_000_000.0)


def compute_volatility_ratio(solution_df: pd.DataFrame, positions: np.ndarray) -> float:
    rf = solution_df["risk_free_rate"].to_numpy()
    fwd = solution_df["forward_returns"].to_numpy()
    strategy_returns = rf * (1.0 - positions) + fwd * positions
    strat_std = strategy_returns.std()
    market_std = fwd.std()
    if market_std == 0:
        return float("nan")
    return float(strat_std / market_std)


def positions_from_params(positive_signal: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    threshold = params.get("threshold", 0.0)
    power = params.get("power", 1.0)
    scale = params.get("scale", 1.0)
    shifted = np.clip(positive_signal - threshold, 0.0, None)
    transformed = shifted ** power
    return np.clip(transformed * scale, MIN_INVESTMENT, MAX_INVESTMENT)


def refine_scale_to_target(
    positive_signal: np.ndarray,
    params: Dict[str, float],
    solution_df: pd.DataFrame,
) -> Tuple[Dict[str, float], np.ndarray, float, float]:
    scale = params.get("scale", 1.0)
    hit_upper_bound = False
    hit_lower_bound = False
    for _ in range(60):
        params["scale"] = float(scale)
        positions = positions_from_params(positive_signal, params)
        vol_ratio = compute_volatility_ratio(solution_df, positions)
        if math.isnan(vol_ratio):
            break
        if abs(vol_ratio - TARGET_VOL_RATIO) <= VOL_TOLERANCE:
            break
        if vol_ratio < TARGET_VOL_RATIO:
            new_scale = min(scale * 1.12, MAX_REFINE_SCALE)
            if new_scale == MAX_REFINE_SCALE:
                hit_upper_bound = True
            scale = new_scale
        else:
            new_scale = max(scale / 1.12, 0.05)
            if new_scale == 0.05:
                hit_lower_bound = True
            scale = new_scale
    params["scale"] = float(scale)
    positions = positions_from_params(positive_signal, params)
    vol_ratio = compute_volatility_ratio(solution_df, positions)
    score = adjusted_sharpe(solution_df, positions)
    if hit_upper_bound:
        logging.warning("Scale refinement hit upper bound (%.2f)", MAX_REFINE_SCALE)
    if hit_lower_bound:
        logging.warning("Scale refinement hit lower bound (0.05)")
    return params, positions, score, vol_ratio


def calibrate_positions(
    solution_df: pd.DataFrame,
    raw_predictions: np.ndarray,
    label: str,
) -> CalibrationResult:
    positive_signal = np.maximum(raw_predictions, 0.0)
    if np.allclose(positive_signal, 0.0):
        logging.warning("All non-positive signals for %s; returning flat zero allocation.", label)
        zero_positions = np.zeros_like(raw_predictions)
        return CalibrationResult(
            params={"threshold": 0.0, "scale": 0.0, "power": 1.0},
            positions=zero_positions,
            score=adjusted_sharpe(solution_df, zero_positions),
            volatility_ratio=0.0,
        )

    quantile_grid = np.linspace(0.0, 0.99, 12)
    thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.0], dtype=np.float64),
                np.quantile(positive_signal, quantile_grid),
            ]
        )
    )
    scale_grid = np.unique(
        np.concatenate(
            [
                np.linspace(5.0, 80.0, 16),
                np.linspace(82.0, 150.0, 14),
            ]
        )
    )
    power_grid = [0.5, 0.65, 0.8, 1.0]

    best_score = -math.inf
    best_params: Dict[str, float] | None = None
    best_positions: np.ndarray | None = None

    fallback_used = False

    for thr in thresholds:
        for power in power_grid:
            candidate_base = {"threshold": float(thr), "power": float(power)}
            shifted = np.clip(positive_signal - thr, 0.0, None)
            if np.allclose(shifted, 0.0):
                continue
            for scale in scale_grid:
                candidate_params = {**candidate_base, "scale": float(scale)}
                positions = positions_from_params(positive_signal, candidate_params)
                if np.allclose(positions, 0.0):
                    continue
                score = adjusted_sharpe(solution_df, positions)
                if score > best_score:
                    best_score = score
                    best_positions = positions
                    best_params = candidate_params

    if best_params is None or best_positions is None:
        target_level = np.quantile(positive_signal, 0.98) + 1e-6
        scale = MAX_INVESTMENT / target_level
        fallback_positions = np.clip(positive_signal * scale, MIN_INVESTMENT, MAX_INVESTMENT)
        fallback_score = adjusted_sharpe(solution_df, fallback_positions)
        best_params = {"threshold": 0.0, "scale": float(scale), "power": 1.0}
        best_positions = fallback_positions
        best_score = fallback_score
        fallback_used = True

    best_params, best_positions, best_score, vol_ratio = refine_scale_to_target(
        positive_signal, best_params, solution_df
    )
    msg = (
        "Calibration for %s | Score %.4f | threshold=%.5f scale=%.4f power=%.2f | vol_ratio=%.3f"
        % (
            label,
            best_score,
            best_params["threshold"],
            best_params["scale"],
            best_params["power"],
            vol_ratio,
        )
    )
    if fallback_used:
        logging.warning("%s (fallback mapping applied)", msg)
    else:
        logging.info(msg)
    return CalibrationResult(best_params, best_positions, best_score, vol_ratio)


def apply_calibration(raw_predictions: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    threshold = params.get("threshold", 0.0)
    scale = params.get("scale", 1.0)
    power = params.get("power", 1.0)
    signal = np.maximum(raw_predictions - threshold, 0.0)
    transformed = signal ** power
    return np.clip(transformed * scale, MIN_INVESTMENT, MAX_INVESTMENT)


def blend_predictions(predictions: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    combined = None
    for name, weight in weights.items():
        vec = predictions[name]
        combined = weight * vec if combined is None else combined + weight * vec
    return combined


def optimize_blend_and_calibration(
    oof_preds: Dict[str, np.ndarray],
    solution_df: pd.DataFrame,
    n_trials: int,
    seed: int,
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    float,
    float,
    List[Dict[str, float]],
    Dict[str, float],
]:
    rng = np.random.default_rng(seed)

    names = sorted(oof_preds.keys())
    matrix = np.column_stack([oof_preds[name] for name in names])
    best_score = -math.inf
    best_weights: Dict[str, float] | None = None
    best_params: Dict[str, float] | None = None
    best_vol = float("inf")

    trial_records: List[Dict[str, float]] = []

    for trial in range(n_trials):
        raw_weights = rng.dirichlet(np.ones(len(names)))
        weights = {name: float(w) for name, w in zip(names, raw_weights)}
        combined = matrix @ raw_weights
        positive_signal = np.maximum(combined, 0.0)
        if np.allclose(positive_signal, 0.0):
            continue

        threshold = float(np.quantile(positive_signal, rng.uniform(0.0, 0.4)))
        power = float(rng.choice([0.5, 0.65, 0.8, 1.0]))
        init_scale = float(rng.uniform(5.0, 80.0))
        params = {"threshold": threshold, "power": power, "scale": init_scale}
        params, positions, score, vol_ratio = refine_scale_to_target(
            positive_signal, params, solution_df
        )
        if vol_ratio > MAX_VOL_RATIO_ALLOWED:
            continue

        trial_records.append(
            {
                "trial": trial,
                "score": score,
                "vol_ratio": vol_ratio,
                "weights": weights,
                "params": params,
            }
        )

        if score > best_score:
            best_score = score
            best_weights = weights
            best_params = params
            best_vol = vol_ratio

    if best_weights is None or best_params is None:
        raise RuntimeError("Optimizer failed to find a valid configuration.")

    greedy_weights = best_weights.copy()
    for _ in range(30):
        target = rng.choice(names)
        delta = rng.uniform(-0.05, 0.05)
        new_weights = greedy_weights.copy()
        new_weights[target] = max(0.0, new_weights[target] + delta)
        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}

        combined = blend_predictions(oof_preds, new_weights)
        positive_signal = np.maximum(combined, 0.0)
        params = best_params.copy()
        params, positions, score, vol_ratio = refine_scale_to_target(
            positive_signal, params, solution_df
        )
        if vol_ratio <= MAX_VOL_RATIO_ALLOWED and score > best_score:
            best_score = score
            best_vol = vol_ratio
            greedy_weights = new_weights
            best_params = params

    return greedy_weights, best_params, best_score, best_vol, trial_records, best_weights


def build_model_builders(random_state: int) -> OrderedDict[str, callable]:
    model_builders: OrderedDict[str, callable] = OrderedDict()

    model_builders["lgbm"] = lambda: LGBMRegressor(
        n_estimators=1800,
        learning_rate=0.02,
        num_leaves=128,
        colsample_bytree=0.65,
        subsample=0.85,
        subsample_freq=3,
        min_child_samples=20,
        min_split_gain=0.0,
        reg_alpha=1.0,
        reg_lambda=3.0,
        objective="rmse",
        random_state=random_state,
        n_jobs=-1,
    )

    model_builders["xgb"] = lambda: XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=1.5,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )

    model_builders["catboost"] = lambda: CatBoostRegressor(
        iterations=2500,
        depth=7,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=random_state,
        l2_leaf_reg=3.0,
        subsample=0.85,
        grow_policy="Depthwise",
        verbose=False,
        allow_writing_files=False,
    )

    model_builders["elasticnet"] = lambda: Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "elasticnet",
                ElasticNet(
                    alpha=5e-4,
                    l1_ratio=0.25,
                    max_iter=7000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model_builders["elasticnet_l1"] = lambda: Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "elasticnet_l1",
                ElasticNet(
                    alpha=1e-3,
                    l1_ratio=0.85,
                    max_iter=7000,
                    random_state=random_state + 1,
                ),
            ),
        ]
    )

    return model_builders


def fit_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> None:
    if name == "lgbm":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
    elif name == "xgb":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=100,
        )
    elif name == "catboost":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)


def run_time_series_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    solution_df: pd.DataFrame,
    model_builders: OrderedDict[str, callable],
    n_splits: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, float]]]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    oof_predictions = {name: np.zeros(len(X), dtype=np.float64) for name in model_builders}
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
        logging.info(
            "Fold %d/%d | train=%d | val=%d",
            fold_idx,
            n_splits,
            len(train_idx),
            len(val_idx),
        )
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        sol_val = solution_df.iloc[val_idx]

        for name, builder in model_builders.items():
            model = builder()
            fit_model(name, model, X_train, y_train, X_val, y_val)
            preds = model.predict(X_val)
            oof_predictions[name][val_idx] = preds
            mse = mean_squared_error(y_val, preds)
            rmse = float(np.sqrt(mse))
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "model": name,
                    "rmse": float(rmse),
                }
            )

            calib = calibrate_positions(sol_val, preds, label=f"{name} fold={fold_idx}")
            fold_metrics[-1]["adjusted_sharpe"] = calib.score
            fold_metrics[-1]["vol_ratio"] = calib.volatility_ratio

    return oof_predictions, fold_metrics


def train_full_models(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    model_builders: OrderedDict[str, callable],
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    base_models: Dict[str, object] = {}
    test_predictions: Dict[str, np.ndarray] = {}

    for name, builder in model_builders.items():
        model = builder()
        logging.info("Training %s on the full dataset.", name)
        model.fit(X, y)
        base_models[name] = model
        test_predictions[name] = model.predict(X_test)

    return base_models, test_predictions


def summarize_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    metrics_df = pd.DataFrame(fold_metrics)
    summary: Dict[str, Dict[str, float]] = {}
    for model_name, group in metrics_df.groupby("model"):
        summary[model_name] = {
            "rmse_mean": group["rmse"].mean(),
            "rmse_std": group["rmse"].std(),
            "adjusted_sharpe_mean": group["adjusted_sharpe"].mean(),
            "vol_ratio_mean": group["vol_ratio"].mean(),
        }
    return summary


def generate_diagnostics(
    output_dir: Path,
    train_df: pd.DataFrame,
    oof_raw: np.ndarray,
    oof_positions: np.ndarray,
    feature_cols: List[str],
    lgbm_model,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    enriched = train_df.copy()
    enriched["predicted_forward_returns"] = oof_raw
    enriched["position"] = oof_positions
    enriched = enriched.sort_values("date_id").reset_index(drop=True)
    enriched["strategy_returns"] = (
        enriched["risk_free_rate"] * (1.0 - enriched["position"])
        + enriched["forward_returns"] * enriched["position"]
    )

    window = 63  # ~ quarter
    vol_scale = math.sqrt(TRADING_DAYS_PER_YEAR)
    enriched["strategy_vol_est"] = enriched["strategy_returns"].rolling(window).std() * vol_scale
    enriched["market_vol_est"] = enriched["forward_returns"].rolling(window).std() * vol_scale

    # Plot 1: predicted vs actual scatter
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=enriched.sample(min(5000, len(enriched)), random_state=RANDOM_STATE),
        x="predicted_forward_returns",
        y="forward_returns",
        hue="position",
        palette="viridis",
        alpha=0.6,
        edgecolor="none",
    )
    plt.title("Predicted vs Actual Forward Returns")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.legend(title="Position", loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(plots_dir / "pred_vs_actual.png")
    plt.close()

    # Plot 2: rolling volatility
    plt.figure(figsize=(10, 6))
    plt.plot(enriched["date_id"], enriched["strategy_vol_est"], label="Strategy (rolling 63d)")
    plt.plot(enriched["date_id"], enriched["market_vol_est"], label="Market (rolling 63d)")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.title("Rolling Volatility Comparison")
    plt.xlabel("date_id")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "rolling_volatility.png")
    plt.close()

    # Plot 3: feature importance (LightGBM)
    if hasattr(lgbm_model, "feature_importances_"):
        importances = pd.Series(lgbm_model.feature_importances_, index=feature_cols)
        top_features = importances.sort_values(ascending=False).head(25)
        plt.figure(figsize=(8, 10))
        sns.barplot(x=top_features.values, y=top_features.index, color="#4c72b0")
        plt.title("LightGBM Feature Importance (Top 25)")
        plt.xlabel("Importance")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance_lightgbm.png")
        plt.close()

    logging.info("Saved diagnostic plots to %s", plots_dir)


def build_submission(
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame | None,
    predictions: np.ndarray,
) -> pd.DataFrame:
    if sample_df is not None and "prediction" in sample_df.columns:
        submission = sample_df.copy()
        submission["prediction"] = predictions
    else:
        submission = pd.DataFrame(
            {
                "date_id": test_df["date_id"].values,
                "prediction": predictions,
            }
        )
    return submission


def save_linear_coefficients(
    models: Dict[str, object],
    feature_cols: List[str],
    output_dir: Path,
) -> None:
    coeff_payload: Dict[str, Dict[str, object]] = {}

    for name, model in models.items():
        if not hasattr(model, "named_steps"):
            continue

        steps = model.named_steps
        if "elasticnet" in steps:
            linear = steps["elasticnet"]
            scaler = steps.get("scaler")
            tag = "elasticnet"
        elif "elasticnet_l1" in steps:
            linear = steps["elasticnet_l1"]
            scaler = steps.get("scaler")
            tag = "elasticnet_l1"
        else:
            continue

        if scaler is None or not hasattr(scaler, "scale_"):
            continue

        scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        coef_raw = linear.coef_ / scale
        intercept_raw = float(
            linear.intercept_ - np.sum((linear.coef_ * scaler.mean_) / scale)
        )
        coeff_payload[name] = {
            "tag": tag,
            "intercept": intercept_raw,
            "coefficients": [
                {"feature": feat, "coefficient": float(coef)}
                for feat, coef in zip(feature_cols, coef_raw)
            ],
        }

    if coeff_payload:
        output_dir.mkdir(parents=True, exist_ok=True)
        coeff_path = output_dir / "linear_coefficients.json"
        with coeff_path.open("w", encoding="utf-8") as fh:
            json.dump(coeff_payload, fh, indent=2)
        logging.info("Stored linear model coefficients to %s", coeff_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, sample_df = load_datasets(args.data_dir)
    feature_cols = select_feature_columns(train_df, test_df)
    if args.enable_rolling_features:
        train_df, test_df, rolling_features = augment_with_rolling_features(
            train_df, test_df, feature_cols
        )
        if rolling_features:
            feature_cols = feature_cols + rolling_features

    X_train = preprocess_features(train_df, feature_cols)
    X_test = preprocess_features(test_df, feature_cols)
    y_train = train_df["forward_returns"].to_numpy(dtype=np.float64)
    solution_cols = ["forward_returns", "risk_free_rate"]
    if "market_forward_excess_returns" in train_df.columns:
        solution_cols.append("market_forward_excess_returns")
    solution_df = train_df.loc[:, solution_cols]

    model_builders = build_model_builders(RANDOM_STATE)
    oof_predictions, fold_metrics = run_time_series_cv(
        X_train, y_train, solution_df, model_builders, n_splits=args.cv_splits
    )

    summary_metrics = summarize_fold_metrics(fold_metrics)
    for model_name, stats in summary_metrics.items():
        logging.info(
            "OOF summary for %s | RMSE %.6f ± %.6f | adj_sharpe_mean %.4f | vol_ratio_mean %.3f",
            model_name,
            stats["rmse_mean"],
            stats["rmse_std"],
            stats["adjusted_sharpe_mean"],
            stats["vol_ratio_mean"],
        )

    model_order = list(model_builders.keys())
    oof_matrix = np.column_stack([oof_predictions[name] for name in model_order])
    meta_model = RidgeCV(alphas=np.logspace(-6, 4, 11))
    meta_model.fit(oof_matrix, y_train)
    oof_raw = meta_model.predict(oof_matrix)

    base_calibrations: Dict[str, CalibrationResult] = {}
    for name, preds in oof_predictions.items():
        base_calibrations[name] = calibrate_positions(solution_df, preds, label=f"{name} (OOF)")

    rule_name = "lag_momentum"
    if "lagged_forward_returns" in train_df.columns:
        lag_train_signal = np.where(
            train_df["lagged_forward_returns"].fillna(0.0).to_numpy(dtype=np.float64) > 0,
            MAX_INVESTMENT,
            MIN_INVESTMENT,
        )
    else:
        lag_train_signal = np.zeros(len(train_df), dtype=np.float64)
    oof_predictions[rule_name] = lag_train_signal
    base_calibrations[rule_name] = calibrate_positions(solution_df, lag_train_signal, label=f"{rule_name} (OOF)")

    optimizer_inputs = dict(oof_predictions)
    optimizer_inputs["ridge"] = oof_raw
    (
        best_weights,
        best_params,
        best_score,
        best_vol,
        trial_records,
        initial_weights,
    ) = optimize_blend_and_calibration(
        optimizer_inputs, solution_df, n_trials=args.optimizer_trials, seed=args.optimizer_seed
    )
    logging.info(
        "Optimizer best | Sharpe %.4f | vol_ratio %.3f | weights=%s | params=%s",
        best_score,
        best_vol,
        best_weights,
        best_params,
    )
    combined_oof = blend_predictions(optimizer_inputs, best_weights)
    final_oof_positions = positions_from_params(combined_oof, best_params)
    final_oof_score = adjusted_sharpe(solution_df, final_oof_positions)
    final_oof_vol = compute_volatility_ratio(solution_df, final_oof_positions)

    base_models, test_predictions = train_full_models(X_train, y_train, X_test, model_builders)
    test_matrix = np.column_stack([test_predictions[name] for name in model_order])
    test_predictions["ridge"] = meta_model.predict(test_matrix)
    if "lagged_forward_returns" in test_df.columns:
        lag_test_signal = np.where(
            test_df["lagged_forward_returns"].fillna(0.0).to_numpy(dtype=np.float64) > 0,
            MAX_INVESTMENT,
            MIN_INVESTMENT,
        )
    else:
        lag_test_signal = np.zeros(len(test_df), dtype=np.float64)
    test_predictions[rule_name] = lag_test_signal
    combined_test = blend_predictions(test_predictions, best_weights)
    test_positions = positions_from_params(combined_test, best_params)

    save_linear_coefficients(base_models, feature_cols, args.output_dir)

    submission = build_submission(test_df, sample_df, test_positions)
    submission_path = args.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    logging.info("Saved submission to %s", submission_path)

    metrics_payload = {
        "oof_adjusted_sharpe": final_oof_score,
        "oof_volatility_ratio": final_oof_vol,
        "calibration_params": best_params,
        "base_models": {
            name: {
                "score": calib.score,
                "volatility_ratio": calib.volatility_ratio,
                "params": calib.params,
            }
            for name, calib in base_calibrations.items()
        },
        "optimizer_best": {
            "weights": best_weights,
            "params": best_params,
            "score": best_score,
            "volatility_ratio": best_vol,
            "trials_recorded": len(trial_records),
            "requested_trials": args.optimizer_trials,
            "optimizer_seed": args.optimizer_seed,
            "initial_weights": initial_weights,
        },
        "fold_metrics": fold_metrics,
    }
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)
    logging.info("Stored metrics to %s", metrics_path)

    optimizer_log = {
        "best_weights": best_weights,
        "best_params": best_params,
        "best_score": best_score,
        "best_volatility_ratio": best_vol,
        "requested_trials": args.optimizer_trials,
        "optimizer_seed": args.optimizer_seed,
        "initial_weights": initial_weights,
        "top_trials": sorted(trial_records, key=lambda x: x["score"], reverse=True)[:20],
    }
    optimizer_path = args.output_dir / "optimizer_results.json"
    with optimizer_path.open("w", encoding="utf-8") as fh:
        json.dump(optimizer_log, fh, indent=2)
    logging.info("Stored optimizer log to %s", optimizer_path)

    if args.generate_plots:
        lgbm_model = base_models.get("lgbm")
        if lgbm_model is None:
            logging.warning("LightGBM model missing; skipping feature importance plot.")
        else:
            generate_diagnostics(
                args.output_dir,
                train_df,
                combined_oof,
                final_oof_positions,
                feature_cols,
                lgbm_model,
            )

    score_check = competition_score(solution_df, pd.DataFrame({"prediction": final_oof_positions}))
    logging.info("OOF adjusted Sharpe (optimizer): %.6f | score_fn: %.6f", best_score, score_check)
    print(f"OOF Adjusted Sharpe: {score_check:.6f}")


if __name__ == "__main__":
    main()
