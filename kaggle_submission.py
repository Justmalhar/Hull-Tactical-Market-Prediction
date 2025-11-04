#!/usr/bin/env python3
"""
Kaggle inference script for the Hull Tactical Market Prediction challenge.

This script mirrors the training pipeline in solution.py, but packages the
models and calibration parameters for use with the competition evaluation API.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import RidgeCV

import kaggle_evaluation.default_inference_server

import solution


DATA_DIR = Path("/kaggle/input/hull-tactical-market-prediction")
USE_ROLLING = os.getenv("ENABLE_ROLLING_FEATURES", "0") == "1"
OPTIMIZER_TRIALS = int(os.getenv("OPTIMIZER_TRIALS", "600"))
OPTIMIZER_SEED = int(os.getenv("OPTIMIZER_SEED", "42"))

_MODEL_STATE: Optional[Dict[str, object]] = None


def _train_models() -> Dict[str, object]:
    train_df, test_df, sample_df = solution.load_datasets(DATA_DIR)
    feature_cols = solution.select_feature_columns(train_df, test_df)

    if USE_ROLLING:
        train_df, test_df, rolling_features = solution.augment_with_rolling_features(
            train_df, test_df, feature_cols
        )
        if rolling_features:
            feature_cols = feature_cols + rolling_features

    X_train = solution.preprocess_features(train_df, feature_cols)
    X_test = solution.preprocess_features(test_df, feature_cols)
    y_train = train_df["forward_returns"].to_numpy(dtype=np.float64)
    solution_df = train_df[["forward_returns", "risk_free_rate"]]

    model_builders = solution.build_model_builders(solution.RANDOM_STATE)
    oof_predictions, _ = solution.run_time_series_cv(
        X_train, y_train, solution_df, model_builders, n_splits=5
    )

    base_calibrations = {}
    for name, preds in oof_predictions.items():
        base_calibrations[name] = solution.calibrate_positions(
            solution_df, preds, label=f"{name} (OOF)"
        )

    rule_name = "lag_momentum"
    if "lagged_forward_returns" in train_df.columns:
        lag_signal = np.where(
            train_df["lagged_forward_returns"].fillna(0.0).to_numpy(dtype=np.float64) > 0,
            solution.MAX_INVESTMENT,
            solution.MIN_INVESTMENT,
        )
    else:
        lag_signal = np.zeros(len(train_df), dtype=np.float64)
    oof_predictions[rule_name] = lag_signal
    base_calibrations[rule_name] = solution.calibrate_positions(
        solution_df, lag_signal, label=f"{rule_name} (OOF)"
    )

    model_order = list(model_builders.keys())
    oof_matrix = np.column_stack([oof_predictions[name] for name in model_order])
    meta_model = RidgeCV(alphas=np.logspace(-6, 4, 11))
    meta_model.fit(oof_matrix, y_train)
    ridge_oof = meta_model.predict(oof_matrix)

    optimizer_inputs = dict(oof_predictions)
    optimizer_inputs["ridge"] = ridge_oof
    (
        best_weights,
        best_params,
        best_score,
        best_vol,
        _,
        initial_weights,
    ) = solution.optimize_blend_and_calibration(
        optimizer_inputs, solution_df, n_trials=OPTIMIZER_TRIALS, seed=OPTIMIZER_SEED
    )

    base_models, _ = solution.train_full_models(
        X_train, y_train, X_test, model_builders
    )

    state = {
        "feature_cols": feature_cols,
        "best_weights": best_weights,
        "best_params": best_params,
        "base_models": base_models,
        "model_order": model_order,
        "meta_model": meta_model,
        "use_rolling": USE_ROLLING,
        "best_score": best_score,
        "best_vol": best_vol,
        "initial_weights": initial_weights,
        "sample_df": sample_df,
    }
    return state


def _prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        for col in missing:
            df[col] = 0.0
    return solution.preprocess_features(df, feature_cols)


def predict(test: pl.DataFrame) -> pl.DataFrame:
    global _MODEL_STATE
    if _MODEL_STATE is None:
        _MODEL_STATE = _train_models()

    state = _MODEL_STATE
    feature_cols = state["feature_cols"]
    base_models = state["base_models"]
    best_weights = state["best_weights"]
    best_params = state["best_params"]
    model_order = state["model_order"]
    meta_model = state["meta_model"]

    test_pdf = test.to_pandas()
    features = _prepare_features(test_pdf.copy(), feature_cols)

    predictions: Dict[str, np.ndarray] = {}
    for name, model in base_models.items():
        predictions[name] = model.predict(features)

    ridge_matrix = np.column_stack([predictions[name] for name in model_order])
    predictions["ridge"] = meta_model.predict(ridge_matrix)

    if "lagged_forward_returns" in test_pdf.columns:
        lag_signal = np.where(
            test_pdf["lagged_forward_returns"].fillna(0.0).to_numpy(dtype=np.float64) > 0,
            solution.MAX_INVESTMENT,
            solution.MIN_INVESTMENT,
        )
    else:
        lag_signal = np.zeros(len(test_pdf), dtype=np.float64)
    predictions["lag_momentum"] = lag_signal

    combined = solution.blend_predictions(predictions, best_weights)
    positions = solution.positions_from_params(combined, best_params)

    return pl.DataFrame({"prediction": positions})


inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway((str(DATA_DIR),))
