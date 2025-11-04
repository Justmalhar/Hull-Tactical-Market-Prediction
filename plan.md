# Hull Tactical Market Prediction – Project Plan

## 1. Goals
- Reproduce or exceed LB ≈ 17.36 using a single-file solution inspired by the top notebook in `reference.md`.
- Deliver a reproducible pipeline that trains locally on the public data, produces a submission CSV, and can be deployed to the Kaggle evaluation server.
- Provide clear documentation, dependency management, logging, and optional visual diagnostics.

## 2. Data Understanding
- Source data: `/kaggle/input/hull-tactical-market-prediction/{train,test}.csv` plus `sample_submission.csv`.
- Target: `forward_returns`, exposure bounds `0 ≤ position ≤ 2`.
- Auxiliary columns: risk-free rate (`risk_free_rate`) and market excess returns (`market_forward_excess_returns`) for volatility control.
- Challenge metric: adjusted Sharpe ratio with penalties for exceeding 1.2× market volatility or underperforming market returns.

## 3. High-Level Approach
1. **Feature preparation**
   - Reuse the signal set emphasized in `reference.md` (E*, I2, P*, S*, etc.) plus additional numeric features after filtering leakage (date_id, lagged targets, flags).
   - Impute missing values with 0, cast to float32 for efficiency, standardize features where helpful.
2. **Modeling**
   - Implement a multi-model ensemble similar to the reference stack:
     - Gradient boosted trees: LightGBM, CatBoost, XGBoost.
     - Linear regularized models (Ridge/ElasticNet) for blending.
     - Additional tree models (RandomForest, ExtraTrees) if compute permits.
   - Use cross-validation by `date_id` rolling folds to respect temporal ordering and tune hyperparameters (learning rates, num_leaves, depth).
3. **Exposure mapping**
   - Transform model predictions of forward returns into allocations via a calibrated sigmoid / piecewise mapping that targets volatility cap (≈1.18× market).
   - Clip predictions to `[0, 2]`.
4. **Ensembling**
   - Weight individual model outputs based on CV performance and align with reference’s near-unity weight on best performer.
   - Optionally fine-tune weights via constrained optimization against out-of-fold predictions and the official metric implementation.

## 4. Evaluation & Diagnostics
- Re-implement the public scoring function locally for validation.
- Log cross-validation Sharpe, volatility, penalties, and final adjusted Sharpe.
- Generate plots (stored under `artifacts/`) for:
  - Distribution of forward returns vs. predicted exposures.
  - Rolling strategy volatility vs. market volatility.
  - Feature importance for main boosters.

## 5. Implementation Deliverables
- `solution.py`
  - CLI flags: `--data-dir`, `--output-dir`, `--generate-plots`.
  - Steps: load data → preprocess → train CV → fit final models → produce submission (`submission.csv`).
  - Logging via `logging` module (INFO pipeline steps, DEBUG metrics).
  - Optional Matplotlib visualizations saved when flag enabled.
- `requirements.txt` with pinned versions for reproducibility.
- `README.md` update with challenge summary and instructions.

## 6. Risks & Mitigations
- **Model overfitting**: use time-aware validation, regularization, and blending.
- **Runtime in Kaggle environment**: keep model count limited, tune hyperparameters for speed, support model persistence to avoid retraining during live inference.
- **Dependency availability**: align versions with Kaggle notebooks (LightGBM, CatBoost, XGBoost, scikit-learn, pandas, polars).

## 7. Stretch Enhancements
- Incorporate lagged engineered features (rolling means, volatility) if time permits.
- Experiment with neural models (TabNet) for incremental gains.
- Evaluate scenario analysis under alternative volatility targets.
- Implement optimizer refinement (e.g., greedy line search after random search) and include rule-based models reported in the top references for additional ensemble diversity.
- Rolling-window features (means/std) are available behind the `--enable-rolling-features` flag; iterate on targeted combinations to improve Sharpe without overwhelming the models.
