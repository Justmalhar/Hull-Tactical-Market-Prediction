# Hull Tactical Market Prediction

This repository contains a reproducible pipeline for the [Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction) challenge.  
The goal is to predict the optimal daily allocation to the S&P 500 (between 0 and 2x leverage) so that the resulting strategy maximizes the competition’s adjusted Sharpe ratio while respecting the 120 % volatility ceiling.

## Repository Structure
- `challenge.md` – original competition description.
- `reference.md` – notes from a high-scoring public Kaggle notebook.
- `plan.md` – implementation plan derived from the reference approach.
- `solution.py` – end-to-end training and inference script that produces a submission CSV.
- `requirements.txt` – Python dependencies for local execution.
- `kaggle/` – raw competition data (mounted in the same layout as on Kaggle).

## Environment Setup
1. Create and activate a Python environment (Python ≥ 3.9 recommended).
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   LightGBM, CatBoost, and XGBoost are the heavy dependencies; make sure a compiler toolchain is available if wheels are not provided for your platform.

## Running the Pipeline
```bash
python3 solution.py \
  --data-dir kaggle/input/hull-tactical-market-prediction \
  --output-dir outputs \
  --generate-plots \
  --enable-rolling-features
```

- `--data-dir` must point to a folder containing `train.csv` and `test.csv` exactly as provided by Kaggle.
- `--output-dir` defaults to `outputs/` and stores:
  - `submission.csv` – allocations for every record in the public test set.
  - `metrics.json` – cross-validation diagnostics, calibration settings, and per-model stats.
  - `plots/` *(optional)* – diagnostic figures (predicted vs. actual returns, rolling volatility, feature importances).
- `--optimizer-trials` / `--optimizer-seed` control the random search that finds the final ensemble weights and calibration parameters subject to the volatility cap.
- `--enable-rolling-features` optionally augments the curated feature set with 5/21 day rolling means/stds for selected columns. Leave this flag off if the additional signals degrade performance on your setup.
- Use `--log-level DEBUG` for more granular progress logs.
- The ensemble now blends LightGBM, XGBoost, CatBoost, two ElasticNet variants, a simple lag-momentum rule, and a RidgeCV meta-model that stacks their out-of-fold predictions.
- Exposure calibration explicitly targets ≈1.05× market volatility before clipping to the `[0, 2]` bounds, so the strategy uses the competition’s full volatility budget by default.
- Extra artifacts under `outputs/`:
  - `optimizer_results.json` – best weights/parameters plus the top 20 trials from the random search.
  - `linear_coefficients.json` – recovered real-space coefficients for each linear pipeline in the stack for auditability.

## Submission Guidance
- The generated `submission.csv` mirrors the public leaderboard format (date_id + prediction). Compress and upload it via Kaggle’s “Submit Predictions” interface for offline scoring.
- For the private evaluation server that streams unseen dates, you can embed the training logic from `solution.py` inside a `predict(test: pl.DataFrame)` function (see `kaggle_evaluation` demo in `challenge.md`). Persisting trained models and reusing the calibration parameters ensures deterministic behaviour.

## Next Steps
- Review `plan.md` for roadmap ideas such as additional feature engineering, alternative learners, or refined exposure mappings.
- Extend `solution.py` with richer diagnostics (SHAP, portfolio analytics) or automated hyper-parameter search to chase higher leaderboard scores.

Happy modeling!
- `kaggle_submission.py` – inference wrapper compatible with the competition evaluation API; it reuses the training pipeline and serves predictions via `predict(test: pl.DataFrame)`.
