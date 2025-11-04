# Running Hull Tactical Submission

## Local Validation
```bash
python3 solution.py \
  --data-dir kaggle/input/hull-tactical-market-prediction \
  --output-dir outputs \
  --optimizer-trials 800 \
  --optimizer-seed 42 \
  --generate-plots
```
Optional rolling features:
```bash
python3 solution.py \
  --data-dir kaggle/input/hull-tactical-market-prediction \
  --output-dir outputs \
  --optimizer-trials 800 \
  --optimizer-seed 42 \
  --generate-plots \
  --enable-rolling-features
```

## Kaggle Submission
Upload the project files to Kaggle:
```bash
cd /d/justmalhar/hull-market-prediction
kaggle kernels push --path solution.py
kaggle kernels push --path kaggle_submission.py
```
Run inference:
```bash
python3 kaggle_submission.py
```
