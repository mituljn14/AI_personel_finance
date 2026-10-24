# AI Personal Finance Forecasting

This project cleans and validates a synthetic personal-finance dataset, creates
chronological train/validation/test splits, compares forecasting models, and
saves the best model for predicting next-month net expenses.

## Data

The dataset is synthetic and contains no real user financial information. The
source files are in `data/extracted/`; quality-checked source transactions and
forecast experiments are in `data/processed/`. Model-ready forecast splits,
metrics, predictions, and the serialized model are in `outputs/`.

Net expenses are expenses minus linked refunds. A negative monthly net expense
is valid when refunds exceed expenses; such records are retained.

## Run the selected-model experiment

```powershell
python -m pip install -r requirements.txt
python train_best_forecast_model.py
```

The script validates forecast targets against the monthly accounting summary,
checks chronological splits and data quality, compares candidate models on
2024 validation data, refits the selected model on data through 2024, and
evaluates it once on the held-out 2025 test split.

## Current results

| Evaluation | Model | MAE (INR) | RMSE (INR) |
|---|---|---:|---:|
| Validation (2024) | HistGradientBoosting, absolute-error loss | 17,263.73 | 33,675.82 |
| Test (2025) | HistGradientBoosting, absolute-error loss | 18,304.70 | 36,066.34 |
| Test (2025) | 12-month historical-average baseline | 19,519.05 | 35,571.92 |

The selected model has lower test MAE than the historical-average baseline.
The results measure performance on this synthetic dataset and are not a
guarantee of accuracy on real household finances.
