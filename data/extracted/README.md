# AI Personal Finance and Smart Expense Predictor - synthetic dataset (Team 3, WeIntern)

Synthetic, reproducible (seed 42) personal-finance data: 150 anonymous users, 2022-01-01 to 2025-12-31, INR. **No real data.**

| File | Contents |
|---|---|
| transactions.csv | 67,945 rows; income / expense / refund / transfer |
| users.csv | 150 profiles |
| monthly_budgets.csv | 66,672 category budgets |
| recurring_transactions.csv | 3,208 recurring-pattern segments (drive the recurring rows in transactions.csv) |
| monthly_financial_summary.csv | 7,200 user-months, reconciled to transactions |
| monthly_expense_forecasting.csv | 7,050 rows, leakage-safe features, target = next-month net expenses, `split` column |
| data_dictionary.csv | every column of every file |
| dataset_validation_report.txt | integrity, reconciliation, distributions, ML results, limitations |
| dataset_generation_methodology.md | rules, formulas, definitions |
| generate_dataset.py | full generator (`python generate_dataset.py --out DIR [--no-ml]`) |

Quick start:
```python
import pandas as pd
tx = pd.read_csv("transactions.csv", parse_dates=["transaction_date"])
done = tx[tx.payment_status == "completed"]          # always exclude failed rows
fc = pd.read_csv("monthly_expense_forecasting.csv")
train, val, test = [fc[fc.split == s] for s in ("train", "validation", "test")]
```
Key rules: only `completed` rows count; transfers are never income/expense; refunds reduce spend (`net_expenses`); `month_end_balance` is not net worth.
Forecast experiment (time-aware split; see report): baselines vs Linear Regression vs Random Forest. Synthetic-data accuracy is not real-world accuracy.

Forecast results snapshot:
```
Features: 64 columns (numeric + one-hot employment_type). Split by TARGET month: train = targets in 2022-2023 (3450 rows), validation = 2024 (1800), test = 2025 (1800). No shuffling.
MAPE is computed only on rows with target >= INR 1,000 (otherwise it is unstable). Models predict next-month NET expenses (expenses - linked refunds; transfers excluded).
VALIDATION 2024 (models fit on 2022-2023 targets only)
model                                                   MAE (INR)   RMSE (INR)    MAPE %       n
Baseline: last month (naive)                                23854        42646      77.0    1800
Baseline: 12-month historical average                       18052        32780      63.1    1800
Baseline: seasonal naive (same month last year)             22441        40804      71.9    1800
LinearRegression                                            18145        33139      65.6    1800
RandomForest                                                18082        33171      67.0    1800
TEST 2025 (models refit on 2022-2024 targets; test never used for fitting)
model                                                   MAE (INR)   RMSE (INR)    MAPE %       n
Baseline: last month (naive)                                26541        48356      82.7    1800
Baseline: 12-month historical average                       19519        35572      65.4    1800
Baseline: seasonal naive (same month last year)             25145        47054      77.1    1800
LinearRegression                                            19900        35131      72.7    1800
RandomForest                                                19533        34691      71.1    1800
Top-10 RandomForest feature importances (fit on train): budget_total_target=0.394; monthly_income_estimate=0.119; income_roll_mean_12=0.070; exp_roll_mean_12=0.036; expected_recurring_target=0.028; budget_total_m0=0.018; exp_roll_mean_6=0.017; exp_roll_std_6=0.014; essential_exp_m0=0.013; exp_lag2=0.013
Seasonal naive is evaluated on rows where the same month of the previous year exists (all validation/test rows qualify).
Interpretation: accuracy on synthetic data reflects the generating rules (smooth trends, bounded noise, known recurring bills) and says little about performance on real household data, which has regime changes, one-off shocks and missing records.
```
