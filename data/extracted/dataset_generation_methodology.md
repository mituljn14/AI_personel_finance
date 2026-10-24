# Dataset generation methodology

**Project:** AI Personal Finance and Smart Expense Predictor (Team 3, WeIntern minor project)
**Everything here is synthetic.** No real person, account, card, merchant or transaction is represented; merchant names are fictional.
Seed 42; run `python generate_dataset.py` to reproduce byte-identical CSVs (same library versions).

## 1. Scope
150 users, 48 months (2022-01-01 to 2025-12-31), INR. Rows in transactions.csv: **67,945** (306 failed attempts, 148 refunds).

## 2. Generation pipeline
1. **Users**: employment mix (80 salaried, 20 freelancer, 18 business owner, 10 gig, 10 student, 12 retired), age by employment, city/state (tier-weighted), household size, income (log-normal around occupation medians), one of four behaviour archetypes (disciplined_saver, balanced, overspender, inconsistent) that sets the spend-to-income ratio, saving fraction, noise level, stability and budget bias.
2. **Recurring patterns first** (salary/pension/allowance, rent, maintenance, electricity, water, gas, internet, mobile, subscriptions, insurance, loan EMIs, tuition, SIP/RD/PPF). Each pattern is a schedule with a start month, frequency, due day, and annual amount steps (e.g. April rent/salary revisions). Transactions for these are produced **only from the schedule**, so recurring_transactions.csv and transactions.csv cannot disagree or duplicate.
3. **Variable spending** per subcategory: monthly expected amount = user pool x taste weight x inflation (bounded annual rates, e.g. food 5.5%, electronics -2%) x income-lifestyle elasticity x seasonal factor x AR(1) shock (stronger for discretionary than essential spend). The count of entries per month follows a stability-controlled Poisson; amounts are log-normal.
4. **Seasonality (category-specific only)**: festival shopping (clothing, electronics, gifts, household; state-specific regional festival plus Diwali), travel peaks (Apr-Jun, Dec), monsoon health/taxi, academic fees (Jun-Jul, Jan), electricity by climate zone, business-income seasonality for seasonal owners.
5. **Shocks and drift**: 2.5-5% monthly chance of an unexpected large expense, 15% of users have a lifestyle shift between months 12-36, 18% of salaried users change employer with a raise, bonus/incentive income for some.
6. **Failed payments and refunds**: ~0.7% of digital expenses get a preceding `failed` attempt (excluded from totals); selected shopping/travel expenses (not cash) get a refund (35% partial) linked via `related_transaction_id`.
7. **Budgets**: per user one style (fixed_monthly 45%, seasonal_adjusted 35%, income_percentage 20%) x archetype bias (overspenders under-budget, savers pad); only some categories are budgeted; amounts are rounded and never identical across users.
8. **Assembly**: exact-duplicate detection (removed: 0), UPI > INR 1 lakh re-labelled as bank transfer (26 rows), ids assigned after sorting by date/time, derived calendar fields computed from the date.

## 3. Taxonomy
Expense categories (12): Housing (rent, maintenance, home_repairs), Food (groceries, restaurants, food_delivery), Transportation (public_transport, fuel, taxi, vehicle_maintenance, vehicle_insurance), Utilities (electricity, water, gas, internet, mobile_recharge), Healthcare (consultation, medicine, diagnostic_tests, insurance), Education (tuition, books, online_courses), Shopping (clothing, electronics, household_goods), Entertainment (movies, subscriptions, events), Personal Care (grooming, toiletries), Travel (train_bus_ticket, flights, hotel_stay), Gifts & Donations (gifts, donations), Financial Expenses (bank_fees, loan_payment, interest_charges).
Income: salary, freelance_work, business_income, incentives, other_income. Savings and Investments (transfers): sip_mutual_fund, recurring_deposit, ppf_contribution, fixed_deposit, savings_withdrawal. Transfers: own_account_transfer, atm_cash_withdrawal.

## 4. Accounting definitions (used in every derived table)
- Only `payment_status = completed` rows count. Failed rows are ignored everywhere.
- **Transfers are neither income nor expense.** Own-account moves, ATM withdrawals (cash expenses are the spend) and savings contributions/redemptions are excluded from income and expenses. Savings contributions are reported in `savings_amount`.
- **Refunds** reduce eligible spend: `net_expenses = total_expenses - refunds_received` (refund carries the original's category, so category net spend is also net of refunds). Refunds are never income.
- `net_cash_flow = total_income - net_expenses`.
- `savings_amount = SIP/RD/PPF/FD contributions - redemptions`; `savings_rate = savings_amount / total_income` (blank when income = 0).
- `essential_expenses` = expenses flagged is_essential (needs); `discretionary_expenses = total_expenses - essential_expenses` (gross, before refunds).
- Budget metrics compare **net category spend** with the category budget; `budget_utilization_percentage = 100 x net spend in budgeted categories / total_budget`.
- `month_end_balance = opening_balance_estimate + cumulative(total_income - net_expenses - savings_amount)` is a liquid-balance tracker, **not net worth** (no assets, liabilities or card dues are modelled).
- Financial year follows the Indian April-March convention; `quarter` is the calendar quarter.

## 5. Missing values
Blank = intentional and documented: unknown time on manual cash entries, unknown merchant on informal cash payments, non-applicable recurring/refund link fields, active recurring patterns (no end date), undefined ratios (zero income / no budget / no expenses), and lag features without enough history. Nothing is auto-imputed in the files; the validation report lists every count.

## 6. Forecasting table (`monthly_expense_forecasting.csv`)
- One row per user per prediction month t (Jan 2022 - Nov 2025); **target = net expenses in month t+1**.
- Features use only data through the end of month t: current/lagged/rolling net expenses, category-level net spend, income, savings, transaction counts, budgets (the target month's budget is set before that month begins), expected recurring bills due next month (schedules are known in advance), and calendar-only seasonal indicators of the target month.
- No future expenses, future counts or target-derived values are used as features. `target_*` columns are calendar facts only.
- **Split by target month, never random**: train = 2022-2023, validation = 2024, test = 2025. Early-test rows legitimately use 2024 history.
- Baselines: last month, 12-month historical average, seasonal naive (same month last year). Models: Linear Regression and Random Forest (median imputation; scaling for linear). Metrics: MAE, RMSE, MAPE (targets >= INR 1,000). Results are in dataset_validation_report.txt section 6. Test results come from models refit on train+validation; the test period is never used for fitting.
- Category-level forecasting: category net spend columns for every user-month are included; build per-category targets by shifting those columns by -1 within user.

## 7. Documented unusual-spending screen (rule-based, not a fraud label)
For a user-month with at least 12 months of history, flag `unusually_high` when `net_expenses > rolling_mean_12 + 3 x rolling_std_12` (computed on prior 12 months only). No ground-truth anomaly or fraud label is provided because no defensible labelling process exists for synthetic data; injected shocks (`planned_or_unplanned = unplanned`, descriptions starting 'Unexpected:') can be used to sanity-check recall but are not guaranteed anomalies.

## 8. Analysis hints
Income vs expenses: monthly_financial_summary. Category trends: transactions grouped by category/month. Essential vs discretionary: is_essential or needs_or_wants. Repeated overruns: categories_over_budget by user. Recurring vs non-recurring: is_recurring. Income variability: std/mean of total_income per user. Festival spending: target_festival_index / transaction dates around the festival windows in the script (`FESTIVALS`). Average-ticket changes: average_transaction_amount over time.

## 9. Limitations
Synthetic rules, not real behaviour; sparse logging of small purchases because volume was calibrated to 30-50k rows; simplified festival calendar; no net-worth modelling; model accuracy here does not transfer to real data. See validation report section 8.
