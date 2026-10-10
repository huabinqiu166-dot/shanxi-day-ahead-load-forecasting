# Data and Evaluation Protocol

## Forecasting Target

For target day D:

Input:
information available no later than D-1

Output:
D 00:15
D 00:30
...
D 23:45
D+1 00:00

There are 96 prediction points per target day.

## Information Boundary

Future realized load values are never used during prediction.

Historical load variables must come strictly from the permitted historical
period.

## Validity Mask

Evaluation is performed only when:

- load_valid == 1
- actual load is finite
- actual load > 0
- prediction is finite

Filled abnormal points may be used as inputs but are excluded from scoring.

## 2025 Validation

Fold1:

Train <= 2025-03-31

Validation:
2025-04-01 to 2025-06-30

Fold2:

Train <= 2025-06-30

Validation:
2025-07-01 to 2025-09-30

Fold3:

Train <= 2025-09-30

Validation:
2025-10-01 to 2025-12-31

## 2026 Rolling Evaluation

Final test period:

2026-01-01 to 2026-08-20

For each month:

1. initialize a fresh model;
2. train using permitted history through the previous month;
3. freeze the model;
4. predict the current month;
5. continue to the next month.

No 2026 performance is used for hyperparameter tuning.

## Metrics

Primary:

MAPE

Additional:

- WMAPE
- MAE
- RMSE

Monthly Mean MAPE is the arithmetic mean of monthly MAPE values.

Pooled MAPE is calculated over all valid points together.
