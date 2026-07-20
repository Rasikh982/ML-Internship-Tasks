# Credit Card Fraud Detection - Cross Validation App

This project uses a credit card fraud dataset to demonstrate different
**cross-validation** techniques used in machine learning.

## Files

- `app.py` - main code file
- `creditcard.csv` - dataset used
- `README.md` - this file

When you run the code, it will also create these output files:
- `learning_curve.png` - shows training vs validation score as data grows
- `validation_curve.png` - shows effect of changing a hyperparameter

## Dataset

The dataset has 284,807 transactions with 31 columns:
- `Time`, `Amount` - transaction details
- `V1` to `V28` - anonymized numeric features
- `Class` - 0 = normal transaction, 1 = fraud

This dataset is **very imbalanced**: only about 0.17% of transactions are
fraud. This is exactly why techniques like Stratified K-Fold matter here.

## What the code demonstrates

1. **Stratified K-Fold Cross Validation**
   - Normal K-Fold can put very few (or zero) fraud cases in some folds.
   - Stratified K-Fold keeps the same fraud/non-fraud ratio in every fold,
     which gives a fairer evaluation.

2. **Learning Curve (Bias-Variance Tradeoff)**
   - Shows training score vs validation score as training data size grows.
   - If both scores are low → **high bias** (model is too simple / underfitting)
   - If training score is high but validation score is low → **high variance**
     (model is overfitting)

3. **Validation Curve**
   - Shows how changing a hyperparameter (here, `C` in Logistic Regression)
     affects training and validation performance.
   - Helps pick a good hyperparameter value.

4. **Statistical Significance Test**
   - Compares Logistic Regression vs Random Forest using a **paired t-test**
     on their cross-validation scores.
   - A small p-value (< 0.05) means one model is *reliably* better, not just
     better by luck on this particular split.

## Note on speed

The full dataset has ~285,000 rows. To keep the demo fast, the code takes
all fraud cases + a random sample of 5,000 normal cases. If you want to run
it on the full dataset, just remove the sampling step in `app.py`
(look for the comment "Chotta Sample Lo").

## Note on Leave-One-Out and Nested CV

- **Leave-One-Out (LOO)** is not used in the code because it trains one
  model per row, which would take a very long time on this dataset
  (285,000 models!). LOO is better suited for small datasets.
- **Nested CV** (cross-validation inside cross-validation, used for
  hyperparameter tuning + honest evaluation together) is not included here
  to keep the code simple, but you can build it by wrapping `GridSearchCV`
  inside `cross_val_score`.

## How to run

1. Install the required libraries:

```
pip install pandas scikit-learn matplotlib scipy
```

2. Make sure `app.py` and `creditcard.csv` are in the same folder.

3. Run the code:

```
python app.py
```
