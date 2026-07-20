# Insurance Charges Prediction — Regression Models

This is a simple project that predicts **medical insurance charges** for a
person, based on things like their age, BMI, number of children, whether
they smoke, and where they live.

It covers all the key skills:
- Linear Regression (and checking its assumptions)
- Ridge, Lasso, and ElasticNet Regression
- Polynomial Regression
- Residual Analysis
- Cross-Validation
- Model Comparison

## Files in this folder

| File | What it is |
|---|---|
| `insurance.csv` | The dataset (1338 people's insurance info) |
| `regression_models.py` | The main code — run this file |
| `model_comparison_results.csv` | Output: a table comparing all models (created after you run the code) |
| `residual_analysis.png` | Output: 3 diagnostic plots (created after you run the code) |

## About the dataset

Each row is one person. The columns are:

- **age** — age of the person
- **sex** — male or female
- **bmi** — body mass index (a measure of body fat)
- **children** — number of children/dependents covered
- **smoker** — yes or no
- **region** — where they live (northeast, northwest, southeast, southwest)
- **charges** — the actual medical cost billed (this is what we are trying to predict)

## How to run it

1. Make sure you have Python installed, along with these packages:
   ```
   pip install pandas numpy scikit-learn matplotlib
   ```
2. Put `insurance.csv` and `regression_models.py` in the same folder.
3. Run:
   ```
   python regression_models.py
   ```
4. Watch the results print in your terminal. Two new files will also be
   created in the same folder:
   - `model_comparison_results.csv`
   - `residual_analysis.png`

## What the code actually does (step by step)

1. **Loads the data** from `insurance.csv`.
2. **Prepares the data**: turns text columns (sex, smoker, region) into
   numbers, since models only understand numbers. This is called
   "one-hot encoding."
3. **Splits the data**: 80% is used to train the models, 20% is kept aside
   to test how good the models are on data they haven't seen.
4. **Scales the numbers**: so that big numbers (like age) don't unfairly
   dominate small numbers.
5. **Trains 5 different models**:
   - **Linear Regression** — draws the simplest straight-line relationship
     between the inputs and charges.
   - **Ridge Regression** — like linear regression, but it shrinks big
     coefficients to avoid overfitting.
   - **Lasso Regression** — similar to Ridge, but it can shrink some
     coefficients all the way to zero (useful for picking important features).
   - **ElasticNet Regression** — a mix of Ridge and Lasso.
   - **Polynomial Regression** — allows curved (non-straight-line)
     relationships, which usually fits this dataset better.
6. **Cross-validation**: instead of testing the model only once, it's tested
   5 times on different slices of the training data, and the scores are
   averaged. This gives a more trustworthy idea of how well the model
   performs.
7. **Model comparison table**: prints and saves a table with these columns
   for every model:
   - **RMSE** — average prediction error, in dollars (lower is better)
   - **MAE** — average absolute prediction error, in dollars (lower is better)
   - **R2 (test set)** — how much of the variation in charges the model
     explains, from 0 to 1 (higher is better)
   - **R2 (5-fold CV average)** — same as above, but averaged over 5 tests
     (more reliable than a single test)
8. **Residual analysis** (for Linear Regression): a residual is simply
   `actual charge - predicted charge`. Three plots are saved to
   `residual_analysis.png`:
   - **Residuals vs Predicted** — checks if errors stay roughly the same
     size across all predictions. If you see a fan/cone shape, that means
     one of the linear regression assumptions is being broken.
   - **Histogram of Residuals** — checks if the errors look roughly like a
     bell curve. This is another linear regression assumption.
   - **Actual vs Predicted** — shows how close predictions are to reality.
     A perfect model would have all points on the red diagonal line.

## What you'll likely notice in the results

- **Linear, Ridge, and Lasso** perform very similarly to each other.
- **ElasticNet** tends to do a bit worse here (its default settings shrink
  coefficients quite a lot).
- **Polynomial Regression** usually performs the best, because the real
  relationship between age/BMI/smoking and charges isn't a straight line —
  for example, smokers' charges jump up much faster than non-smokers'.

## Where to go from here (optional ideas)

- Try different `alpha` values for Ridge/Lasso/ElasticNet to see if they
  improve.
- Try a higher polynomial degree (e.g. degree=3) — but be careful, too high
  a degree can overfit.
- Try other models like Random Forest or Gradient Boosting for comparison.
