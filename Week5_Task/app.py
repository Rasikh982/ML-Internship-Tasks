
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------
# STEP 1: Load the data
# ---------------------------------------------------------
df = pd.read_csv("insurance.csv")
print("First 5 rows of data:")
print(df.head())
print("\nData shape:", df.shape)

# ---------------------------------------------------------
# STEP 2: Prepare the data
# ---------------------------------------------------------
# Convert text columns (sex, smoker, region) into numbers using one-hot encoding
df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

# X = input features, y = the thing we want to predict (charges)
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the numbers so all features are on a similar scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# STEP 3: Helper function to evaluate any model
# ---------------------------------------------------------
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, results_list):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)

    # 5-fold cross-validation score (using R2)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2")

    print(f"\n--- {name} ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R2 (test set): {r2:.4f}")
    print(f"R2 (5-fold CV average): {cv_scores.mean():.4f}")

    results_list.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2_test": r2,
        "R2_CV_avg": cv_scores.mean(),
    })
    return model, y_pred


results = []

# ---------------------------------------------------------
# STEP 4: Linear Regression
# ---------------------------------------------------------
linreg = LinearRegression()
linreg, y_pred_lin = evaluate_model(
    "Linear Regression", linreg, X_train_scaled, X_test_scaled, y_train, y_test, results
)

# ---------------------------------------------------------
# STEP 5: Ridge, Lasso, ElasticNet
# ---------------------------------------------------------
ridge = Ridge(alpha=1.0)
evaluate_model("Ridge Regression", ridge, X_train_scaled, X_test_scaled, y_train, y_test, results)

lasso = Lasso(alpha=1.0)
evaluate_model("Lasso Regression", lasso, X_train_scaled, X_test_scaled, y_train, y_test, results)

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
evaluate_model("ElasticNet Regression", elastic, X_train_scaled, X_test_scaled, y_train, y_test, results)

# ---------------------------------------------------------
# STEP 6: Polynomial Regression (degree 2)
# ---------------------------------------------------------
poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    LinearRegression()
)
# Polynomial model gets its own pipeline, so we feed it the raw (unscaled) split data
evaluate_model(
    "Polynomial Regression (deg=2)", poly_model, X_train, X_test, y_train, y_test, results
)

# ---------------------------------------------------------
# STEP 7: Model Comparison Table
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n\n=========== MODEL COMPARISON TABLE ===========")
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison_results.csv", index=False)
print("\nSaved comparison table to model_comparison_results.csv")

# ---------------------------------------------------------
# STEP 8: Residual Analysis (for Linear Regression)
# ---------------------------------------------------------
residuals = y_test - y_pred_lin

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Residuals vs Predicted (checks constant spread - "homoscedasticity")
axes[0].scatter(y_pred_lin, residuals, alpha=0.5)
axes[0].axhline(y=0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted charges")
axes[0].set_ylabel("Residual (actual - predicted)")
axes[0].set_title("Residuals vs Predicted")

# 2. Histogram of residuals (checks if errors look roughly bell-shaped/normal)
axes[1].hist(residuals, bins=30, edgecolor="black")
axes[1].set_xlabel("Residual")
axes[1].set_title("Histogram of Residuals")

# 3. Actual vs Predicted (checks overall fit)
axes[2].scatter(y_test, y_pred_lin, alpha=0.5)
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
axes[2].set_xlabel("Actual charges")
axes[2].set_ylabel("Predicted charges")
axes[2].set_title("Actual vs Predicted")

plt.tight_layout()
plt.savefig("residual_analysis.png", dpi=120)
print("Saved residual plots to residual_analysis.png")

print("\nDone! Check the printed table above and the two output files.")
