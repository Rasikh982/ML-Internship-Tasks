"""
Task 03 - Feature Engineering Mastery
Dataset: House Prices - Advanced Regression Techniques (Kaggle)

Simple, beginner-friendly feature engineering script.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import Lasso, LinearRegression

# -------------------------------------------------
# 1. Load the data
# -------------------------------------------------
df = pd.read_csv("train.csv")
print("Original shape:", df.shape)

# -------------------------------------------------
# 2. Handle missing values (simple approach)
# -------------------------------------------------
# Fill numeric columns with median, categorical with "None"
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna("None")

# -------------------------------------------------
# 3. Create new features from existing ones
# -------------------------------------------------
# Total square footage (basement + 1st floor + 2nd floor)
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

# House age when sold
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

# Years since last remodel
df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

# Total number of bathrooms
df["TotalBath"] = (df["FullBath"] + (0.5 * df["HalfBath"]) +
                    df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"]))

# Does the house have a garage / pool / basement (simple yes-no features)
df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
df["HasPool"] = (df["PoolArea"] > 0).astype(int)
df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)

# -------------------------------------------------
# 4. Polynomial features (for the most important numeric column)
# -------------------------------------------------
# OverallQual is one of the strongest predictors of SalePrice
df["OverallQual_sq"] = df["OverallQual"] ** 2
df["GrLivArea_sq"] = df["GrLivArea"] ** 2

# -------------------------------------------------
# 5. Binning (turn a continuous column into categories)
# -------------------------------------------------
df["HouseAge_bin"] = pd.cut(
    df["HouseAge"],
    bins=[-1, 5, 20, 50, 100, 1000],
    labels=["New", "Recent", "Mid-age", "Old", "Very Old"]
)

# -------------------------------------------------
# 6. Target encoding (replace category with mean SalePrice for that category)
# -------------------------------------------------
neighborhood_means = df.groupby("Neighborhood")["SalePrice"].mean()
df["Neighborhood_encoded"] = df["Neighborhood"].map(neighborhood_means)

# -------------------------------------------------
# 7. Feature selection
# -------------------------------------------------
# Only use numeric columns for feature selection
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["Id", "SalePrice"])
X = numeric_df
y = df["SalePrice"]

# --- Mutual Information ---
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("\nTop 10 features by Mutual Information:")
print(mi_series.head(10))

# --- RFE (Recursive Feature Elimination) ---
model = LinearRegression()
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X, y)
rfe_selected = X.columns[rfe.support_]
print("\nTop 10 features selected by RFE:")
print(list(rfe_selected))

# --- L1 (Lasso) feature selection ---
lasso = Lasso(alpha=100, random_state=42, max_iter=10000)
lasso.fit(X, y)
lasso_scores = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
print("\nTop 10 features by Lasso (L1):")
print(lasso_scores.head(10))

# -------------------------------------------------
# 8. Save the new dataset with engineered features
# -------------------------------------------------
df.to_csv("train_with_new_features.csv", index=False)
print("\nDone! New file saved as train_with_new_features.csv")
print("Final shape:", df.shape)
