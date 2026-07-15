# Task 03 — Feature Engineering Mastery

**Dataset:** House Prices - Advanced Regression Techniques (Kaggle)

## What this project does

This script takes the House Prices dataset and creates new, better features
from the existing columns, then picks the most important ones.

## Steps in the code

1. **Load data** — reads `train.csv`
2. **Handle missing values** — fills empty numbers with median, empty text with "None"
3. **Create new features**
   - `TotalSF` — total square footage of the house
   - `HouseAge` — how old the house was when sold
   - `YearsSinceRemodel` — years since last renovation
   - `TotalBath` — total number of bathrooms
   - `HasGarage`, `HasPool`, `HasBasement` — simple yes/no features
4. **Polynomial features** — squared versions of the two strongest features
   (`OverallQual`, `GrLivArea`)
5. **Binning** — groups `HouseAge` into categories: New, Recent, Mid-age, Old, Very Old
6. **Target encoding** — replaces `Neighborhood` with the average house price for that neighborhood
7. **Feature selection** — finds the most useful features using 3 methods:
   - Mutual Information
   - RFE (Recursive Feature Elimination)
   - Lasso (L1 regularization)
8. **Save result** — saves everything to `train_with_new_features.csv`

## How to run

```
pip install pandas numpy scikit-learn
python feature_engineering.py
```

## Files

- `train.csv` — original dataset
- `feature_engineering.py` — main script
- `train_with_new_features.csv` — output file with new features (created after running)

## Output

The script prints the top 10 most important features according to each
feature selection method, so you can compare them and pick the best ones
for your model.
