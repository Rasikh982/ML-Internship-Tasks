
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    validation_curve,
)
from scipy import stats

data = pd.read_csv("creditcard.csv")
print("Dataset ka size:", data.shape)
print(data["Class"].value_counts())
print("Fraud cases sirf itne percent hain:",
      round(data["Class"].mean() * 100, 3), "%")

fraud = data[data["Class"] == 1]
non_fraud = data[data["Class"] == 0].sample(n=5000, random_state=42)
sample_data = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42)

X = sample_data.drop("Class", axis=1)
y = sample_data["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="f1")

print("\nStratified K-Fold F1 Scores (5 folds):", scores)
print("Average F1 Score:", scores.mean(), "  Std Dev:", scores.std())


train_sizes, train_scores, val_scores = learning_curve(
    model, X_scaled, y, cv=skf, scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
)

plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Training Score")
plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label="Validation Score")
plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend()
plt.savefig("learning_curve.png")
print("\nLearning curve 'learning_curve.png' main save kar di gayi hai.")

param_range = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores_vc, val_scores_vc = validation_curve(
    model, X_scaled, y, param_name="C", param_range=param_range,
    cv=skf, scoring="f1"
)

plt.figure()
plt.plot(param_range, train_scores_vc.mean(axis=1), marker="o", label="Training Score")
plt.plot(param_range, val_scores_vc.mean(axis=1), marker="o", label="Validation Score")
plt.xscale("log")
plt.title("Validation Curve (C parameter)")
plt.xlabel("C (Regularization Strength)")
plt.ylabel("F1 Score")
plt.legend()
plt.savefig("validation_curve.png")
print("Validation curve 'validation_curve.png' main save kar di gayi hai.")

model_lr = LogisticRegression(class_weight="balanced", max_iter=1000)
model_rf = RandomForestClassifier(class_weight="balanced", random_state=42)

scores_lr = cross_val_score(model_lr, X_scaled, y, cv=skf, scoring="f1")
scores_rf = cross_val_score(model_rf, X_scaled, y, cv=skf, scoring="f1")

print("\nLogistic Regression F1 Scores:", scores_lr)
print("Random Forest F1 Scores:", scores_rf)

t_stat, p_value = stats.ttest_rel(scores_lr, scores_rf)
print(f"\nPaired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

if p_value < 0.05:
    print("Result: Difference statistically significant hai (p < 0.05).")
else:
    print("Result: Difference statistically significant NAHI hai (p >= 0.05).")
