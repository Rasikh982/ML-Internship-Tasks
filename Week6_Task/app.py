
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("diabetes.csv")

print("Dataset ka size:", data.shape)
print(data.head())

# X = input features, y = target (Outcome column)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ---------------------------
# Step 2: Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Step 3: Data Scale Karo
# ---------------------------
# Logistic Regression jaise models scale hui data pe better perform karte hain
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# Step 4 & 5: Model Train Karo
# ---------------------------
# class_weight='balanced' -> for class imbalance handle
model = LogisticRegression(class_weight="balanced")
model.fit(X_train_scaled, y_train)

# ---------------------------
# Step 6: Predictions aur Results
# ---------------------------
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

sample = [[2, 120, 70, 30, 80, 25.5, 0.5, 30]]
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("\nSample Prediction (0 = No Diabetes, 1 = Diabetes):", prediction[0])
