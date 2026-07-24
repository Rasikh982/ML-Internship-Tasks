import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("🏥 Insurance Charges Predictor (Light Version)")
st.write("Halka version — kam resources use karta hai.")

file = st.file_uploader("insurance.csv upload karein", type=["csv"])

if file is None:
    st.stop()

df = pd.read_csv(file)
st.write("Data Preview:")
st.dataframe(df.head())

df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("Results")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R2 Score: {r2:.4f}")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
st.pyplot(fig)
