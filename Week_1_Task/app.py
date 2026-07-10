import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Find the CSV file (works no matter where the app runs from)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Telco-Customer-Churn.csv")
# Load the data
df = pd.read_csv(csv_path)
# Title
st.title("Telco Customer Churn Dashboard")
# Show the data table
st.write("Here is the data:")
st.dataframe(df.head(10))
# Show how many customers churned
churned = (df["Churn"] == "Yes").sum()
st.write("Customers who left the company (Churn):", churned)
# Show one simple chart
st.write("Chart: Churn by Contract Type")
fig, ax = plt.subplots()
sns.countplot(x="Contract", hue="Churn", data=df, ax=ax)
st.pyplot(fig)
