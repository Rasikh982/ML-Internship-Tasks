import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Telco Churn EDA", layout="wide")
st.title("📊 Telco Customer Churn - EDA Dashboard")
st.write("Simple interactive dashboard to explore the Telco Customer Churn dataset.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    # Fix TotalCharges (it has some blank values stored as text)
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
contract_filter = st.sidebar.multiselect(
    "Select Contract Type",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

df_filtered = df[df["Contract"].isin(contract_filter)]

# Basic info
st.subheader("Dataset Preview")
st.dataframe(df_filtered.head(10))

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df_filtered))
col2.metric("Churned Customers", (df_filtered["Churn"] == "Yes").sum())
col3.metric("Churn Rate", f"{(df_filtered['Churn']=='Yes').mean()*100:.1f}%")

# Numeric columns distribution
st.subheader("Numeric Feature Distribution")
num_col = st.selectbox("Choose a numeric column", ["tenure", "MonthlyCharges", "TotalCharges"])

fig, ax = plt.subplots()
sns.histplot(df_filtered[num_col], kde=True, ax=ax)
ax.set_title(f"Distribution of {num_col}")
st.pyplot(fig)

# Churn by category
st.subheader("Churn Rate by Category")
cat_col = st.selectbox(
    "Choose a categorical column",
    ["gender", "SeniorCitizen", "Partner", "Dependents", "InternetService", "PaymentMethod"]
)

fig2, ax2 = plt.subplots()
sns.countplot(x=cat_col, hue="Churn", data=df_filtered, ax=ax2)
ax2.set_title(f"Churn Count by {cat_col}")
st.pyplot(fig2)

# Correlation heatmap
st.subheader("Correlation Heatmap")
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
fig3, ax3 = plt.subplots()
sns.heatmap(df_filtered[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

st.write("---")
st.caption("Task 01 - Exploratory Data Analysis Deep Dive | Telco Customer Churn Dataset")
