import streamlit as st
import pandas as pd

st.title("🧹 NYC Airbnb Data Cleaning App")
st.write("Task 02 - Advanced Data Cleaning & Preprocessing")

# Data load karo
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "AB_NYC_2019.csv"))

st.subheader("Original Data")
st.write("Shape:", df.shape)
st.dataframe(df.head())

st.subheader("Missing Values (Before Cleaning)")
st.write(df.isnull().sum())

# ---- Cleaning Steps ----
df["name"] = df["name"].fillna("Unknown")
df["host_name"] = df["host_name"].fillna("Unknown")
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

df_clean = df[(df["price"] > 0) & (df["price"] < 1000)]
df_clean = df_clean[df_clean["minimum_nights"] <= 365]
df_clean = df_clean.drop_duplicates()

st.subheader("Cleaned Data")
st.write("Shape:", df_clean.shape)
st.dataframe(df_clean.head())

st.subheader("Missing Values (After Cleaning)")
st.write(df_clean.isnull().sum())

# Download button
csv = df_clean.to_csv(index=False).encode("utf-8")
st.download_button("Download Cleaned CSV", csv, "AB_NYC_2019_cleaned.csv", "text/csv")
