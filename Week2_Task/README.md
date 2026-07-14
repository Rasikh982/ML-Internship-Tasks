Task 02 — Advanced Data Cleaning & Preprocessing

Month 1 | ML Internship Program | Student: Rasikh

📌 Task Overview

In this task, a reusable data cleaning pipeline was created for the NYC Airbnb dataset (AB_NYC_2019.csv). The pipeline performs the following operations:

Handles missing values
Removes outliers
Infers data types (date conversion)
Performs basic schema validation
📂 Files
AB_NYC_2019.csv — Original dataset
colab_code.py — Data cleaning script for Google Colab
app.py — Streamlit application that visually demonstrates the data cleaning process
requirements.txt — List of required Python libraries
.gitignore — Excludes unnecessary files from Git tracking
🚀 How to Run the Streamlit App
pip install -r requirements.txt
streamlit run app.py
🧠 Data Cleaning Steps
Replace missing values in name and host_name with "Unknown".
Replace missing values in reviews_per_month with 0.
Convert the last_review column to the proper datetime format.
Remove rows with price outliers (price = 0 or price > 1000).
Remove rows where minimum_nights is greater than 365.
Remove duplicate rows.
Perform basic schema validation to ensure data consistency.
📊 Dataset Source

NYC Airbnb Open Data 2019 (Kaggle)
