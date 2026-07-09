# Task 01 - Exploratory Data Analysis Deep Dive

Simple Streamlit web app to explore the **Telco Customer Churn** dataset from Kaggle.

## What this app does
- Shows the dataset preview
- Shows total customers, churned customers, and churn rate
- Shows distribution of numeric columns (tenure, MonthlyCharges, TotalCharges)
- Shows churn count by category (gender, contract, internet service, etc.)
- Shows a correlation heatmap

## Files
- `app.py` — main Streamlit app
- `requirements.txt` — Python libraries needed
- `Telco-Customer-Churn.csv` — dataset
- `.gitignore` — files to ignore in git

## How to run
1. Install the requirements:
```
pip install -r requirements.txt
```
2. Run the app:
```
streamlit run app.py
```
3. It will open in your browser automatically.

## Dataset
[Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Author
Rasikh — BS Computer Science / AI, IBADAT International University
