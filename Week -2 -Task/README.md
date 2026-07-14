# Task 02 — Advanced Data Cleaning & Preprocessing

Month 1 | ML Internship Program | Student: Rasikh

## 📌 Task Overview

Is task mein NYC Airbnb dataset (`AB_NYC_2019.csv`) par ek reusable data cleaning pipeline banayi gayi hai jo:
- Missing values handle karti hai
- Outliers remove karti hai
- Type inference (date conversion) karti hai
- Basic schema validation karti hai

## 📂 Files

- `AB_NYC_2019.csv` — Original dataset
- `colab_code.py` — Google Colab ke liye simple cleaning script
- `app.py` — Streamlit app jo cleaning process ko visually dikhata hai
- `requirements.txt` — Required Python libraries
- `.gitignore` — Unnecessary files ko git se exclude karne ke liye

## 🚀 How to Run (Streamlit App)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 Cleaning Steps

1. `name` aur `host_name` missing values → "Unknown"
2. `reviews_per_month` missing values → 0
3. `last_review` → proper datetime format
4. Price outliers (0 ya >1000) remove
5. Minimum nights > 365 wale rows remove
6. Duplicate rows remove
7. Schema validation check

## 📊 Dataset Source

NYC Airbnb Open Data 2019 (Kaggle)
