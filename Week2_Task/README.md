# Task 02 --- Advanced Data Cleaning & Preprocessing

**Month 1 \| ML Internship Program \| Student: Rasikh**

## 📌 Task Overview

This project implements a reusable data cleaning pipeline for the **NYC
Airbnb dataset (`AB_NYC_2019.csv`)**. The pipeline:

-   Handles missing values
-   Removes outliers
-   Performs data type inference (date conversion)
-   Validates the dataset schema

## 📂 Files

-   **AB_NYC_2019.csv** --- Original dataset
-   **colab_code.py** --- Data cleaning script for Google Colab
-   **app.py** --- Streamlit application that demonstrates the data
    cleaning process
-   **requirements.txt** --- Required Python libraries
-   **.gitignore** --- Excludes unnecessary files from Git tracking

## 🚀 How to Run

``` bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 Data Cleaning Steps

1.  Replace missing values in `name` and `host_name` with `"Unknown"`.
2.  Replace missing values in `reviews_per_month` with `0`.
3.  Convert `last_review` to the proper datetime format.
4.  Remove rows where `price` is `0` or greater than `1000`.
5.  Remove rows where `minimum_nights` is greater than `365`.
6.  Remove duplicate rows.
7.  Perform basic schema validation.

## 📊 Dataset Source

**NYC Airbnb Open Data 2019 (Kaggle)**

## 👨‍💻 Author

**Rasikh**\
ML Internship Program -- Month 1
