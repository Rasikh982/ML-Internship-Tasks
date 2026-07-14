# Task 02 - Advanced Data Cleaning & Preprocessing
# Dataset: AB_NYC_2019.csv (NYC Airbnb Data)

import pandas as pd
import numpy as np

# Step 1: Dataset load karo
df = pd.read_csv("AB_NYC_2019.csv")

print("Original Shape:", df.shape)
print(df.isnull().sum())

# Step 2: Missing values handle karo
# 'name' aur 'host_name' mein missing hai -> "Unknown" se fill karo
df["name"] = df["name"].fillna("Unknown")
df["host_name"] = df["host_name"].fillna("Unknown")

# 'reviews_per_month' mein missing matlab koi review nahi -> 0 se fill karo
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

# 'last_review' date column hai, missing ko NaT rehne do, ya drop kar do
df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

# Step 3: Type inference / conversion
# 'last_review' already datetime ban chuki hai
# Baaki columns already sahi types mein hain

# Step 4: Outliers handle karo (price aur minimum_nights)
# Price = 0 ya bohot zyada wale rows ajeeb hain
df = df[(df["price"] > 0) & (df["price"] < 1000)]

# Minimum nights bohot zyada (e.g. > 365) wale bhi outliers hain
df = df[df["minimum_nights"] <= 365]

# Step 5: Duplicate rows remove karo
df = df.drop_duplicates()

# Step 6: Schema validation (basic check)
expected_columns = [
    "id", "name", "host_id", "host_name", "neighbourhood_group",
    "neighbourhood", "latitude", "longitude", "room_type", "price",
    "minimum_nights", "number_of_reviews", "last_review",
    "reviews_per_month", "calculated_host_listings_count", "availability_365"
]
assert list(df.columns) == expected_columns, "Schema mismatch!"

print("\nCleaned Shape:", df.shape)
print(df.isnull().sum())

# Step 7: Cleaned data save karo
df.to_csv("AB_NYC_2019_cleaned.csv", index=False)
print("\nCleaning complete! File saved as AB_NYC_2019_cleaned.csv")
