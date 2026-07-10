# Step 1: Tools mangwana
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: CSV file load karna
df = pd.read_csv("Telco-Customer-Churn.csv")

# Step 3: Data ko dekhna
print("Total rows and columns:", df.shape)
df.head(10)

# Step 4: TotalCharges column theek karna (kuch values khali thi)
df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
df["TotalCharges"] = df["TotalCharges"].astype(float)

# Step 5: Kitne customers churn hue, count karna
churned = (df["Churn"] == "Yes").sum()
print("Customers who left (Churn):", churned)

# Step 6: Chart - Tenure ka distribution
sns.histplot(df["tenure"])
plt.title("Tenure Distribution")
plt.show()

# Step 7: Chart - Contract type ke hisaab se Churn
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.show()

# Step 8: Chart - Correlation heatmap
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
sns.heatmap(df[num_cols].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()
