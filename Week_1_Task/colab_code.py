import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
df = pd.read_csv("Telco-Customer-Churn.csv") 
print("Total rows and columns:", df.shape)
df.head(10)
)
df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
df["TotalCharges"] = df["TotalCharges"].astype(float)
churned = (df["Churn"] == "Yes").sum()
print("Customers who left (Churn):", churned)
sns.histplot(df["tenure"])
plt.title("Tenure Distribution")
plt.show()
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.show()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
sns.heatmap(df[num_cols].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()
