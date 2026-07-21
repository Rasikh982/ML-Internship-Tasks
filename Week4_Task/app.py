import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Page title
st.title("A/B Test Analysis")

# Load data
df = pd.read_csv("ab_data.csv")

# Keep correct records only
df = df[
    ((df["group"] == "control") & (df["landing_page"] == "old_page")) |
    ((df["group"] == "treatment") & (df["landing_page"] == "new_page"))
]

# Remove duplicate users
df = df.drop_duplicates("user_id")

# Split groups
control = df[df["group"] == "control"]
treatment = df[df["group"] == "treatment"]

# Conversion rates
p_control = control["converted"].mean()
p_treatment = treatment["converted"].mean()

st.subheader("Conversion Rates")
st.write("Control:", round(p_control, 4))
st.write("Treatment:", round(p_treatment, 4))

# Chi-square test
table = pd.crosstab(df["group"], df["converted"])
chi2, p, _, _ = chi2_contingency(table)

st.subheader("Chi-Square Test")
st.write("P-value:", round(p, 4))

if p < 0.05:
    st.success("Significant difference found.")
else:
    st.info("No significant difference found.")

# Confidence interval
diff = p_treatment - p_control
se = np.sqrt(
    p_control * (1 - p_control) / len(control) +
    p_treatment * (1 - p_treatment) / len(treatment)
)

low = diff - 1.96 * se
high = diff + 1.96 * se

st.subheader("95% Confidence Interval")
st.write(f"[{low:.4f}, {high:.4f}]")

# Final result
st.subheader("Conclusion")

if p < 0.05:
    st.write("The new page has a significant effect on conversions.")
else:
    st.write("The new page does not significantly improve conversions.")
