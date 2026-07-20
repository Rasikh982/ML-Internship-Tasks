"""
Task 04 - Statistical Analysis and Hypothesis Testing
Streamlit app for analyzing an A/B test dataset (control vs treatment).

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="A/B Test Statistical Analysis", layout="wide")
st.title("Task 04: Statistical Analysis and Hypothesis Testing")
st.write("Dataset: A/B Testing (control/old_page vs treatment/new_page)")


# -----------------------------
# Step 1: Load and clean data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ab_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Some rows have mismatched group/landing_page (a known issue in this dataset)
    # Keep only correct pairs: control-old_page and treatment-new_page
    correct_pairs = (
        ((df["group"] == "control") & (df["landing_page"] == "old_page")) |
        ((df["group"] == "treatment") & (df["landing_page"] == "new_page"))
    )
    df = df[correct_pairs]

    # Remove duplicate users (same user tested twice)
    df = df.drop_duplicates(subset="user_id", keep="first")

    # Extra columns used later for more tests
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()

    return df


df = load_data()

st.subheader("1. Data Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total rows (after cleaning)", f"{len(df):,}")
col2.metric("Control users", f"{(df['group'] == 'control').sum():,}")
col3.metric("Treatment users", f"{(df['group'] == 'treatment').sum():,}")

st.dataframe(df.head(10))

control = df[df["group"] == "control"]
treatment = df[df["group"] == "treatment"]


# -----------------------------
# Step 2: Conversion rates
# -----------------------------
st.subheader("2. Conversion Rates")

n_con, n_treat = len(control), len(treatment)
conv_con, conv_treat = control["converted"].sum(), treatment["converted"].sum()
p_con, p_treat = conv_con / n_con, conv_treat / n_treat

col1, col2 = st.columns(2)
col1.metric("Control conversion rate", f"{p_con:.4f}")
col2.metric("Treatment conversion rate", f"{p_treat:.4f}")


# -----------------------------
# Step 3: Chi-square test (conversion vs group)
# -----------------------------
st.subheader("3. Chi-Square Test: Does landing page affect conversion?")

st.write("H0 (null hypothesis): Conversion rate is the same for both pages.")
st.write("H1 (alternative): Conversion rate is different between pages.")

contingency_table = pd.crosstab(df["group"], df["converted"])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)

st.write(f"Chi-square statistic = {chi2:.4f}")
st.write(f"p-value = {p_chi:.4f}")

if p_chi < 0.05:
    st.success("p-value < 0.05 -> Reject H0. Conversion rate is significantly different.")
else:
    st.info("p-value >= 0.05 -> Fail to reject H0. No significant difference found.")


# -----------------------------
# Step 4: Confidence Interval for the difference in proportions
# -----------------------------
st.subheader("4. Confidence Interval (Difference in Conversion Rates)")

diff = p_treat - p_con
se = np.sqrt(p_con * (1 - p_con) / n_con + p_treat * (1 - p_treat) / n_treat)
ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se

st.write(f"Difference (treatment - control) = {diff:.4f}")
st.write(f"95% Confidence Interval = [{ci_low:.4f}, {ci_high:.4f}]")
st.caption("If the interval contains 0, the difference is not statistically significant.")


# -----------------------------
# Step 5: Effect Size
# -----------------------------
st.subheader("5. Effect Size")

# Cohen's h for two proportions
cohens_h = 2 * np.arcsin(np.sqrt(p_treat)) - 2 * np.arcsin(np.sqrt(p_con))

# Cramer's V for the chi-square test
n_total = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2 / (n_total * (min(contingency_table.shape) - 1)))

col1, col2 = st.columns(2)
col1.metric("Cohen's h", f"{cohens_h:.4f}")
col2.metric("Cramer's V", f"{cramers_v:.4f}")
st.caption("Values near 0 mean a very small (practically weak) effect.")


# -----------------------------
# Step 6: T-test (are visit hours different between groups?)
# -----------------------------
st.subheader("6. T-Test: Visit Hour (Control vs Treatment)")
st.write("Checks if users in each group visited at different times of day, on average.")

t_stat, p_t = stats.ttest_ind(control["hour"], treatment["hour"])
st.write(f"t-statistic = {t_stat:.4f}, p-value = {p_t:.4f}")


# -----------------------------
# Step 7: Mann-Whitney U test (non-parametric alternative to t-test)
# -----------------------------
st.subheader("7. Mann-Whitney U Test: Visit Hour (Non-Parametric)")
st.write("Same question as the t-test, but does not assume normal distribution.")

u_stat, p_u = stats.mannwhitneyu(control["hour"], treatment["hour"])
st.write(f"U-statistic = {u_stat:.1f}, p-value = {p_u:.4f}")


# -----------------------------
# Step 8: ANOVA (conversion across weekdays)
# -----------------------------
st.subheader("8. ANOVA: Conversion Rate Across Weekdays")
st.write("Checks if conversion differs across more than two groups (the 7 weekdays).")

weekday_groups = [g["converted"].values for _, g in df.groupby("weekday")]
f_stat, p_f = stats.f_oneway(*weekday_groups)
st.write(f"F-statistic = {f_stat:.4f}, p-value = {p_f:.4f}")


# -----------------------------
# Step 9: Multiple Comparison Correction
# -----------------------------
st.subheader("9. Multiple Comparison Correction")
st.write("Running one chi-square test per weekday means 7 tests total. "
         "Running many tests raises the chance of a false positive, "
         "so p-values must be corrected.")

days = sorted(df["weekday"].unique())
raw_pvalues = []

for day in days:
    day_df = df[df["weekday"] == day]
    c = day_df[day_df["group"] == "control"]
    t = day_df[day_df["group"] == "treatment"]
    pc, pt = c["converted"].sum() / len(c), t["converted"].sum() / len(t)
    p_pool = (c["converted"].sum() + t["converted"].sum()) / (len(c) + len(t))
    se_day = np.sqrt(p_pool * (1 - p_pool) * (1 / len(c) + 1 / len(t)))
    z_day = (pt - pc) / se_day
    p_day = 2 * (1 - stats.norm.cdf(abs(z_day)))
    raw_pvalues.append(p_day)

# Bonferroni correction: multiply each p-value by number of tests
bonferroni = [min(p * len(raw_pvalues), 1) for p in raw_pvalues]

# Benjamini-Hochberg (FDR) correction
m = len(raw_pvalues)
order = np.argsort(raw_pvalues)
sorted_p = np.array(raw_pvalues)[order]
bh_temp = sorted_p * m / (np.arange(m) + 1)
bh_sorted = np.minimum.accumulate(bh_temp[::-1])[::-1]
bh_final = np.empty(m)
bh_final[order] = np.minimum(bh_sorted, 1)

results_table = pd.DataFrame({
    "Weekday": days,
    "Raw p-value": [round(p, 4) for p in raw_pvalues],
    "Bonferroni corrected": [round(p, 4) for p in bonferroni],
    "Benjamini-Hochberg corrected": [round(p, 4) for p in bh_final],
})
st.dataframe(results_table)


# -----------------------------
# Step 10: Final conclusion
# -----------------------------
st.subheader("10. Conclusion")
st.write(
    "The chi-square test and confidence interval both show no statistically "
    "significant difference in conversion rate between the old page and the "
    "new page. The effect size is close to zero, meaning even if there is a "
    "difference, it is too small to matter in practice. Based on this data, "
    "the new page does not clearly improve conversions."
)
