# Task 04 - Statistical Analysis and Hypothesis Testing

## What this project does

This project analyzes an A/B test dataset. Users were shown either an
**old landing page (control group)** or a **new landing page (treatment
group)**, and we check if the new page leads to more conversions.

## Dataset

- File: `ab_data.csv`
- Columns: `user_id`, `timestamp`, `group` (control/treatment),
  `landing_page` (old_page/new_page), `converted` (0 or 1)
- Source: A/B Testing dataset (294,478 rows)

## Statistical tests used

| Test | Purpose |
|---|---|
| Chi-square test | Check if conversion rate differs between page versions |
| Confidence Interval | Estimate the range of the true difference in conversion rate |
| Effect size (Cohen's h, Cramer's V) | Measure how big the difference actually is |
| T-test | Compare average visit hour between the two groups |
| Mann-Whitney U test | Non-parametric version of the t-test |
| ANOVA | Compare conversion rate across all 7 weekdays |
| Multiple comparison correction (Bonferroni, Benjamini-Hochberg) | Adjust p-values when running many tests at once |

## How to run

1. Install requirements:
```
pip install -r requirements.txt
```

2. Run the app:
```
streamlit run app.py
```

3. Make sure `ab_data.csv` is in the same folder as `app.py`.

## Main finding

The chi-square test and confidence interval show **no statistically
significant difference** in conversion rate between the old page and the
new page. The effect size is very small. So, based on this data, the new
page does not clearly perform better than the old page.

## Files in this project

- `app.py` - Streamlit app with all the analysis and tests
- `ab_data.csv` - the dataset
- `requirements.txt` - Python packages needed
- `README.md` - this file
