# FB2NEP Python Cheat‑Sheet (Colab/Jupyter)

This one‑pager covers the most common things you’ll do in FB2NEP notebooks.

---

## 0) Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Stats & modelling (install if missing)
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula_api as smf
```

If a library is missing in Colab:
```python
!pip -q install statsmodels
# then: Runtime → Restart runtime
```

---

## 1) Load / save data

```python
# CSV from local upload or Drive
df = pd.read_csv("my_data.csv")

# CSV from GitHub (raw)
url = "https://raw.githubusercontent.com/ggkuhnke/fb2nep-eoi/main/data/synthetic/fb2nep.csv"
df = pd.read_csv(url)

# Save
df.to_csv("output.csv", index=False)
```

---

## 2) Quick look

```python
df.head()
df.tail()
df.shape
df.info()
df.describe(include="all")
df["sex"].value_counts(dropna=False)
df.isna().mean()  # fraction missing per column
```

---

## 3) Select / filter / transform

```python
# Columns
df[["age", "bmi"]]

# Rows
df[df["age"] >= 50]

# New columns
df["bmi_sq"] = df["bmi"] ** 2

# Rename
df = df.rename(columns={"cholesterol": "chol"})

# Sort
df = df.sort_values(["age", "bmi"], ascending=[True, False])
```

---

## 4) Grouping & summaries

```python
df.groupby("group")["bmi"].mean()
df.groupby(["group", "sex"])["sbp"].agg(["mean", "std", "count"])

# Crosstab
pd.crosstab(df["group"], df["sex"], margins=True, normalize="index")
```

---

## 5) Plotting (quick)

```python
df["bmi"].hist(bins=20)
plt.title("BMI distribution")
plt.xlabel("BMI"); plt.ylabel("Count")
plt.show()

# Boxplot by group
df.boxplot(column="sbp", by="group")
plt.suptitle(""); plt.title("SBP by group"); plt.xlabel("group"); plt.ylabel("SBP")
plt.show()

# Scatter
plt.scatter(df["bmi"], df["sbp"])
plt.xlabel("BMI"); plt.ylabel("SBP"); plt.show()
```

---

## 6) Basic stats

```python
# Two-sample t-test
a = df.loc[df["group"] == "A", "sbp"]
b = df.loc[df["group"] == "B", "sbp"]
stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")

# Chi-square test on a 2x2
tab = pd.crosstab(df["group"], df["sex"])
stats.chi2_contingency(tab)
```

---

## 7) Simple models (statsmodels)

```python
# OLS regression
model = smf.ols("sbp ~ age + bmi + C(sex) + C(group)", data=df).fit()
print(model.summary())

# Logistic regression (binary outcome)
# e.g., 'high_sbp' is 0/1
logit = smf.logit("high_sbp ~ age + bmi + C(sex) + C(group)", data=df).fit()
print(logit.summary())
```

---

## 8) Jupyter basics

- Run cell: **Shift + Enter**
- Insert cell above/below: **A / B**
- Interrupt: stop button (■) or `Kernel/Runtime → Interrupt`
- Restart: `Kernel/Runtime → Restart`
- Markdown cell: text with `#` headings, `**bold**`, lists, etc.

---

## 9) Reproducibility

```python
SEED = 11088
np.random.seed(SEED)
```

- Record: dataset version, random seed, and exact code you ran.
