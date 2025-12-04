#!/usr/bin/env python3
"""
Quick validations for fb2nep synthetic cohort (N≈25,000).
- Ranges & missingness
- Key correlations & gradients
- Incidence proportions & event-date logic
"""
from __future__ import annotations
import os, sys, math
import pandas as pd
import numpy as np

PATH = os.path.join("data", "synthetic", "fb2nep.csv")
if not os.path.exists(PATH):
    sys.exit(f"Not found: {PATH}. Run scripts/generate_dataset.py first.")

df = pd.read_csv(PATH)

# --- Basic shape ---
n = len(df)
assert 20000 <= n <= 30000, f"Row count {n} out of expected range"

# --- Required columns ---
required = {
    "id","baseline_date","follow_up_years","age","sex","menopausal_status",
    "IMD_quintile","SES_class","smoking_status","physical_activity",
    "family_history_cvd","family_history_cancer","BMI","SBP",
    "energy_kcal","fruit_veg_g_d","red_meat_g_d","ssb_ml_d","fibre_g_d",
    "alcohol_units_wk","salt_g_d","plasma_vitC_umol_L","urinary_sodium_mmol_L",
    "CVD_incident","CVD_date","Cancer_incident","Cancer_date"
}
missing_cols = required - set(df.columns)
assert not missing_cols, f"Missing columns: {sorted(missing_cols)}"

# --- Ranges (allow a small tolerance) ---
rng_checks = {
    "age": (40, 90),
    "BMI": (15, 55),
    "SBP": (80, 220),
    "energy_kcal": (800, 4500),
    "fruit_veg_g_d": (0, 1500),
    "red_meat_g_d": (0, 500),
    "ssb_ml_d": (0, 2000),
    "fibre_g_d": (0, 100),
    "salt_g_d": (2, 20),
    "plasma_vitC_umol_L": (5, 120),
    "urinary_sodium_mmol_L": (10, 250),
}
for col, (lo, hi) in rng_checks.items():
    s = df[col].dropna()
    frac_in = s.between(lo, hi).mean()
    assert frac_in > 0.98, f"{col}: >2% values outside [{lo},{hi}]"

# --- Missingness ---
overall_missing = df.isna().mean().mean()
assert 0.02 <= overall_missing <= 0.12, f"Overall missingness {overall_missing:.3f} outside 2–12%"

# --- Key associations ---
sub1 = df.dropna(subset=["fruit_veg_g_d","plasma_vitC_umol_L"])
corr_vitc = sub1["fruit_veg_g_d"].corr(sub1["plasma_vitC_umol_L"])
assert corr_vitc > 0.45, f"FV→VitC correlation too weak ({corr_vitc:.2f})"

sub2 = df.dropna(subset=["salt_g_d","urinary_sodium_mmol_L"])
corr_urna = sub2["salt_g_d"].corr(sub2["urinary_sodium_mmol_L"])
assert corr_urna > 0.55, f"Salt→UrNa correlation too weak ({corr_urna:.2f})"

# SBP ~ age (monotone trend; Spearman)
from scipy.stats import spearmanr
spear = spearmanr(df["age"], df["SBP"], nan_policy="omit").correlation
assert spear > 0.35, f"SBP vs age Spearman too low ({spear:.2f})"

# IMD/SES diet gradients (broad checks)
mean_fv_abc1 = df.loc[df["SES_class"]=="ABC1","fruit_veg_g_d"].mean()
mean_fv_c2de = df.loc[df["SES_class"]=="C2DE","fruit_veg_g_d"].mean()
assert mean_fv_abc1 > mean_fv_c2de, "FV should be higher in ABC1 than C2DE"

mean_ssb_imd12 = df.loc[df["IMD_quintile"].isin([1,2]),"ssb_ml_d"].mean()
mean_ssb_imd45 = df.loc[df["IMD_quintile"].isin([4,5]),"ssb_ml_d"].mean()
assert mean_ssb_imd12 > mean_ssb_imd45, "SSB should be higher in IMD 1–2 than 4–5"

# --- Incidence & event-date logic ---
p_cvd = df["CVD_incident"].mean()
p_ca  = df["Cancer_incident"].mean()
assert 0.10 <= p_cvd <= 0.15, f"CVD incidence {p_cvd:.3f} not in 10–15%"
assert 0.08 <= p_ca  <= 0.12, f"Cancer incidence {p_ca:.3f} not in 8–12%"

# Event date present only if incident=1; otherwise blank/NA
def check_dates(flag_col, date_col):
    f = df[flag_col].fillna(0).astype(int)
    d = df[date_col].fillna("")
    assert ( (f==1) <= (d!="") ).all(), f"{date_col}: some incident=1 rows missing a date"
    assert ( (f==0) <= (d=="") ).mean() > 0.98, f"{date_col}: >2% of incident=0 rows have a date"
check_dates("CVD_incident","CVD_date")
check_dates("Cancer_incident","Cancer_date")

print("Validation OK.")


print(df["CVD_incident"].mean(), df["Cancer_incident"].mean())