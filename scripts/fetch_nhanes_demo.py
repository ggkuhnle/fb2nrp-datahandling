#!/usr/bin/env python3
"""
Helper functions to download a small NHANES demographic dataset for teaching.

- Downloads DEMO_J (demographic) and BMX_J (body measures) for NHANES 2017–2018.
- Merges them on SEQN, the participant identifier.
- Creates a tidy DataFrame with:

    - age_years
    - sex
    - race_eth  (simplified categories)
    - education (simplified categories)
    - bmi       (kg/m^2)

- Restricts to adults (age_years >= 20).

The resulting data set is suitable for teaching concepts of
representativeness, sampling, and central tendency.

Only public NHANES files are used. No data are shipped with the repository.
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import pandas as pd

# URLs for NHANES 2017–2018 XPT files (DEMO_J and BMX_J)
DEMO_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/Datafiles/DEMO_J.XPT"
BMX_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/Datafiles/BMX_J.XPT"

# Local cache path for the processed CSV
CACHE_PATH = pathlib.Path("data/private/nhanes_demo_2017_2018.csv")


def _download_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download raw NHANES DEMO_J and BMX_J files from the CDC website.

    Returns
    -------
    demo : pandas.DataFrame
        Raw demographic data (DEMO_J).
    bmx : pandas.DataFrame
        Raw body measures data (BMX_J).
    """
    print("Downloading DEMO_J (demographics) from NHANES …")
    demo = pd.read_sas(DEMO_URL, format="xport")

    print("Downloading BMX_J (body measures) from NHANES …")
    bmx = pd.read_sas(BMX_URL, format="xport")

    return demo, bmx


def _tidy_nhanes(demo: pd.DataFrame, bmx: pd.DataFrame) -> pd.DataFrame:
    """
    Create a tidy NHANES teaching dataset from raw DEMO_J and BMX_J.

    The function:

    - Renames key variables (age, sex, race/ethnicity, education, BMI).
    - Maps coded values to readable categories.
    - Merges DEMO and BMX on SEQN.
    - Restricts to adults aged 20 years and older.
    """
    # Keep only columns needed for teaching; this keeps memory use modest.
    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2"]].copy()
    bmx = bmx[["SEQN", "BMXBMI"]].copy()

    # Rename columns to more readable names.
    demo = demo.rename(
        columns={
            "RIDAGEYR": "age_years",
            "RIAGENDR": "sex_code",
            "RIDRETH3": "race_eth_code",
            "DMDEDUC2": "education_code",
        }
    )
    bmx = bmx.rename(columns={"BMXBMI": "bmi"})

    # Merge on SEQN.
    merged = demo.merge(bmx, on="SEQN", how="left", validate="one_to_one")

    # Map sex codes: 1 = Male, 2 = Female.
    sex_map = {1: "Male", 2: "Female"}
    merged["sex"] = merged["sex_code"].map(sex_map)

    # Map race/ethnicity codes (RIDRETH3) to simplified categories.
    # 1: Mexican American
    # 2: Other Hispanic
    # 3: Non-Hispanic White
    # 4: Non-Hispanic Black
    # 6: Non-Hispanic Asian
    # 7: Other race, including multi-racial
    race_map_simple = {
        1: "Hispanic",
        2: "Hispanic",
        3: "White",
        4: "Black",
        6: "Asian",
        7: "Other",
    }
    merged["race_eth"] = merged["race_eth_code"].map(race_map_simple)

    # Map education codes (DMDEDUC2) for adults:
    # 1: Less than 9th grade
    # 2: 9–11th grade (including 12th grade with no diploma)
    # 3: High school graduate / GED or equivalent
    # 4: Some college or AA degree
    # 5: College graduate or above
    # 7: Refused
    # 9: Do not know
    def map_education(code):
        if code in (1, 2, 3):
            return "≤High school"
        if code == 4:
            return "Some college"
        if code == 5:
            return "College+"
        if code in (7, 9):
            return "Unknown"
        return "Missing"

    merged["education"] = merged["education_code"].apply(map_education)

    # Restrict to adults aged 20 years and older.
    merged = merged[merged["age_years"] >= 20].copy()

    # Keep only the tidy columns we actually need.
    tidy = merged[
        [
            "SEQN",
            "age_years",
            "sex",
            "race_eth",
            "education",
            "bmi",
        ]
    ].reset_index(drop=True)

    return tidy


def load_nhanes_demo(cache: bool = True) -> pd.DataFrame:
    """
    Load a small processed NHANES data set for teaching.

    The function behaves as follows:

    1. If cache is True and the cached CSV exists in data/,
       load it and return it.
    2. Otherwise download DEMO_J and BMX_J from the CDC website,
       tidy the data, and (optionally) write the cached CSV.

    Parameters
    ----------
    cache : bool, default True
        If True, save the processed data to CACHE_PATH
        and reuse it on the next call.

    Returns
    -------
    df : pandas.DataFrame
        Processed NHANES data with one row per participant and
        the columns: age_years, sex, race_eth, education, bmi.
    """
    # 1. Try to read from cache if requested.
    if cache and CACHE_PATH.exists():
        print(f"Loading NHANES demo from cache: {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH)

    # 2. Download raw data and create tidy version.
    demo, bmx = _download_raw()
    df = _tidy_nhanes(demo, bmx)

    # Ensure that the data directory exists before caching.
    if cache:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
        print(f"Saved processed NHANES demo to {CACHE_PATH}")

    return df
