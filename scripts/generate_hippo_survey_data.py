#!/usr/bin/env python3
"""
Generate a small synthetic 'hippo dietary survey' dataset.

This script is intended for the FB2NEP teaching repository.
It creates a tidy CSV file at:

    data/hippo_diet_survey.csv

The dataset is deliberately small and simple so that it can be used
in introductory teaching material to illustrate:

- loading data from CSV (pd.read_csv),
- inspecting structure (.head(), .info(), .describe()),
- selecting columns and rows,
- using methods such as .mean() and .groupby(),
- creating simple plots.

Usage
-----
From the repository root (one level above 'scripts'):

    python scripts/generate_hippo_diet_survey.py

The script will create the 'data' directory if it does not exist
and then write 'hippo_diet_survey.csv' into it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build_hippo_dataframe(random_state: int = 11088) -> pd.DataFrame:
    """
    Construct a small synthetic hippo dietary survey dataset.

    The values are partly fixed (names, habitats) and partly generated
    with small random variation (ages and intakes), using a fixed seed
    for reproducibility.

    Parameters
    ----------
    random_state : int
        Seed for the NumPy random number generator.

    Returns
    -------
    pandas.DataFrame
        A tidy table with one row per hippo and columns:

        - hippo_id
        - name
        - age_years
        - habitat
        - fruit_portions
        - veg_portions
        - grass_kg
    """
    rng = np.random.default_rng(random_state)

    # Fixed names and habitats to keep the dataset recognisable in teaching.
    names = ["Helga", "Bruno", "Ama", "Kofi", "Lina", "Otto", "Sita", "Milo"]
    habitats = ["River", "River", "Lake", "Zoo", "Lake", "River", "Zoo", "Lake"]

    n = len(names)
    hippo_ids = np.arange(1, n + 1)

    # Ages: integers between 3 and 20 years (inclusive).
    age_years = rng.integers(low=3, high=21, size=n)

    # Fruit and vegetable portions per day: small integers with plausible range.
    # These are kept simple to make summary statistics and plots easy to follow.
    fruit_portions = rng.integers(low=1, high=4, size=n)   # 1–3 portions
    veg_portions = rng.integers(low=2, high=6, size=n)     # 2–5 portions

    # Grass intake in kg per day.
    # Start from a habitat-specific baseline and add a little random noise.
    baseline_grass = {
        "River": 55.0,
        "Lake": 45.0,
        "Zoo":  50.0,
    }
    grass_kg = []
    for hab in habitats:
        mean = baseline_grass[hab]
        # Add normally distributed noise with mean 0 and sd 4 kg.
        value = rng.normal(loc=mean, scale=4.0)
        # Round to one decimal place for neatness.
        grass_kg.append(round(float(value), 1))

    df = pd.DataFrame(
        {
            "hippo_id": hippo_ids,
            "name": names,
            "age_years": age_years,
            "habitat": habitats,
            "fruit_portions": fruit_portions,
            "veg_portions": veg_portions,
            "grass_kg": grass_kg,
        }
    )

    return df


def main() -> int:
    """
    Entry point for command-line use.

    Determines the repository root as the parent directory of this script,
    creates the 'data' directory if needed, generates the dataset and writes
    it to 'data/hippo_diet_survey.csv'.
    """
    # Assume this file lives in 'scripts' and the repository root is one level up.
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]

    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "hippo_diet_survey.csv"

    df = build_hippo_dataframe(random_state=11088)
    df.to_csv(output_path, index=False)

    print(f"Generated hippo dietary survey with {len(df)} rows.")
    print(f"Saved CSV to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

