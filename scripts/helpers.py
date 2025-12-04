#!/usr/bin/env python3
"""
Helper functions for simulating the synthetic practical data.

This keeps the data-generation logic separate from the notebooks,
so that:
- notebooks stay readable, and
- the same simulation can be reused elsewhere if needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_practical_data(seed: int = 11088) -> pd.DataFrame:
    """
    Simulate a small dataset mimicking the Part 2 practicals.

    Creates:
    - sex (F/M)
    - age (years)
    - coffee_arm: low / medium / high
    - cereal_arm: bran / cornflakes / muesli
    - food_arm: apple / biscuit / yoghurt
    - bp_change: change in blood pressure (mmHg)
    - glucose: postprandial blood glucose (arbitrary units)
    - appetite_vas: appetite VAS (0–100)

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility (default: 11088).

    Returns
    -------
    df : pandas.DataFrame
        Simulated dataset with one row per participant.
    """
    rng = np.random.default_rng(seed)

    n = 180

    sex = rng.choice(["F", "M"], size=n, p=[0.6, 0.4])
    age = rng.normal(21, 2, n).round()

    coffee_arm = rng.choice(["low", "medium", "high"], size=n, p=[0.33, 0.33, 0.34])
    cereal_arm = rng.choice(["bran", "cornflakes", "muesli"], size=n)
    food_arm = rng.choice(["apple", "biscuit", "yoghurt"], size=n)

    # Simulate outcomes with small arm-specific shifts
    bp_change = (
        rng.normal(0, 5, n)
        + (coffee_arm == "high") * 3
        - (coffee_arm == "low") * 1
    )

    glucose = (
        rng.normal(5.0, 0.5, n)
        + (cereal_arm == "cornflakes") * 0.7
        - (cereal_arm == "muesli") * 0.3
    )

    appetite = (
        rng.normal(60, 15, n)
        - (food_arm == "apple") * 10
        + (food_arm == "biscuit") * 5
    )
    appetite = np.clip(appetite, 0, 100)

    df = pd.DataFrame({
        "sex": sex,
        "age": age,
        "coffee_arm": coffee_arm,
        "cereal_arm": cereal_arm,
        "food_arm": food_arm,
        "bp_change": bp_change,
        "glucose": glucose,
        "appetite_vas": appetite,
    })

    return df
