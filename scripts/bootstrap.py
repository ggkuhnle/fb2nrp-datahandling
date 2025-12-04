#!/usr/bin/env python3
"""
Bootstrap script for the fb2nrp-datahandling repository.

Responsibilities:
- Detect the repository root (works both locally and in Colab).
- Make the scripts/ directory importable so helpers.py can be used.
- Generate a synthetic dataset using simulate_practical_data().
- Return:
    df  : the synthetic dataset as a pandas DataFrame.
    CTX : a small context object with paths and flags.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # conventional for data work, even if not used heavily

# ----------------------------------------------------------------------
# Basic repo metadata
# ----------------------------------------------------------------------

REPO_NAME = "fb2nrp-datahandling"
REPO_URL = "https://github.com/ggkuhnle/fb2nrp-datahandling"


@dataclass
class Context:
    """Simple container for bootstrap context information."""
    repo_root: Path
    in_colab: bool
    repo_name: str = REPO_NAME
    repo_url: str = REPO_URL


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _detect_repo_root() -> Path:
    """
    Detect the repository root based on the location of this file.

    Assumes this file lives in <repo_root>/scripts/bootstrap.py.
    """
    return Path(__file__).resolve().parents[1]


def _detect_colab() -> bool:
    """
    Detect whether we are running inside Google Colab.

    Light-weight check based on the presence of the 'google.colab' module.
    """
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_import_paths(repo_root: Path) -> None:
    """
    Ensure that the repository root and scripts directory are on sys.path.

    This makes it possible to:
        from helpers import simulate_practical_data
    and also:
        import scripts.bootstrap   # if needed
    """
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    scripts_dir = repo_root / "scripts"
    scripts_str = str(scripts_dir)
    if scripts_dir.is_dir() and scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def init():
    """
    Initialise the fb2nrp-datahandling environment.

    Steps
    -----
    - Detect repo root and whether we are in Colab.
    - Add repo_root and scripts/ to sys.path.
    - Import simulate_practical_data() from scripts/helpers.py.
    - Generate a synthetic dataset `df` using the given seed.
    - Build and return a Context object.

    Parameters
    ----------
    seed : int, optional
        Random seed for the data simulation (default: 11088).

    Returns
    -------
    df : pandas.DataFrame
        Synthetic dataset for the practicals.
    CTX : Context
        Context with basic metadata (paths, flags).
    """
    repo_root = _detect_repo_root()
    in_colab = _detect_colab()

    _ensure_import_paths(repo_root)

    CTX = Context(repo_root=repo_root, in_colab=in_colab)
    return CTX
