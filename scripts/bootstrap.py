"""
FB2NEP bootstrap (Colab + local)

Typical use in a notebook (top cell):

    import runpy, pathlib
    for p in ["scripts/bootstrap.py", "../scripts/bootstrap.py", "../../scripts/bootstrap.py"]:
        if pathlib.Path(p).exists():
            print(f"Bootstrapping via: {p}")
            runpy.run_path(p)
            break
    else:
        raise FileNotFoundError("scripts/bootstrap.py not found")

    # Now: df, CTX, CSV_REL, REPO_ROOT, IN_COLAB are available
    df.head()

Alternative (import) style:

    from scripts.bootstrap import init
    df, ctx = init()

This script is deliberately written in a single file so that
it can be reused across multiple notebooks.
"""

from __future__ import annotations

import os
import sys
import shlex
import pathlib
import subprocess
from dataclasses import dataclass

# --------------------------------------------------------------------
# Configuration (can be overridden via environment variables)
# --------------------------------------------------------------------
# The idea is that, by default, everything is configured for fb2nep-epi,
# but you can change paths for testing by setting FB2NEP_* env variables.

# Name of the repository directory (used when cloning in Colab).
REPO_NAME   = os.getenv("FB2NEP_REPO", "fb2nep-epi")

# URL of the Git repository (used when cloning in Colab).
REPO_URL    = os.getenv("FB2NEP_REPO_URL", "https://github.com/ggkuhnle/fb2nep-epi.git")

# Relative path to the main synthetic dataset, starting from the repo root.
CSV_REL     = os.getenv("FB2NEP_CSV", "data/synthetic/fb2nep.csv")

# Path to the data generator script (again, relative to the repo root).
GEN_SCRIPT  = os.getenv("FB2NEP_GEN", "scripts/generate_dataset.py")

# Requirements file used in Colab to install dependencies.
REQS_FILE   = os.getenv("FB2NEP_REQS", "requirements.txt")

# If set to "1" in the environment, forces regeneration of the dataset.
FORCE_REGEN = os.getenv("FB2NEP_FORCE_REGEN", "0") == "1"

# Simple flag: are we running inside Google Colab?
IN_COLAB = "google.colab" in sys.modules


@dataclass
class Context:
    """
    Small container for metadata about the current environment.

    This is returned by init() so that notebooks have easy access to:
    - the repository root directory,
    - the relative CSV path,
    - the generator script path,
    - whether we are in Colab,
    - the Git repository URL and name.
    """
    repo_root: pathlib.Path
    csv_rel: str
    gen_script: str
    in_colab: bool
    repo_url: str
    repo_name: str


def _run(cmd: str) -> int:
    """
    Run a shell command as a subprocess and return the exit code.

    Parameters
    ----------
    cmd : str
        Command string, e.g. "pip install -r requirements.txt".

    Returns
    -------
    int
        Exit code returned by the subprocess (0 means success).
    """
    print(">", cmd)  # Echo the command so that the user sees what is happening.
    return subprocess.run(shlex.split(cmd), check=False).returncode


def _looks_like_root(p: pathlib.Path) -> bool:
    """
    Heuristic check: does this path look like the repository root?

    We treat a directory as the repository root if it contains
    both "scripts" and "notebooks" subdirectories.
    """
    return (p / "scripts").exists() and (p / "notebooks").exists()


def ensure_repo_root() -> pathlib.Path:
    """
    Ensure that the current working directory (CWD) is the repository root.

    Behaviour:
    ----------
    - If the current directory already looks like the root (has scripts/ and notebooks/),
      keep it as-is.
    - If the parent directory looks like the root, change into the parent.
      This handles the common case where a notebook is opened from notebooks/.
    - In Colab, if neither current nor parent look like the root, attempt to:
        * clone the repository from REPO_URL into /content/REPO_NAME (if not present),
        * change into that directory.
    - As a fallback, if none of the above work, stay in the current directory and
      print a warning. Relative paths may then require manual adjustment.

    Returns
    -------
    pathlib.Path
        The directory that is being treated as the repository root.
    """
    here = pathlib.Path.cwd()

    # Case 1: current directory already has the correct structure.
    if _looks_like_root(here):
        return here

    # Case 2: we are inside a subdirectory (for example notebooks/).
    if _looks_like_root(here.parent):
        os.chdir(here.parent)
        return here.parent

    # Case 3: Colab, where we often start in /content.
    if IN_COLAB:
        print("Cloning repository (Colab)…")
        # If the repository is not already cloned into /content, clone it.
        repo_path = pathlib.Path("/content") / REPO_NAME
        if not repo_path.exists():
            _run(f"git clone {REPO_URL}")
        # After cloning, or if it was already present, change into it.
        if (pathlib.Path.cwd() / REPO_NAME).exists():
            os.chdir(REPO_NAME)
            return pathlib.Path.cwd()

    # Fallback: we did not manage to identify the root.
    # We keep the current directory and warn the user.
    print("⚠️ Could not auto-detect repo root; continuing in", here)
    return here


def ensure_deps():
    """
    Ensure that Python dependencies are available.

    Behaviour:
    ----------
    - Try to import a minimal set of core libraries:
      numpy, pandas, matplotlib, statsmodels.
    - If this fails and we are in Colab:
        * If REQS_FILE exists, install from that via pip.
        * Otherwise, install a small default set with pip.
    - If this fails on a local machine:
        * Print a warning, but do not modify the user's environment.

    The aim is to make notebooks mostly self-contained in Colab,
    while respecting local virtual environments.
    """
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import matplotlib  # noqa: F401
        import statsmodels  # noqa: F401
    except Exception as e:
        if IN_COLAB:
            print("Installing Python dependencies (Colab)…")
            # Preferred route: install from the repository's requirements.txt file.
            if os.path.exists(REQS_FILE):
                _run(f"pip install -q -r {REQS_FILE}")
            else:
                # Fallback: install a minimal useful set directly.
                _run("pip install -q numpy pandas matplotlib seaborn statsmodels")
        else:
            # For local use, we simply inform the user.
            print("⚠️ Missing dependencies locally:", e)
            print("   Consider: `pip install -r requirements.txt` in your virtual environment.")


def ensure_data(csv_rel: str, gen_script: str):
    """
    Ensure that the main teaching dataset exists on disk.

    Parameters
    ----------
    csv_rel : str
        Relative path to the dataset CSV file, e.g. "data/synthetic/fb2nep.csv".
    gen_script : str
        Relative path to the dataset generator script, e.g. "scripts/generate_dataset.py".

    Behaviour:
    ----------
    - If the CSV file already exists and FORCE_REGEN is False:
        * Do nothing (assume the existing file is fine).
    - If FORCE_REGEN is True, or the CSV file is missing:
        * If the generator script exists, attempt to run it.
        * If generation appears to succeed and the CSV file now exists,
          stop and accept the generated file.
    - If we are in Colab and the file is still missing:
        * Offer a manual upload option via google.colab.files.upload().
    - If, after all attempts, the file is still missing:
        * Raise FileNotFoundError.
    """
    # Case 1: dataset already present and no forced regeneration requested.
    if os.path.exists(csv_rel) and not FORCE_REGEN:
        print(f"Dataset found: {csv_rel} ✅")
        return

    # Case 2: dataset missing or we explicitly requested regeneration.
    if FORCE_REGEN or not os.path.exists(csv_rel):
        if os.path.exists(gen_script):
            print("Generating dataset…")
            rc = _run(f"python {gen_script}")
            if rc == 0 and os.path.exists(csv_rel):
                print(f"Generated: {csv_rel} ✅")
                return
            print("⚠️ Generation failed or file still missing.")

    # Case 3: Colab fallback – allow manual upload.
    if IN_COLAB and not os.path.exists(csv_rel):
        try:
            from google.colab import files  # type: ignore
            target_dir = os.path.dirname(csv_rel) or "."
            os.makedirs(target_dir, exist_ok=True)
            print(f"Upload fb2nep.csv (will be saved to {csv_rel}) …")
            uploaded = files.upload()
            if "fb2nep.csv" in uploaded:
                # Save uploaded file to the expected location.
                with open(csv_rel, "wb") as f:
                    f.write(uploaded["fb2nep.csv"])
                print(f"Uploaded: {csv_rel} ✅")
                return
            else:
                print("⚠️ fb2nep.csv not provided.")
        except Exception as e:
            print("Upload fallback failed:", e)

    # Final check: if the file still does not exist, give up with an error.
    if not os.path.exists(csv_rel):
        raise FileNotFoundError(f"Could not obtain dataset at {csv_rel}")


def init():
    """
    Top-level initialisation function for notebooks.

    This is intended to be the single entry point that notebooks call.

    Behaviour:
    ----------
    1. Ensures that we are in the repository root (ensure_repo_root).
    2. Ensures that core dependencies are available (ensure_deps).
    3. Ensures that the main dataset exists (ensure_data).
    4. Loads the dataset into a pandas DataFrame named df.
    5. Constructs a Context object (ctx) with metadata.
    6. Injects df, CTX, CSV_REL, REPO_ROOT, IN_COLAB into the global
       namespace, which is convenient for `%run`-style use.
    7. Returns (df, ctx).

    Returns
    -------
    df : pandas.DataFrame
        The main synthetic FB2NEP dataset.
    ctx : Context
        Metadata about the current environment and repository.
    """
    import pandas as pd

    # Step 1: determine the repository root and change directory if necessary.
    repo_root = ensure_repo_root()

    # Step 2: ensure that required Python libraries are available.
    ensure_deps()

    # Step 3: ensure that the dataset exists (generate or request upload if needed).
    ensure_data(CSV_REL, GEN_SCRIPT)

    # Step 4: load the main dataset.
    df = pd.read_csv(CSV_REL)

    # Step 5: construct a context object with useful metadata.
    ctx = Context(
        repo_root=repo_root,
        csv_rel=CSV_REL,
        gen_script=GEN_SCRIPT,
        in_colab=IN_COLAB,
        repo_url=REPO_URL,
        repo_name=REPO_NAME,
    )

    # Step 6: make a few convenience variables available at module level.
    # This is especially useful if the script is run via runpy.run_path(...).
    globals().update({
        "df": df,
        "CTX": ctx,
        "CSV_REL": CSV_REL,
        "REPO_ROOT": repo_root,
        "IN_COLAB": IN_COLAB,
    })

    # Step 7: return the freshly loaded DataFrame and context.
    return df, ctx


if __name__ == "__main__":
    # If the script is executed directly (e.g. `python scripts/bootstrap.py`),
    # run init() so that it behaves sensibly on its own.
    init()
