"""
I/O helper functions.

Handles reading/writing YAML configs, loading/saving DataFrames
in Parquet format, and model serialisation shortcuts.
"""

import yaml
import pandas as pd
from pathlib import Path


def load_yaml(path: str) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
    """
    ...


def save_dataframe(df: pd.DataFrame, path: str, fmt: str = "parquet") -> None:
    """Save a DataFrame to disk.

    Parameters
    ----------
    df : pd.DataFrame
    path : str
    fmt : str
        ``'parquet'`` or ``'csv'``.
    """
    ...


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a DataFrame from Parquet or CSV based on extension.

    Parameters
    ----------
    path : str

    Returns
    -------
    pd.DataFrame
    """
    ...


def ensure_dir(path: str) -> Path:
    """Create a directory (and parents) if it doesn't exist.

    Parameters
    ----------
    path : str

    Returns
    -------
    Path
    """
    ...
