"""
Load raw datasets (M5, Rossmann, Favorita) from disk.

Handles reading CSV/Parquet files, basic dtype casting, and returning
unified DataFrames ready for the cleaning pipeline.
"""

import pandas as pd
from pathlib import Path


def load_m5(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load the M5 Walmart dataset files.

    Parameters
    ----------
    data_dir : str
        Path to the ``data/raw/m5`` directory.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys like ``'sales'``, ``'calendar'``, ``'prices'``.
    """
    ...


def load_rossmann(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load the Rossmann Store Sales dataset.

    Parameters
    ----------
    data_dir : str
        Path to the ``data/raw/rossmann`` directory.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys like ``'train'``, ``'store'``.
    """
    ...


def load_favorita(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load the Favorita Grocery Sales dataset.

    Parameters
    ----------
    data_dir : str
        Path to the ``data/raw/favorita`` directory.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys like ``'train'``, ``'stores'``, ``'items'``.
    """
    ...


def load_dataset(name: str, data_dir: str) -> dict[str, pd.DataFrame]:
    """Dispatch loader by dataset name.

    Parameters
    ----------
    name : str
        One of ``'m5'``, ``'rossmann'``, ``'favorita'``.
    data_dir : str
        Root data directory.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    ...
