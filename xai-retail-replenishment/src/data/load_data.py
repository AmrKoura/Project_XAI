"""
Load raw datasets (M5, Rossmann) from disk.

Handles reading CSV files, basic dtype casting, and returning
DataFrames ready for the cleaning pipeline.
"""

from pathlib import Path

import pandas as pd


def _validate_data_dir(data_dir: str) -> Path:
    """Validate that data_dir exists and is a directory."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory path, got: {path}")
    return path


def _read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV file with common defaults."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def load_m5(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load M5 Forecasting dataset files.

    Parameters
    ----------
    data_dir : str
        Path to the data/raw/m5 directory.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys can include:
        - calendar
        - prices
        - sample_submission
        - sales_train_validation (if present)
        - sales_train_evaluation (if present)
    """
    root = _validate_data_dir(data_dir)

    calendar = _read_csv(root / "calendar.csv", parse_dates=["date"])
    prices = _read_csv(root / "sell_prices.csv")
    sample_submission = _read_csv(root / "sample_submission.csv")

    data: dict[str, pd.DataFrame] = {
        "calendar": calendar,
        "prices": prices,
        "sample_submission": sample_submission,
    }

    validation_path = root / "sales_train_validation.csv"
    evaluation_path = root / "sales_train_evaluation.csv"

    if validation_path.exists():
        data["sales_train_validation"] = _read_csv(validation_path)

    if evaluation_path.exists():
        data["sales_train_evaluation"] = _read_csv(evaluation_path)

    if "sales_train_validation" not in data and "sales_train_evaluation" not in data:
        raise FileNotFoundError(
            "M5 sales file not found. Expected at least one of: "
            "sales_train_validation.csv or sales_train_evaluation.csv"
        )

    return data


def load_rossmann(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load Rossmann Store Sales dataset files.

    Parameters
    ----------
    data_dir : str
        Path to the data/raw/rossmann directory.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys:
        - train
        - test
        - store
        - sample_submission
    """
    root = _validate_data_dir(data_dir)

    return {
        "train": _read_csv(root / "train.csv", parse_dates=["Date"]),
        "test": _read_csv(root / "test.csv", parse_dates=["Date"]),
        "store": _read_csv(root / "store.csv"),
        "sample_submission": _read_csv(root / "sample_submission.csv"),
    }


def load_dataset(name: str, data_dir: str) -> dict[str, pd.DataFrame]:
    """Dispatch loader by dataset name.

    Parameters
    ----------
    name : str
        One of 'm5' or 'rossmann'.
    data_dir : str
        Dataset-specific directory path.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    key = name.strip().lower()

    if key == "m5":
        return load_m5(data_dir)
    if key == "rossmann":
        return load_rossmann(data_dir)

    raise ValueError("Unsupported dataset name. Use 'm5' or 'rossmann'.")
