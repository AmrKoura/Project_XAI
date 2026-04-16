"""Fix 28-day lag/rolling features in processed lgbm_28 splits.

Recomputes:
- sales_lag_28 as previous 28-day horizon converted to row offset from date step
- sales_roll_mean_28 as rolling mean over prior 28-day horizon converted to row window

The script reads train/val/test from data/processed/lgbm_28, combines them in
chronological order, recomputes the features per item_id, then writes the
corrected split files back to disk.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


def _infer_step_days(df: pd.DataFrame) -> int:
    diffs = df.groupby("item_id", sort=False)["date"].diff().dt.days.dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1
    return max(1, int(round(float(diffs.median()))))


def _days_to_rows(days: int, step_days: int) -> int:
    return max(1, int(np.ceil(days / step_days)))


def _recompute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out[out["date"].notna()].copy()
    out = out.sort_values(["item_id", "date"]).reset_index(drop=True)

    step_days = _infer_step_days(out)
    grouped = out.groupby("item_id", sort=False)["aggregated_sales_28"]

    # Previous 28 days in row units derived from date step.
    out["sales_lag_28"] = grouped.shift(_days_to_rows(28, step_days))

    # Rolling 28-day mean in row units derived from date step.
    out["sales_roll_mean_28"] = grouped.transform(
        lambda x: x.shift(1).rolling(_days_to_rows(28, step_days), min_periods=1).mean()
    )

    out[["sales_lag_28", "sales_roll_mean_28"]] = out[
        ["sales_lag_28", "sales_roll_mean_28"]
    ].fillna(0.0)

    return out


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "processed" / "lgbm_28"

    required = ["train.csv", "val.csv", "test.csv"]
    for name in required:
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"Missing required file: {data_dir / name}")

    frames: list[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        split_df = pd.read_csv(data_dir / f"{split_name}.csv")
        split_df["_split"] = split_name
        frames.append(split_df)

    all_df = pd.concat(frames, ignore_index=True)

    for col in ["item_id", "date", "aggregated_sales_28"]:
        if col not in all_df.columns:
            raise ValueError(f"Required column not found: {col}")

    fixed = _recompute_features(all_df)

    for split_name in ("train", "val", "test"):
        out = fixed[fixed["_split"] == split_name].copy()
        out = out.drop(columns=["_split"])
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(data_dir / f"{split_name}.csv", index=False)

    print("Updated files:")
    for split_name in ("train", "val", "test"):
        p = data_dir / f"{split_name}.csv"
        check = pd.read_csv(p, nrows=1)
        print(f" - {p} (cols={check.shape[1]})")


if __name__ == "__main__":
    main()
