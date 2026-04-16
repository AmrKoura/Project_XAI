"""Rebuild lag/rolling sales features for processed horizon datasets.

Targets directories under data/processed where train/val/test files exist and
schema contains one aggregated target column: aggregated_sales_<days>.

For each dataset, this script recomputes all columns matching:
- sales_lag_<days>
- sales_roll_mean_<days>
- sales_roll_std_<days>

The <days> suffix is interpreted as day horizon and converted to row counts
using inferred median date step per item_id.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


LAG_RE = re.compile(r"^sales_lag_(\d+)$")
ROLL_MEAN_RE = re.compile(r"^sales_roll_mean_(\d+)$")
ROLL_STD_RE = re.compile(r"^sales_roll_std_(\d+)$")


def infer_step_days(df: pd.DataFrame, group_col: str = "item_id") -> int:
    diffs = df.groupby(group_col, sort=False)["date"].diff().dt.days.dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1
    return max(1, int(round(float(diffs.median()))))


def days_to_rows(days: int, step_days: int) -> int:
    return max(1, int(np.ceil(days / step_days)))


def find_aggregated_target(columns: list[str]) -> str:
    targets = [c for c in columns if re.match(r"^aggregated_sales_\d+$", c)]
    if len(targets) != 1:
        raise ValueError(f"Expected exactly 1 aggregated_sales_<days> target, got: {targets}")
    return targets[0]


def rebuild_dataset_dir(dataset_dir: Path) -> None:
    split_paths = {name: dataset_dir / f"{name}.csv" for name in ("train", "val", "test")}
    for name, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} split: {path}")

    parts = []
    for split_name, path in split_paths.items():
        df = pd.read_csv(path)
        df["_split"] = split_name
        parts.append(df)

    all_df = pd.concat(parts, ignore_index=True)
    if "item_id" not in all_df.columns or "date" not in all_df.columns:
        raise ValueError(f"{dataset_dir}: requires item_id and date columns")

    target_col = find_aggregated_target(list(all_df.columns))

    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce")
    all_df = all_df[all_df["date"].notna()].copy()
    all_df = all_df.sort_values(["item_id", "date"]).reset_index(drop=True)

    step_days = infer_step_days(all_df, "item_id")

    grouped = all_df.groupby("item_id", sort=False)[target_col]

    lag_cols = [c for c in all_df.columns if LAG_RE.match(c)]
    mean_cols = [c for c in all_df.columns if ROLL_MEAN_RE.match(c)]
    std_cols = [c for c in all_df.columns if ROLL_STD_RE.match(c)]

    for col in lag_cols:
        horizon_days = int(LAG_RE.match(col).group(1))
        lag_rows = days_to_rows(horizon_days, step_days)
        all_df[col] = grouped.shift(lag_rows)

    for col in mean_cols:
        horizon_days = int(ROLL_MEAN_RE.match(col).group(1))
        window_rows = days_to_rows(horizon_days, step_days)
        all_df[col] = grouped.transform(lambda x: x.shift(1).rolling(window_rows, min_periods=1).mean())

    for col in std_cols:
        horizon_days = int(ROLL_STD_RE.match(col).group(1))
        window_rows = days_to_rows(horizon_days, step_days)
        all_df[col] = grouped.transform(lambda x: x.shift(1).rolling(window_rows, min_periods=1).std())

    recalc_cols = lag_cols + mean_cols + std_cols
    if recalc_cols:
        all_df[recalc_cols] = all_df[recalc_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    for split_name in ("train", "val", "test"):
        out = all_df[all_df["_split"] == split_name].copy()
        out = out.drop(columns=["_split"])
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(split_paths[split_name], index=False)

    print(f"{dataset_dir.name}: step_days={step_days}, target={target_col}")
    print(f"  lag_cols={lag_cols}")
    print(f"  roll_mean_cols={mean_cols}")
    print(f"  roll_std_cols={std_cols}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    processed = root / "data" / "processed"

    targets = [
        processed / "lgbm_28",
        processed / "lgbm_14",
        processed / "LGBM_XGB_7_V3",
    ]

    for dataset_dir in targets:
        rebuild_dataset_dir(dataset_dir)

    print("Done rebuilding lag/rolling features for all target datasets.")


if __name__ == "__main__":
    main()
