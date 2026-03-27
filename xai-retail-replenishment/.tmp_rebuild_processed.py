import pandas as pd
from pathlib import Path

base = Path("data/processed")
src = base / "full_merged_clean.csv"

if not src.exists():
    raise FileNotFoundError(f"Missing source file: {src}")

df = pd.read_csv(src, low_memory=False)

date_col = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
if date_col is None:
    raise ValueError("No date column found (expected date or Date).")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df[df[date_col].notna()].copy()
df = df.sort_values(date_col).reset_index(drop=True)

# overwrite canonical full merged
(df).to_csv(base / "full_merged.csv", index=False)

# date-based split 80/10/10
unique_dates = df[date_col].dropna().sort_values().unique()
n = len(unique_dates)
if n < 3:
    raise ValueError(f"Not enough unique dates to split safely: {n}")

train_end = max(1, int(n * 0.8))
val_end = max(train_end + 1, int(n * 0.9))
val_end = min(val_end, n - 1)

train_dates = set(unique_dates[:train_end])
val_dates = set(unique_dates[train_end:val_end])
test_dates = set(unique_dates[val_end:])

train = df[df[date_col].isin(train_dates)].copy()
val = df[df[date_col].isin(val_dates)].copy()
test = df[df[date_col].isin(test_dates)].copy()

train.to_csv(base / "train.csv", index=False)
val.to_csv(base / "val.csv", index=False)
test.to_csv(base / "test.csv", index=False)

inventory_cols = [
    "stock_on_hand",
    "lead_time_days",
    "reorder_cost",
    "supplier_min_order_qty",
    "supplier_order_multiple",
]

forecasting = df.drop(columns=[c for c in inventory_cols if c in df.columns])
forecasting.to_csv(base / "forecasting_dataset.csv", index=False)

replenishment_candidates = [
    "Date", "date", "Store", "store_id", "item_id", "sku_id",
    "Sales", "sales",
    "stock_on_hand", "lead_time_days", "reorder_cost",
    "supplier_min_order_qty", "supplier_order_multiple",
]
replenishment_cols = [c for c in replenishment_candidates if c in df.columns]
replenishment = df[replenishment_cols].copy()
replenishment.to_csv(base / "replenishment_dataset.csv", index=False)

print("Done")
print("full_merged", df.shape)
print("train", train.shape, "val", val.shape, "test", test.shape)
print("forecasting_dataset", forecasting.shape)
print("replenishment_dataset", replenishment.shape)
