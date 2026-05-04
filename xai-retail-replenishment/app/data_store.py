"""
Global data store for the XAI Retail Replenishment dashboard — V4.

Loaded once at app startup via load(). Supports switching between
7-day / 14-day / 28-day windows AND between LightGBM / XGBoost /
Random Forest / Neural Network model types.

Forecast q50 values come from the pre-computed future predictions CSVs
(data/future/*/predictions_2016_04_30.csv) — not from re-running the
model on test data at startup.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import math
import joblib
import numpy as np
import pandas as pd
import shap

from xai.uncertainty import compute_sku_residual_std, compute_prediction_interval
from xai.local_shap import compute_local_shap
from decision.safety_stock import safety_stock_quantile
from decision.replenishment_rules import generate_replenishment_card

# ── paths ─────────────────────────────────────────────────────────────────────

_REPORTS   = _ROOT / "artifacts" / "reports"
_MODELS    = _ROOT / "artifacts" / "models"
_FUTURE    = _ROOT / "data" / "future"
_PROCESSED = _ROOT / "data" / "processed"

# ── model configs ─────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "7d": {
        "data_dir":   _PROCESSED / "7_Day_Dataset",
        "future_csv": _FUTURE / "7_days_window" / "predictions_2016_04_30.csv",
        "models_dir": _MODELS / "7_day_models",
        "suffix":     "7d",
        "target":     "aggregated_sales_7",
        "label":      "7-day forecast",
        "horizon":    7,
        "shap_file":  "global_shap_importance_7d_v4.csv",
        "audit_file": "feature_quality_audit_7d_v4.csv",
    },
    "14d": {
        "data_dir":   _PROCESSED / "14_Day_Dataset",
        "future_csv": _FUTURE / "14_days_window" / "predictions_2016_04_30.csv",
        "models_dir": _MODELS / "14_day_models",
        "suffix":     "14d",
        "target":     "aggregated_sales_14",
        "label":      "14-day forecast",
        "horizon":    14,
        "shap_file":  "global_shap_importance_14d_v4.csv",
        "audit_file": "feature_quality_audit_14d_v4.csv",
    },
    "28d": {
        "data_dir":   _PROCESSED / "28_Day_Dataset",
        "future_csv": _FUTURE / "28_days_window" / "predictions_2016_04_30.csv",
        "models_dir": _MODELS / "28_day_models",
        "suffix":     "28d",
        "target":     "aggregated_sales_28",
        "label":      "28-day forecast",
        "horizon":    28,
        "shap_file":  "global_shap_importance_28d_v4.csv",
        "audit_file": "feature_quality_audit_28d_v4.csv",
    },
}

# Model type labels shown in the UI dropdown
MODEL_TYPES = {
    "lightgbm":       "LightGBM",
    "xgboost":        "XGBoost",
    "random_forest":  "Random Forest",
    "neural_network": "Neural Network",
}

# Column in the future predictions CSV for each model type
_PRED_COL = {
    "lightgbm":       "pred_lightgbm",
    "xgboost":        "pred_xgboost",
    "random_forest":  "pred_random_forest",
    "neural_network": "pred_neural_network",
}

# ── business constants ────────────────────────────────────────────────────────

LEAD_TIME    = 7
UNIT_MARGIN  = 3.50
HOLDING_COST = 0.80

# ── feature display labels ───────────────────────────────────────────────────
# Maps raw feature names (any window suffix stripped) to human-readable labels.

_FEATURE_LABELS = {
    "is_month_end":            "Month End",
    "aggregated_sell_price":   "Sell Price",
    "snap_ca":                 "SNAP CA",
    "event_christmas":         "Christmas",
    "event_easter":            "Easter",
    "event_eid_al_fitr":       "Eid al-Fitr",
    "event_eid_al_adha":       "Eid al-Adha",
    "event_fathers_day":       "Father's Day",
    "event_halloween":         "Halloween",
    "event_mothers_day":       "Mother's Day",
    "event_newyear":           "New Year",
    "event_orthodox_christmas":"Orthodox Christmas",
    "event_orthodox_easter":   "Orthodox Easter",
    "event_ramadan_starts":    "Ramadan",
    "event_thanksgiving":      "Thanksgiving",
    "event_valentines_day":    "Valentine's Day",
    "event_superbowl":         "Super Bowl",
    "event_independence_day":  "Independence Day",
    "event_memorial_day":      "Memorial Day",
    "event_labor_day":         "Labor Day",
    "event_mlk_day":           "MLK Day",
    "event_presidents_day":    "Presidents' Day",
    "event_columbus_day":      "Columbus Day",
    "event_veterans_day":      "Veterans Day",
    "event_st_patricks_day":   "St. Patrick's Day",
    "event_cinco_de_mayo":     "Cinco de Mayo",
    "event_chanukah":          "Chanukah",
    "event_lent_start":        "Lent Start",
    "event_lent_week2":        "Lent Week 2",
    "event_pesach_end":        "Pesach End",
    "event_purim_end":         "Purim End",
}


def feature_label(name: str) -> str:
    """Convert a raw feature name to a human-readable label."""
    import re
    # Strip sklearn prefixes
    name = re.sub(r"^(num__|cat__)", "", str(name))
    # Strip window suffix (_7, _14, _28)
    base = re.sub(r"_(\d+)$", "", name)
    return _FEATURE_LABELS.get(base, _FEATURE_LABELS.get(name, name.replace("_", " ").title()))


# ── forecast date context ─────────────────────────────────────────────────────
# These are fixed: the dataset ends Apr 23 2016, future predictions start Apr 30

DATA_LAST_DATE   = "Apr 23, 2016"
FORECAST_START   = pd.Timestamp("2016-04-30")


def _fmt(ts: pd.Timestamp, year: bool = False) -> str:
    s = ts.strftime("%b %d, %Y") if year else ts.strftime("%b %d")
    # remove leading zero from day (e.g. "May 06" → "May 6")
    return s.replace(" 0", " ")


def forecast_range_str() -> str:
    """Return e.g. 'Apr 30 – May 6, 2016' for the current window."""
    start = _fmt(FORECAST_START)
    end   = _fmt(FORECAST_START + pd.Timedelta(days=HORIZON - 1), year=True)
    return f"{start} – {end}"

# ── module-level state ────────────────────────────────────────────────────────

current_model_key:  str          = "7d"
current_model_type: str          = "lightgbm"
model:              object        = None
train_df:           pd.DataFrame  = None
test_df:            pd.DataFrame  = None
full_df:            pd.DataFrame  = None
X_test:             pd.DataFrame  = None
X_future:           pd.DataFrame  = None
feature_cols:       list[str]     = []
TARGET:             str           = "aggregated_sales_7"
HORIZON:            int           = 7
LAG_COL:            str           = ""

SKU_LIST:           list[str]     = []
forecasts:          dict          = {}
sku_std:            pd.Series     = None
cards_df:           pd.DataFrame  = None
global_shap_df:     pd.DataFrame  = None
feature_audit_df:   pd.DataFrame  = None
stockout_risk_df:   pd.DataFrame  = None
cold_start_df:      pd.DataFrame  = None
subgroup_eval_df:   pd.DataFrame  = None
interval_coverage:  float         = None
confidence_dist:    dict          = {}

_train_preds:      np.ndarray = None
_test_preds:       np.ndarray = None
_loaded:           bool       = False
_local_shap_cache: dict[str, shap.Explanation] = {}


# ── public API ────────────────────────────────────────────────────────────────

def is_loaded() -> bool:
    return _loaded


def load(model_key: str = "7d", model_type: str = "lightgbm") -> None:
    """Populate all module-level state. Idempotent for same key+type combo."""
    global model, train_df, test_df, full_df, X_test, X_future, feature_cols
    global TARGET, HORIZON, LAG_COL
    global SKU_LIST, forecasts, sku_std, cards_df
    global global_shap_df, feature_audit_df, stockout_risk_df
    global cold_start_df, subgroup_eval_df, interval_coverage, confidence_dist
    global _train_preds, _test_preds, _loaded
    global current_model_key, current_model_type

    if _loaded and model_key == current_model_key and model_type == current_model_type:
        return

    cfg    = MODEL_CONFIGS[model_key]
    TARGET = cfg["target"]
    HORIZON= cfg["horizon"]
    LAG_COL= ""   # V4 datasets have no lag column

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model_file = cfg["models_dir"] / f"{model_type}_{cfg['suffix']}_tuned.joblib"
    print(f"[data_store] Loading {cfg['label']} · {MODEL_TYPES[model_type]}…")
    model = joblib.load(model_file)

    # ── 2. Load processed datasets ────────────────────────────────────────────
    train_df = pd.read_csv(cfg["data_dir"] / "train.csv")
    test_df  = pd.read_csv(cfg["data_dir"] / "test.csv")

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    if "date" in full_df.columns:
        full_df["date"] = pd.to_datetime(full_df["date"], errors="coerce")
    if "date" in train_df.columns:
        train_df["date"] = pd.to_datetime(train_df["date"], errors="coerce")
    if "date" in test_df.columns:
        test_df["date"]  = pd.to_datetime(test_df["date"],  errors="coerce")

    # Feature cols: everything except target, item_id, date
    feature_cols = [c for c in train_df.columns if c not in (TARGET, "item_id", "date")]
    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()
    y_train = train_df[TARGET].copy()

    # ── 3. Compute train predictions for uncertainty (residual std) ───────────
    print("[data_store] Computing per-SKU residual std…")
    _train_preds = model.predict(X_train)
    _test_preds  = model.predict(X_test)
    sku_std      = compute_sku_residual_std(y_train, _train_preds, train_df["item_id"])
    global_std   = float(sku_std.mean())

    # ── 4. Load future predictions CSV (q50 source) ───────────────────────────
    print("[data_store] Loading future predictions…")
    future_preds = pd.read_csv(cfg["future_csv"])
    pred_col     = _PRED_COL[model_type]

    # Load the future feature CSV for What-If / SHAP on future data
    future_feat_csv = cfg["future_csv"].parent / "future_2016_04_30.csv"
    if future_feat_csv.exists():
        future_feat_df = pd.read_csv(future_feat_csv)
        X_future = future_feat_df[feature_cols].copy() if all(
            c in future_feat_df.columns for c in feature_cols
        ) else X_test.copy()
    else:
        X_future = X_test.copy()

    # ── 5. Build forecasts dict from future predictions ───────────────────────
    print("[data_store] Building forecasts and replenishment cards…")
    _forecasts, _cards = {}, []
    _rng = np.random.default_rng(seed=42)

    for _, row in future_preds.iterrows():
        sku   = row["item_id"]
        q50   = float(row[pred_col])
        std_v = float(sku_std.get(sku, global_std))

        # Q10/Q90 from future q50 ± 1.282σ (σ from training residuals)
        iv = compute_prediction_interval(q50, std_v)
        iv["q50"] = q50   # keep the ceiled future prediction as q50

        # Safety stock: consistent across all windows via √(lead_time/horizon)
        ss = safety_stock_quantile(iv["q50"], iv["q90"],
                                   lead_time_days=LEAD_TIME,
                                   forecast_horizon=HORIZON)

        # Stock-on-hand: anchored to demand + safety stock, scaled by random multiplier
        # multiplier 0.5 → understocked (CRITICAL), 2.0 → well stocked (LOW)
        avg_daily = q50 / float(HORIZON) if HORIZON > 0 else 0.0
        multiplier = float(_rng.uniform(0.5, 2.0))
        soh = math.ceil((avg_daily * LEAD_TIME) + (ss * multiplier))

        card = generate_replenishment_card(sku, iv, soh, LEAD_TIME, ss,
                                           forecast_horizon=HORIZON)
        _forecasts[sku] = iv
        _cards.append(card)

    forecasts = _forecasts
    cards_df  = pd.DataFrame(_cards)
    SKU_LIST  = sorted(forecasts.keys())

    # ── 6. Interval coverage on test set ─────────────────────────────────────
    print("[data_store] Computing reliability metrics…")
    from collections import Counter
    from xai.uncertainty import confidence_label, detect_cold_start_skus, subgroup_evaluation

    _q10_list, _q90_list, _conf_list = [], [], []
    for pred, sku in zip(_test_preds, test_df["item_id"]):
        std_v = float(sku_std.get(sku, global_std))
        iv    = compute_prediction_interval(float(pred), std_v)
        _q10_list.append(iv["q10"])
        _q90_list.append(iv["q90"])
        _conf_list.append(confidence_label(iv["width"], iv["q50"]))

    _y_test  = test_df[TARGET].values
    _covered = ((_y_test >= np.array(_q10_list)) & (_y_test <= np.array(_q90_list))).mean()
    interval_coverage = round(float(_covered) * 100, 1)
    confidence_dist   = dict(Counter(_conf_list))

    # Subgroup evaluation by category
    _cat_groups  = test_df["item_id"].apply(lambda s: "_".join(s.split("_")[:2]))
    subgroup_eval_df = subgroup_evaluation(_y_test, _test_preds, _cat_groups)

    # Cold-start (V4 datasets may not have weeks_since_first_seen)
    try:
        cold_start_df = detect_cold_start_skus(
            train_df, sku_col="item_id",
            weeks_col="weeks_since_first_seen", min_history_weeks=13,
        )
    except Exception:
        cold_start_df = pd.DataFrame(columns=["sku_id", "max_weeks_seen", "is_cold_start"])

    # Stockout risk (V4 has no lag column — skip)
    stockout_risk_df = pd.DataFrame()

    # ── 7. Global SHAP importance (model-type-specific, lazy) ───────────────
    _shap_path = _REPORTS / f"global_shap_{cfg['suffix']}_{model_type}.csv"
    if _shap_path.exists():
        print(f"[data_store] Loading global SHAP for {MODEL_TYPES[model_type]}…")
        global_shap_df = pd.read_csv(_shap_path)
    else:
        print(f"[data_store] SHAP file not found for {MODEL_TYPES[model_type]} — will compute on first visit.")
        global_shap_df = None

    # ── 8. Feature quality audit (lazy — skip if file missing) ───────────────
    _audit_path = _REPORTS / cfg["audit_file"]
    if _audit_path.exists():
        print("[data_store] Loading feature quality audit…")
        feature_audit_df = pd.read_csv(_audit_path)
    else:
        print("[data_store] Audit file not found — skipping.")
        feature_audit_df = None

    _local_shap_cache.clear()
    current_model_key  = model_key
    current_model_type = model_type
    _loaded = True
    print(f"[data_store] Ready. {len(SKU_LIST)} SKUs · {cfg['label']} · {MODEL_TYPES[model_type]}.")


def reload(model_key: str, model_type: str = "lightgbm") -> None:
    """Switch to a different window or model type and reload all state."""
    global _loaded
    _loaded = False
    load(model_key, model_type)


def get_sku_X_row(sku_id: str) -> pd.DataFrame:
    """Return the future feature row for a SKU (used by What-If simulator)."""
    if X_future is not None and len(X_future) > 0:
        future_feat_csv = MODEL_CONFIGS[current_model_key]["future_csv"].parent / "future_2016_04_30.csv"
        if future_feat_csv.exists():
            fdf  = pd.read_csv(future_feat_csv)
            mask = fdf["item_id"] == sku_id
            if mask.any():
                return fdf[mask][feature_cols].iloc[[0]]
    # Fallback: last test row
    mask = test_df["item_id"] == sku_id
    if not mask.any():
        raise KeyError(f"SKU '{sku_id}' not found.")
    return X_test[mask].iloc[[-1]]


def get_sku_test_rows(sku_id: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return historical test rows and predictions for a SKU (time series plot)."""
    mask      = test_df["item_id"] == sku_id
    sku_df    = test_df[mask].copy()
    sku_preds = _test_preds[mask.values]
    return sku_df, sku_preds


def load_second_model(model_type: str):
    """Load a different model type for the current window without replacing ds.model."""
    cfg  = MODEL_CONFIGS[current_model_key]
    path = cfg["models_dir"] / f"{model_type}_{cfg['suffix']}_tuned.joblib"
    return joblib.load(path)


def get_local_shap(sku_id: str) -> shap.Explanation:
    """Compute (or retrieve from cache) the local SHAP explanation for a SKU."""
    if sku_id in _local_shap_cache:
        return _local_shap_cache[sku_id]
    mask    = test_df["item_id"] == sku_id
    indices = np.where(mask.values)[0]
    if len(indices) == 0:
        raise KeyError(f"SKU '{sku_id}' not found in test set.")
    explanation = compute_local_shap(model, X_test, idx=int(indices[0]))
    _local_shap_cache[sku_id] = explanation
    return explanation
