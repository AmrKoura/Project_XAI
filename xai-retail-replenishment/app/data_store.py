"""
Global data store for the XAI Retail Replenishment dashboard.

Loaded once at app startup via load(). Supports switching between
7-day, 14-day, and 28-day forecast models via reload(model_key).

All pages and callbacks read from module-level state.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import shap

from xai.uncertainty import compute_sku_residual_std, compute_prediction_interval
from xai.local_shap import compute_local_shap
from decision.safety_stock import safety_stock_quantile
from decision.replenishment_rules import generate_replenishment_card

# ── model configs ─────────────────────────────────────────────────────────────

_REPORTS = _ROOT / "artifacts" / "reports"

MODEL_CONFIGS = {
    "7d": {
        "model_path": _ROOT / "artifacts" / "models" / "tuned_lgbm_7_v3.joblib",
        "data_dir":   _ROOT / "data" / "processed" / "LGBM_XGB_7_V3",
        "target":     "aggregated_sales_7",
        "label":      "7-day forecast",
        "horizon":    7,
        "shap_file":  "global_shap_importance_7d.csv",
        "audit_file": "feature_quality_audit_7d.csv",
        "lag_col":    "sales_lag_7",
    },
    "14d": {
        "model_path": _ROOT / "artifacts" / "models" / "untuned_lgbm_14.joblib",
        "data_dir":   _ROOT / "data" / "processed" / "lgbm_14",
        "target":     "aggregated_sales_14",
        "label":      "14-day forecast",
        "horizon":    14,
        "shap_file":  "global_shap_importance_14d.csv",
        "audit_file": "feature_quality_audit_14d.csv",
        "lag_col":    "sales_lag_7",
    },
    "28d": {
        "model_path": _ROOT / "artifacts" / "models" / "tuned_lgbm_28.joblib",
        "data_dir":   _ROOT / "data" / "processed" / "lgbm_28",
        "target":     "aggregated_sales_28",
        "label":      "28-day forecast",
        "horizon":    28,
        "shap_file":  "global_shap_importance_28d.csv",
        "audit_file": "feature_quality_audit_28d.csv",
        "lag_col":    "sales_lag_7",
    },
}

# ── business constants ────────────────────────────────────────────────────────

LEAD_TIME    = 7
UNIT_MARGIN  = 3.50
HOLDING_COST = 0.80

# ── module-level state ────────────────────────────────────────────────────────

current_model_key: str          = "7d"
model:             object        = None
train_df:          pd.DataFrame  = None
test_df:           pd.DataFrame  = None
full_df:           pd.DataFrame  = None
X_test:            pd.DataFrame  = None
feature_cols:      list[str]     = []
TARGET:            str           = "aggregated_sales_7"
HORIZON:           int           = 7
LAG_COL:           str           = "sales_lag_7"

SKU_LIST:          list[str]     = []
forecasts:         dict          = {}
sku_std:           pd.Series     = None
cards_df:          pd.DataFrame  = None
global_shap_df:    pd.DataFrame  = None
feature_audit_df:  pd.DataFrame  = None
stockout_risk_df:  pd.DataFrame  = None
cold_start_df:     pd.DataFrame  = None
subgroup_eval_df:  pd.DataFrame  = None
interval_coverage: float         = None
confidence_dist:   dict          = {}

_train_preds:  np.ndarray = None
_test_preds:   np.ndarray = None
_loaded:       bool       = False
_local_shap_cache: dict[str, shap.Explanation] = {}


# ── public API ────────────────────────────────────────────────────────────────

def is_loaded() -> bool:
    return _loaded


def load(model_key: str = "7d") -> None:
    """Populate all module-level state. Idempotent for same model key."""
    global model, train_df, test_df, full_df, X_test, feature_cols, TARGET, HORIZON, LAG_COL
    global SKU_LIST, forecasts, sku_std, cards_df
    global global_shap_df, feature_audit_df, stockout_risk_df
    global cold_start_df, subgroup_eval_df, interval_coverage, confidence_dist
    global _train_preds, _test_preds, _loaded, current_model_key

    if _loaded and model_key == current_model_key:
        return

    cfg = MODEL_CONFIGS[model_key]
    TARGET  = cfg["target"]
    HORIZON = cfg["horizon"]
    LAG_COL = cfg["lag_col"]

    print(f"[data_store] Loading {cfg['label']}…")
    model    = joblib.load(cfg["model_path"])
    train_df = pd.read_csv(cfg["data_dir"] / "train.csv")
    test_df  = pd.read_csv(cfg["data_dir"] / "test.csv")

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    if "date" in full_df.columns:
        full_df["date"] = pd.to_datetime(full_df["date"], errors="coerce")

    # Derive feature columns from the model's preprocessor so that models
    # which use item_id/date as categoricals (e.g. 28d) are handled correctly.
    prep = model.named_steps.get("prep") or model.named_steps.get("preprocessor")
    if prep is not None and hasattr(prep, "transformers_"):
        feature_cols = []
        for _, _, cols in prep.transformers_:
            feature_cols.extend(cols)
    else:
        # Fallback: drop only the target
        feature_cols = [c for c in train_df.columns if c != TARGET]

    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()
    y_train = train_df[TARGET].copy()

    print("[data_store] Running inference…")
    _train_preds = model.predict(X_train)
    _test_preds  = model.predict(X_test)

    print("[data_store] Computing per-SKU uncertainty…")
    sku_std    = compute_sku_residual_std(y_train, _train_preds, train_df["item_id"])
    global_std = float(sku_std.mean())

    print("[data_store] Computing Q9 reliability metrics…")
    from collections import Counter
    from xai.uncertainty import (
        compute_prediction_interval, confidence_label,
        detect_cold_start_skus, subgroup_evaluation,
    )

    # Cold-start detection
    cold_start_df = detect_cold_start_skus(
        train_df, sku_col="item_id",
        weeks_col="weeks_since_first_seen", min_history_weeks=13,
    )

    # Confidence distribution + interval coverage across all test rows
    _q10_list, _q90_list, _conf_list = [], [], []
    for pred, sku in zip(_test_preds, test_df["item_id"]):
        std_v = float(sku_std.get(sku, global_std))
        iv    = compute_prediction_interval(float(pred), std_v)
        _q10_list.append(iv["q10"])
        _q90_list.append(iv["q90"])
        _conf_list.append(confidence_label(iv["width"], iv["q50"]))

    _y_test   = test_df[TARGET].values
    _covered  = ((_y_test >= np.array(_q10_list)) & (_y_test <= np.array(_q90_list))).mean()
    interval_coverage = round(float(_covered) * 100, 1)
    confidence_dist   = dict(Counter(_conf_list))

    # Subgroup evaluation by product category
    _cat_groups = test_df["item_id"].apply(lambda s: "_".join(s.split("_")[:2]))
    subgroup_eval_df = subgroup_evaluation(_y_test, _test_preds, _cat_groups)

    print("[data_store] Building forecasts and replenishment cards…")
    _cards, _forecasts = [], {}
    # Simulate stock-on-hand as a random fraction of forecast (0 – 2× q50).
    # Seeded so the dashboard shows the same values every run.
    _rng      = np.random.default_rng(seed=42)
    first_idx = test_df.groupby("item_id").apply(lambda g: g.index[0])
    for sku in first_idx.index:
        row_idx  = first_idx[sku]
        pred     = float(_test_preds[test_df.index.get_loc(row_idx)])
        std_v    = float(sku_std.get(sku, global_std))
        iv       = compute_prediction_interval(pred, std_v)
        ss       = safety_stock_quantile(iv["q50"], iv["q90"], LEAD_TIME)
        # Random stock: uniform between 0 and 2× forecast — gives realistic mix
        # of CRITICAL / HIGH / LOW urgency across SKUs.
        soh      = float(_rng.uniform(0.0, 2.0 * max(iv["q50"], 1.0)))
        card     = generate_replenishment_card(sku, iv, soh, LEAD_TIME, ss)
        _forecasts[sku] = iv
        _cards.append(card)

    forecasts = _forecasts
    cards_df  = pd.DataFrame(_cards)
    SKU_LIST  = sorted(forecasts.keys())

    print("[data_store] Loading global SHAP importance…")
    _shap_path = _REPORTS / cfg["shap_file"]
    if _shap_path.exists():
        global_shap_df = pd.read_csv(_shap_path)
    else:
        print("[data_store] SHAP not found — computing (may take ~60 s)…")
        from xai.global_shap import compute_global_shap, rank_feature_importance
        shap_vals      = compute_global_shap(model, X_test, max_samples=500)
        global_shap_df = rank_feature_importance(shap_vals)
        global_shap_df.to_csv(_shap_path, index=False)

    print("[data_store] Loading feature quality audit…")
    _audit_path = _REPORTS / cfg["audit_file"]
    if _audit_path.exists():
        feature_audit_df = pd.read_csv(_audit_path)
    else:
        print("[data_store] Audit not found — computing…")
        from xai.global_shap import feature_quality_audit
        feature_audit_df = feature_quality_audit(X_test)
        feature_audit_df.to_csv(_audit_path, index=False)

    print("[data_store] Computing stockout risk summary…")
    if LAG_COL in full_df.columns:
        from xai.stockout_analysis import flag_potential_stockouts
        _flagged = flag_potential_stockouts(full_df, lag_col=LAG_COL)
        stockout_risk_df = (
            _flagged.groupby("item_id")
            .agg(
                total_periods    = ("is_potential_stockout", "count"),
                stockout_periods = ("is_potential_stockout", "sum"),
            )
            .assign(stockout_rate=lambda d: d["stockout_periods"] / d["total_periods"])
            .sort_values("stockout_rate", ascending=False)
            .reset_index()
        )
    else:
        stockout_risk_df = pd.DataFrame()

    _local_shap_cache.clear()
    current_model_key = model_key
    _loaded = True
    print(f"[data_store] Ready. {len(SKU_LIST)} SKUs · {cfg['label']}.")


def reload(model_key: str) -> None:
    """Switch to a different model and reload all state."""
    global _loaded
    _loaded = False
    load(model_key)


def get_sku_X_row(sku_id: str) -> pd.DataFrame:
    mask = test_df["item_id"] == sku_id
    if not mask.any():
        raise KeyError(f"SKU '{sku_id}' not found in test set.")
    return X_test[mask].iloc[[0]]


def get_sku_test_rows(sku_id: str) -> tuple[pd.DataFrame, np.ndarray]:
    mask      = test_df["item_id"] == sku_id
    sku_df    = test_df[mask].copy()
    sku_preds = _test_preds[mask.values]
    return sku_df, sku_preds


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
