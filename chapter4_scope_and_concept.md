# Chapter 4: Scope & Concept

This chapter highlights the boundaries of the demand forecasting and replenishment planning system and provides a blueprint of its design. Section 4.1 addresses the project scope — what the system covers, what it excludes, and the reason behind those decisions. Section 4.2 covers the system concept: the architecture, data pipeline, the forecasting and explainability components.

---

## 4.1 Project Scope

### 4.1.1 Domain and Problem Boundary

This thesis focuses on the domain of grocery and fast-moving consumer goods (FMCG) retail, in particular the problem of **SKU-level demand forecasting and replenishment planning**. The system is tailored for inventory planners and category managers who need to make reordering decisions weekly or bi-weekly, and who may not have a solid background in forecasting.

The core problem being addressed is the **actionability and explainability gap** — the bridge between what a forecasting model outputs and what the user can understand, trust, and act upon.

The system operates at the **SKU-store level**: it produces individual forecasting and replenishment recommendations for each product and explains why in a human-readable way. This granularity was chosen because replenishment decisions differ for each SKU depending on its location, region, and customer demographics.

---

### 4.1.2 Data Scope

The system is developed using a combination of publicly available benchmark datasets and, later, real company data. The public datasets are used for model development, while the private company data allows the system to be stress-tested and validated by real users.

The following public datasets will be used in the development phase:

- **M5 Forecasting (Walmart, Kaggle):** A dataset containing daily unit sales for 3,049 products across 10 Walmart stores in three U.S. states, spanning 5 years. This is the **primary dataset**.
- **Rossmann Store Sales (Kaggle):** Contains daily sales for 1,115 Rossmann drug stores across Germany, including promotional indicators, store type, and competitor distance features. This is the **secondary dataset** and provides a perspective on promotional demand effects in a European retail context.
- **Synthetic Data Augmentation:** In case any of the datasets lack critical inventory-specific fields (stock on hand, lead time, reorder cost, supplier constraints), synthetic values will be generated using statistically grounded simulation.

In the **final validation phase**, the trained system will be applied to a private retail company dataset to evaluate the system's generalizability and assess the practicality of the NLM used to output natural language explanations. Due to confidentiality constraints, results from this phase will be reported in aggregate form.

---

#### Table 4.1 — Walmart M5 Dataset File Breakdown

| File Name | Size | What It Shows | Key Columns | Mapping | Notes |
|---|---|---|---|---|---|
| `calendar.csv` | 1,969 rows | Walmart's calendar — maps each `d_*` key to real dates and event features | `date`, `wm_yr_wk`, `weekday`, `wday`, `month`, `year`, `d`, `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2`, `snap_CA`, `snap_TX`, `snap_WI` | Sales table uses `d_*` as key; `calendar.csv` maps each `d_*` to real date/event features; `sell_prices.csv` joins via `store_id + item_id + wm_yr_wk` | `sample_submission.csv` is for Kaggle format only. `sell_train_validation.csv` and `sell_train_evaluation.csv` share the same headers; evaluation adds extra days. |
| `sell_prices.csv` | 6.8M rows | Item price per store per Walmart week | `store_id`, `item_id`, `wm_yr_wk`, `sell_price` | | |
| `sell_train_validation.csv` | ~30,490 rows + many day cols | Each row is one SKU-store series; each `d_*` column is unit sales for that day | `id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`, `d_1`, `d_2`, ... up to validation horizon | | |
| `sell_train_evaluation.csv` | ~30,490 rows + many day cols | Same as `sell_train_validation` but extended by additional days | Same as above | | |
| `sample_submissions.csv` | — | 28 forecast days Kaggle expects for submission (no real data) | `id`, `F1`...`F28` | | |

---

#### Table 4.1 (cont.) — Rossmann Dataset File Breakdown

| File Name | Size | What It Shows | Key Columns | Mapping | Notes |
|---|---|---|---|---|---|
| `train.csv` | 1,017,209 rows | Main training table — daily sales per store with customer count, open flag, promo, and holiday flags | `Store`, `DayOfWeek`, `Date`, `Sales`, `Customers`, `Open`, `Promo`, `StateHoliday`, `SchoolHoliday` | `train.csv` and `store.csv` connected by `Store` key | 1. `test.csv` has no sales (Kaggle prediction only — not for model training). 2. `PromoInterval` shows months the promo was running. 3. Only `train.csv` and `store.csv` are needed for the proposed work. |
| `test.csv` | 41,088 rows | Test set — no sales values (to be predicted) | `Id`, `Store`, `DayOfWeek`, `Date`, `Open`, `Promo`, `StateHoliday`, `SchoolHoliday` | | |
| `store.csv` | 1,115 rows | Store-level context — store type, assortment type, competition info, and promo details | `Store`, `StoreType`, `Assortment`, `CompetitionDistance`, `CompetitionOpenSinceMonth`, `CompetitionOpenSinceYear`, `Promo2`, `Promo2SinceWeek`, `Promo2SinceYear`, `PromoInterval` | | |
| `sample_submission.csv` | — | Template for plugging in sales prediction numbers | `Id`, `Sales` | | |

---

#### Table 4.3 — Synthetic Missing Data Fields

| Missing Data | How Will It Be Calculated | Why Do We Need It | How Will We Use It |
|---|---|---|---|
| **Stock on Hand** | Derived from sales history (rolling cumulative with simulated replenishment cycles) | A forecast alone cannot tell you how much to order unless you know current inventory. | `OrderQty = max(0, Forecast × lead time + Safety Stock − Stock On Hand)`. Also used for stockout analysis (Q7) and service-level KPI. |
| **Lead Time** | Sampled from a realistic distribution (e.g. 5–14 days per store type) | Reorder must cover demand during supplier wait time, not just next-day demand. | Aggregate forecast across lead-time horizon, set reorder point, and run what-if scenarios in dashboard. |
| **Reorder Cost** | Assigned per store/product tier | Q10 needs the financial impact of under/over-ordering, not just units. | Cost simulation compares stockout penalty vs overstock/holding cost. |
| **Supplier Constraints** | Min order quantities per assortment type | Real suppliers enforce min order, order multiples, caps, and case-pack rules. | Raw recommended order is adjusted to a feasible order: min order quantity, multiple of pack size, optional max-cap constraints. |

---

### 4.1.3 What Is Within Scope

- SKU-level demand forecasting using **LightGBM** as primary model, benchmarked against XGBoost, ARIMA, and Exponential Smoothing.
- **Quantile Forecasting** to handle prediction uncertainty.
- **SHAP-based post-hoc explainability** applied to forecasts and replenishment recommendations, covering both local and global explanations.
- **Natural language generation** for non-technical users.
- **Replenishment quantity recommendations** taking into consideration both variable lead times and safety stock logic.
- **Interactive Python Dash Dashboard** with 4 views: forecast overview, SKU explorer, explanations panel, what-if simulator.
- A set of **10 explainability questions** that the system is designed to answer (see Section 4.2.3).
- **Model validation** across both public and private data.

---

### 4.1.4 What Is Outside Scope

| Out-of-Scope Area | Reason |
|---|---|
| **Real-time live data integration** | The system operates on historical data. Integrating real-time feeds increases system complexity significantly. |
| **Automated supplier communication and order execution** | The system generates recommendations but does not place orders automatically — human judgment is preserved as the final decision point. |
| **Non-FMCG product categories** | Slow-moving, high-value, or highly seasonal products require different demand dynamics and modeling approaches. |
| **Multi-location network optimization** | The system does not handle inventory allocation across a store network. Each store is treated independently. |

---

## 4.2 System Concept

### 4.2.2 The Forecasting Component

**LightGBM** is used as the main forecasting model because it:
- Can process large datasets faster than other forecasting models.
- Handles categorical features (e.g. Store ID and SKU) without requiring encoding.
- Supports quantile regression natively.

Three additional models are trained on the same features and datasets for comparison:
- **ARIMA** — classical statistical method.
- **Exponential Smoothing** — classical statistical method.
- **XGBoost** — one of the most commonly used ML models, included for direct comparison.

All methods are evaluated using the following metrics:

---

#### Table 4.4 — Evaluation KPIs

| Abbr. | Full Name | What It Measures |
|---|---|---|
| **Group 1 — Accuracy Metrics** | | |
| **MAE** | Mean Absolute Error | The average error between predicted and actual demand, in units. If MAE = 20, the model is off by 20 units on average. |
| **RMSE** | Root Mean Squared Error | Similar to MAE but penalizes large errors more heavily. A single huge miss raises RMSE significantly. |
| **SMAPE** | Symmetric Mean Absolute Percentage Error | Error as a percentage of demand, symmetrically. Used instead of MAPE because MAPE breaks when actual demand = 0, which is common in FMCG data. |
| **Group 2 — Bias Metrics** | | |
| **BIAS** | Forecast Bias | Mean of (forecast − actual) across all predictions. Positive = consistently over-predicts; Negative = consistently under-predicts. A biased model leads to systematic over- or under-ordering. |
| **MAD** | Mean Absolute Deviation | Average of absolute deviations between forecast and actual demand. Used directly in safety stock calculations — higher MAD means more buffer stock is needed. |
| **Group 3 — Business & Inventory KPIs** | | |
| **VA** | Value Add | Measures how much the ML model improves over a naive baseline (e.g., last week's sales as forecast). Expressed as % improvement in MAE or SMAPE. Directly justifies the use of a complex model. |
| **SL** | Service Level | Percentage of demand periods where stock was available to meet demand without a stockout. Directly measures the operational success of replenishment recommendations. |

---

### 4.2.3 The Explainability Component — The 10 Questions

The explainability system is structured around **10 planner questions**. Rather than exposing raw SHAP charts (which require interpretation), each question is answered using the most appropriate XAI technique and fed into a natural language model that outputs a plain-English briefing per SKU.

#### Table 4.5 — Planner Questions, Techniques, and Target Users

| Q | Planner Question | Technique Used | Target Users |
|---|---|---|---|
| **Q1** | Why is the system recommending a reorder for this SKU right now? | Local SHAP | Inventory Planner |
| **Q2** | What are the most important features driving demand across all SKUs? | Local + Global SHAP | Category Manager, Supply Chain Analyst |
| **Q3** | How confident is the model in this forecast? | Quantile Forecast + Prediction Intervals + Calibration | Inventory Planner, Supply Chain Analyst |
| **Q4** | What would happen to the forecast if a promotion were added or removed? | Counterfactual (PDP as support) | Category Manager, Retail Founder |
| **Q5** | Why is the reorder quantity higher for this SKU than a similar one? | Comparative SHAP | Inventory Planner, Category Manager |
| **Q6** | Is this SKU experiencing a demand spike or stable growth? | SHAP Grouped by Time Features | Inventory Planner, Category Manager |
| **Q7** | Could past stockouts have distorted the demand signal for this SKU? | SHAP on Stockout Flag + Counterfactual | Supply Chain Analyst |
| **Q8** | Are there any data quality issues affecting the model's reliability? | Global SHAP + Feature Audit | Supply Chain Analyst |
| **Q9** | Is the model less reliable for this SKU due to limited history? | Subgroup Evaluation + Uncertainty Analysis | Inventory Planner, Supply Chain Analyst |
| **Q10** | What is the financial cost of under- or over-ordering based on this forecast? | Probabilistic Simulation + Cost Impact Modeling | Retail Founder, Category Manager |

#### Example Natural Language Output

> *"Demand for this product is forecast to increase by approximately 18% next week, primarily driven by the scheduled end-of-aisle promotion. The model is moderately uncertain about this figure because this SKU has limited historical data during promotional periods (Q9). Based on a 7-day lead time and current stock of 120 units, a reorder of 340 units is recommended to maintain a 95% service level. Under-ordering by 20% would result in an estimated stockout cost of approximately 1,200 EGP (Q10)."*

---

### 4.2.4 The Replenishment Logic

The replenishment logic answers the question: **"How many units should I order right now?"**

To answer this, four inputs are required:

1. **How much will sell before new stock arrives?** — The LightGBM forecast over the lead-time horizon (e.g. if lead time = 7 days, forecast demand for the next 7 days).
2. **What is the safety buffer?** — Derived from forecast uncertainty (quantile interval width). A wider interval = higher uncertainty = larger buffer needed.
3. **How much stock is on hand right now?**
4. **How long until the order arrives?** — Used for what-if recalculation if lead time changes.

#### Order Quantity Formula

```
Order Quantity = Forecasted Demand during Lead Time
              + Safety Stock Buffer (from uncertainty)
              - Current Stock on Hand
```

**Example:** LightGBM predicts 500 units over the next 7 days. The quantile interval is wide (high uncertainty), so a buffer of 80 units is set. Current stock on hand = 120 units.

```
Order Quantity = 500 + 80 − 120 = 460 units
```

**Variable lead time handling:** If the supplier is running late (e.g. usual 7 days → actual 10 days), the system recalculates all three inputs on the 10-day horizon and outputs a new order quantity. This recalculation is handled in the **What-If Simulator** (see Section 4.2.5).

---

### 4.2.5 The Dashboard Interface

The dashboard is built in **Python Dash** and has 4 main pages:

| Page | File | Primary Function | Questions Addressed |
|---|---|---|---|
| **Dashboard** | `dashboard.py` | Landing page and primary overview. Displays demand forecasts and replenishment cards for all SKUs. Flags products with high uncertainty, imminent reorder triggers, and data quality concerns. | Q2, Q8 |
| **SKU Explorer** | `sku_explorer.py` | Detailed breakdown for a selected SKU. Shows forecast time series with quantile bands, local SHAP waterfall chart, and the natural language brief. | Q1, Q3, Q6 |
| **Explanations Panel** | `explanations.py` | Presents global SHAP summaries, Partial Dependence Plots (PDP), comparative SHAP across similar SKUs, stockout distortion analysis, and feature audit. | Q2, Q4, Q5, Q7, Q8 |
| **What-If Simulator** | `what_if.py` | Interactive scenario tool. Planners can modify promotion status, lead time, current stock level, and service level target to see how forecast, uncertainty, and replenishment recommendations change. | Q4, Q10 |

---

## 4.3 Summary

This chapter covered the scope and concept of the proposed system:

- **Scope:** SKU-level FMCG retail forecasting and replenishment. Public datasets (M5 Walmart, Rossmann) are used for training; private company data is used for stress-testing and real-world validation.
- **Architecture:** The system is built around 4 layers — data preprocessing, forecasting, explainability, and the dashboard interface.
- **Explainability:** Structured around **10 planner questions**, each answered by a dedicated analytical module using the most appropriate XAI technique (SHAP, Counterfactual, PDP, Uncertainty Analysis, Cost Simulation).
- **Output:** Each SKU receives a machine-generated plain-English briefing that summarizes the forecast, explains the uncertainty, and justifies the replenishment recommendation.
