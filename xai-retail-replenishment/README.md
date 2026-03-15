# XAI-Driven Retail Replenishment System

An explainable AI system for SKU-level demand forecasting and replenishment planning
in grocery and FMCG retail.

## Overview

This project bridges the gap between machine learning forecast outputs and actionable,
trustworthy replenishment decisions. It is designed for inventory planners and category
managers who need to make weekly/bi-weekly reordering decisions without a deep
background in data science.

### Core Components

| Layer | Description |
|-------|-------------|
| **Data Ingestion & Preprocessing** | Raw data cleaning, feature engineering, synthetic field generation |
| **Forecasting Engine** | LightGBM (primary) with XGBoost, ARIMA, Exponential Smoothing benchmarks |
| **Explainability & Replenishment** | SHAP-based explanations answering 10 planner questions + NLG briefings |
| **Dashboard Interface** | Python Dash app with 4 views: Overview, SKU Explorer, Explanations, What-If |

### The 10 Planner Questions

| # | Question | Technique |
|---|----------|-----------|
| Q1 | Why reorder this SKU now? | Local SHAP |
| Q2 | Top features driving demand? | Global SHAP |
| Q3 | How confident is the forecast? | Quantile Forecasts + Calibration |
| Q4 | What if promotion changes? | Counterfactual + PDP |
| Q5 | Why more than a similar SKU? | Comparative SHAP |
| Q6 | Spike or stable growth? | Temporal SHAP |
| Q7 | Did stockouts distort data? | SHAP on Stockout Flag |
| Q8 | Any data quality issues? | Global SHAP + Feature Audit |
| Q9 | Less reliable due to limited history? | Subgroup Eval + Uncertainty |
| Q10 | Cost of under/over-ordering? | Probabilistic Simulation |

### Datasets

- **M5 Forecasting** (Walmart/Kaggle) — primary dataset
- **Rossmann Store Sales** (Kaggle) — secondary, European promotional context
- **Favorita Grocery Sales** (Kaggle) — validation in non-Western market
- **Private retail data** — final validation phase (reported in aggregate)

## Project Structure

```
xai-retail-replenishment/
├── config/          # YAML configuration files
├── data/            # Raw, interim, processed datasets
├── notebooks/       # Jupyter notebooks for EDA and analysis
├── src/             # Core source code
│   ├── data/        # Data loading and cleaning
│   ├── features/    # Feature engineering
│   ├── models/      # Training and evaluation
│   ├── xai/         # Explainability modules (10 questions)
│   ├── decision/    # Replenishment rules and simulation
│   ├── visualization/
│   └── utils/
├── app/             # Dash dashboard application
├── artifacts/       # Saved models, figures, reports
└── tests/           # Unit tests
```

## Quick Start

```bash
pip install -r requirements.txt
python app/app.py
```

## Evaluation Metrics

- **Accuracy**: MAE, RMSE, SMAPE, Quantile Loss
- **Bias**: Forecast Bias, MAD, Tracking Signal
- **Business KPIs**: Value Add, Service Level, Stockout Rate, Inventory Turnover
