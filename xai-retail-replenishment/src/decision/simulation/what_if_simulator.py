"""
What-if simulator.

Interactive scenario tool allowing planners to modify promotion status,
lead time, current stock, and service-level target, then see how the
forecast, uncertainty, and replenishment recommendations change.
"""

import pandas as pd
import numpy as np


def simulate_scenario(
    model: object,
    X_row: pd.Series,
    overrides: dict[str, float | int],
) -> dict:
    """Run a single what-if scenario with user-specified parameter changes.

    Parameters
    ----------
    model : object
        Trained forecasting model.
    X_row : pd.Series
        Baseline feature values for the SKU.
    overrides : dict[str, float | int]
        Features to override, e.g. ``{'promo': 1, 'lead_time': 10}``.

    Returns
    -------
    dict
        ``{'baseline_pred', 'scenario_pred', 'reorder_qty', 'safety_stock', ...}``
    """
    ...


def simulate_batch_scenarios(
    model: object,
    X_row: pd.Series,
    scenarios: list[dict[str, float | int]],
) -> pd.DataFrame:
    """Run multiple what-if scenarios and return a comparison table.

    Parameters
    ----------
    model : object
    X_row : pd.Series
    scenarios : list[dict]

    Returns
    -------
    pd.DataFrame
        One row per scenario with prediction and replenishment outputs.
    """
    ...


def generate_scenario_summary_text(scenario_result: dict) -> str:
    """Generate a natural-language summary of a what-if scenario.

    Parameters
    ----------
    scenario_result : dict

    Returns
    -------
    str
    """
    ...
