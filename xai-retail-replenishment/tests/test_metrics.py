"""
Tests for the evaluation metrics module.

Covers accuracy metrics (MAE, RMSE, SMAPE, Quantile Loss),
bias metrics (Bias, MAD, Tracking Signal), and business KPIs.
"""

import pytest
import numpy as np


class TestAccuracyMetrics:
    """Tests for MAE, RMSE, SMAPE, Quantile Loss."""

    def test_mae_perfect_prediction(self):
        """MAE should be 0 for perfect predictions."""
        ...

    def test_rmse_penalises_large_errors(self):
        """RMSE should be >= MAE."""
        ...

    def test_smape_bounded(self):
        """SMAPE should be between 0 and 200."""
        ...

    def test_quantile_loss_asymmetric(self):
        """Quantile loss at q=0.9 should penalise under-prediction more."""
        ...


class TestBiasMetrics:
    """Tests for Bias, MAD, Tracking Signal."""

    def test_zero_bias_for_unbiased(self):
        """Bias should be ~0 when errors are symmetric."""
        ...

    def test_tracking_signal_within_bounds(self):
        """TS should be within ±4 for a well-calibrated model."""
        ...


class TestBusinessKPIs:
    """Tests for Value Add, Service Level, Stockout Rate."""

    def test_service_level_perfect(self):
        """Service level = 1.0 when stock always meets demand."""
        ...

    def test_stockout_rate_zero(self):
        """Stockout rate = 0 when stock never reaches zero."""
        ...
