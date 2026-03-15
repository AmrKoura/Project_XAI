"""
Tests for the feature engineering modules.

Covers time features, lag features, and promotional/price features
to ensure correct column creation and value ranges.
"""

import pytest
import pandas as pd
import numpy as np


class TestTimeFeatures:
    """Tests for src.features.time_features."""

    def test_add_basic_date_features_creates_expected_columns(self):
        """Verify that basic date features are added correctly."""
        ...

    def test_add_holiday_proximity_returns_nonnegative(self):
        """Holiday proximity should be >= 0."""
        ...

    def test_add_weekend_flag_binary(self):
        """Weekend flag should be 0 or 1."""
        ...


class TestLagFeatures:
    """Tests for src.features.lag_features."""

    def test_add_lag_features_correct_shift(self):
        """Lag-7 should equal the value from 7 rows earlier."""
        ...

    def test_add_rolling_features_window_size(self):
        """Rolling mean over window=7 should produce NaN for first 6 rows."""
        ...


class TestPromoPriceFeatures:
    """Tests for src.features.promo_price_features."""

    def test_add_promo_features_flag(self):
        """Promo flag should be binary."""
        ...

    def test_add_price_features_discount_depth(self):
        """Discount depth should be between 0 and 1."""
        ...
