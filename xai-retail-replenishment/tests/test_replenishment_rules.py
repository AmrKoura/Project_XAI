"""
Tests for the replenishment rules module.

Covers reorder quantity calculation, reorder-point logic,
and safety stock computation.
"""

import pytest
import numpy as np


class TestReorderQuantity:
    """Tests for replenishment_rules.compute_reorder_quantity."""

    def test_reorder_quantity_positive(self):
        """Reorder qty should be positive when stock < demand + safety."""
        ...

    def test_reorder_quantity_zero_when_sufficient_stock(self):
        """Reorder qty should be 0 when stock covers demand + safety."""
        ...

    def test_reorder_quantity_never_negative(self):
        """Reorder qty should be floored at 0."""
        ...


class TestShouldReorder:
    """Tests for replenishment_rules.should_reorder."""

    def test_triggers_when_below_rop(self):
        """Should return True when stock < reorder point."""
        ...

    def test_no_trigger_when_above_rop(self):
        """Should return False when stock >= reorder point."""
        ...


class TestSafetyStock:
    """Tests for safety_stock module."""

    def test_quantile_method_nonnegative(self):
        """Safety stock from quantile method should be >= 0."""
        ...

    def test_mad_method_increases_with_service_level(self):
        """Higher service level should yield higher safety stock."""
        ...
