"""Tests for signal processing geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.signal_processing import compute_center_distance


class TestComputeCenterDistance:
    """Tests for center/coverage scoring."""

    def test_scores_geometric_center_as_perfectly_centered(self) -> None:
        centroid = np.array([15.5, 15.5])

        assert compute_center_distance(centroid, grid_size=32) == pytest.approx(1.0)

    def test_scores_symmetric_locations_equally_on_even_grid(self) -> None:
        top_left = np.array([15.0, 15.0])
        bottom_right = np.array([16.0, 16.0])

        assert compute_center_distance(top_left, grid_size=32) == pytest.approx(
            compute_center_distance(bottom_right, grid_size=32)
        )
