"""Tests for signal processing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.signal_processing import compute_center_distance


def test_even_grid_middle_cells_score_symmetrically() -> None:
    """A 32x32 grid is centered at 15.5, with equal adjacent middle cells."""
    center_score = compute_center_distance(
        np.array([15.5, 15.5], dtype=np.float32), grid_size=32
    )
    middle_scores = [
        compute_center_distance(np.array([row, col], dtype=np.float32), grid_size=32)
        for row, col in ((15, 15), (15, 16), (16, 15), (16, 16))
    ]

    assert center_score == pytest.approx(1)
    assert middle_scores == pytest.approx([middle_scores[0]] * 4)


def test_opposite_corners_score_symmetrically() -> None:
    """Opposite corners should be equally far from the geometric center."""
    top_left = compute_center_distance(np.array([0, 0], dtype=np.float32), grid_size=32)
    bottom_right = compute_center_distance(
        np.array([31, 31], dtype=np.float32), grid_size=32
    )

    assert top_left == pytest.approx(bottom_right)


def test_single_cell_grid_scores_as_centered() -> None:
    """The only cell in a 1x1 grid is centered by definition."""
    assert compute_center_distance(np.array([0, 0], dtype=np.float32), grid_size=1) == 1
