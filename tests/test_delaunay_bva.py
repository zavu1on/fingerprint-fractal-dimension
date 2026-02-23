"""
Boundary Value Analysis tests for delaunay_dimension.

Each test targets a boundary of a parameter range:
num_scales extremes, min_frac limits, edge_factor edges, minimal point sets.

Includes: pytest.importorskip (assumption), pytest.skip (assumption).
"""
from app.fractal_dimension.delaunay_dimension import delaunay_dimension
import numpy as np
import pytest

# Assumption: scipy must be available for Delaunay
scipy = pytest.importorskip(
    "scipy", reason="scipy required for Delaunay triangulation")


# TC-BV-01: Default params on known fractal

def test_default_params(sierpinski_carpet_243):
    """Baseline: D = 1.893, R^2 > 0.9."""
    res = delaunay_dimension(sierpinski_carpet_243)
    assert res["dimension"] == pytest.approx(1.8928, abs=0.15)
    assert res["r_squared"] > 0.9


# TC-BV-02: num_scales = 2 (minimum for regression)

def test_num_scales_min(sierpinski_carpet_243):
    """With only 2 scales, regression should still be possible."""
    res = delaunay_dimension(sierpinski_carpet_243, num_scales=2)
    assert len(res["deltas"]) >= 2


# TC-BV-03: num_scales = 50 (large value)

def test_num_scales_large(sierpinski_carpet_243):
    """Many scales - D should stay reasonable."""
    res = delaunay_dimension(sierpinski_carpet_243, num_scales=50)
    assert res["dimension"] == pytest.approx(1.8928, abs=0.15)


# TC-BV-04: min_frac = 0.001 (very small)

def test_min_frac_small(sierpinski_carpet_243):
    """Very fine lower bound -> more scales, no crash."""
    res = delaunay_dimension(sierpinski_carpet_243, min_frac=0.001)
    assert len(res["deltas"]) >= 2


# TC-BV-05: min_frac = 0.19 (almost equal to max_frac)

def test_min_frac_near_max(sierpinski_carpet_243):
    """Narrow range -> fewer scales but still works."""
    res = delaunay_dimension(sierpinski_carpet_243, min_frac=0.19)
    assert len(res["deltas"]) >= 1


# TC-BV-06: edge_factor = 0.0 (all triangles filtered)

def test_edge_factor_zero(sierpinski_carpet_243):
    """Zero factor filters every triangle; expect degenerate result."""
    res = delaunay_dimension(sierpinski_carpet_243, edge_factor=0.0)
    assert len(res["num_triangles"]) == 0
    assert res["dimension"] == 0.0


# TC-BV-07: edge_factor = 100 vs default

def test_edge_factor_large_differs(sierpinski_carpet_243):
    """Disabling the filter must change the dimension estimate."""
    res_default = delaunay_dimension(sierpinski_carpet_243, edge_factor=3.0)
    res_large = delaunay_dimension(sierpinski_carpet_243, edge_factor=100.0)
    # the values must differ noticeably
    assert abs(res_large["dimension"] - res_default["dimension"]) > 0.01


# TC-BV-08: Exactly 3 points (minimum for Delaunay)

def test_three_points():
    """Triangulation must succeed with exactly 3 float points."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    res = delaunay_dimension(pts, num_scales=3, min_frac=0.01, max_frac=0.9)
    assert isinstance(res, dict)
    assert "dimension" in res


# TC-BV-09: 2 points (below Delaunay minimum)

def test_two_points():
    """2 points cannot form a triangle; expect graceful degenerate result."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    res = delaunay_dimension(pts, num_scales=3, min_frac=0.01, max_frac=0.9)
    assert len(res["num_triangles"]) == 0
    assert res["dimension"] == 0.0


# TC-BV-10: Collinear points

def test_collinear_points():
    """10 collinear points -> QhullError handled gracefully."""
    pts = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])
    res = delaunay_dimension(pts, num_scales=4, min_frac=0.01, max_frac=0.5)
    assert isinstance(res, dict)
    assert res["dimension"] == 0.0


# TC-BV-11: Single nonzero row in 2D mask

def test_single_row_mask():
    """64x64 mask with only one row True - should not crash."""
    arr = np.zeros((64, 64), dtype=bool)
    arr[32, :] = True
    res = delaunay_dimension(arr)
    assert isinstance(res, dict)


# TC-BV-12: Consistency of array lengths

def test_result_lengths_consistent(sierpinski_carpet_243):
    """All per-scale arrays must have the same length."""
    res = delaunay_dimension(sierpinski_carpet_243)
    n = len(res["deltas"])
    assert len(res["num_triangles"]) == n
    assert len(res["triangulations"]) == n
    assert len(res["subsampled_pts"]) == n
    assert len(res["kept_masks"]) == n
