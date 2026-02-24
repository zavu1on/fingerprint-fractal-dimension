"""
Equivalence Partitioning tests for box_counting_2d.

Each test targets a specific equivalence class of inputs:
array shape/fill, min_box values, use_divisors mode, output structure.

Includes: parametrized tests, pytest.approx matcher, numpy.testing matchers.
"""
import numpy as np
import pytest

from app.fractal_dimension.box_counting_2d import box_counting_2d
from app.helpers.sierpinski_carpet import generate_sierpinski_carpet
from app.helpers.sierpinski_triangle import generate_sierpinski_triangle

REQUIRED_KEYS = {"dimension", "epsilons",
                 "counts", "slope", "intercept", "r_squared"}


# TC-EP-01 + TC-EP-02: Parametrized fractal dimension check
# Tests two equivalence classes at once via parametrize

@pytest.mark.parametrize("gen_func, depth, expected_dim", [
    (generate_sierpinski_carpet, 5, np.log(8) / np.log(3)),   # EP-A1: side=3^n
    (generate_sierpinski_triangle, 7, np.log(3) / np.log(2)),  # EP-A2: side=2^n
], ids=["carpet_243", "triangle_128"])
def test_known_fractal_dimension(gen_func, depth, expected_dim):
    """Known fractals must match analytical dimension within tolerance."""
    arr = gen_func(depth=depth)
    res = box_counting_2d(arr)

    # matcher: pytest.approx for approximate float comparison
    assert res["dimension"] == pytest.approx(expected_dim, abs=0.1)
    assert res["r_squared"] > 0.95


# TC-EP-03: Rectangular (non-square) array

def test_rectangular_array(rectangular_array):
    """Non-square array must be processed without errors."""
    res = box_counting_2d(rectangular_array)
    assert isinstance(res, dict)
    assert 0.5 < res["dimension"] < 1.5


# TC-EP-04: Fully filled -> D = 2.0

def test_full_array_dimension(full_array_64):
    """Solid plane must have D close to 2."""
    res = box_counting_2d(full_array_64)
    # matcher: pytest.approx
    assert res["dimension"] == pytest.approx(2.0, abs=0.3)


# TC-EP-05: Single pixel -> D = 0

def test_single_pixel_dimension(single_pixel_64):
    """A point-like object must have D close to 0."""
    res = box_counting_2d(single_pixel_64)
    assert abs(res["dimension"]) < 0.5


# TC-EP-06: Minimal 2x2 array

def test_minimal_array():
    """2x2 input must not crash; returns dimension=0 (only 1 scale)."""
    arr = np.array([[True, False], [False, True]])
    res = box_counting_2d(arr)
    assert res["dimension"] == 0.0  # too few scales for regression


# TC-EP-07: Float array identical to bool

def test_float_array_equals_bool(sierpinski_carpet_243):
    """Float 0.0/1.0 array must give the same result as bool."""
    float_arr = sierpinski_carpet_243.astype(np.float64)
    res_bool = box_counting_2d(sierpinski_carpet_243)
    res_float = box_counting_2d(float_arr)

    # matcher: numpy.testing.assert_allclose for array-level comparison
    np.testing.assert_allclose(
        res_float["dimension"], res_bool["dimension"], atol=1e-10
    )


# TC-EP-08: use_divisors=False still gives correct D

def test_use_divisors_false(sierpinski_carpet_243):
    """Geometric-only mode should approximate the same dimension."""
    res = box_counting_2d(sierpinski_carpet_243, use_divisors=False)
    assert res["dimension"] == pytest.approx(1.8928, abs=0.15)


# TC-EP-09: min_box is respected

def test_min_box_respected(sierpinski_carpet_243):
    """Smallest epsilon must be >= min_box."""
    res = box_counting_2d(sierpinski_carpet_243, min_box=4)
    assert res["epsilons"].min() >= 4


# TC-EP-10: Output contains all required keys

def test_output_keys(sierpinski_carpet_243):
    """Result dict must have all six expected keys."""
    res = box_counting_2d(sierpinski_carpet_243)
    assert set(res.keys()) == REQUIRED_KEYS


# TC-EP-11: Counts are monotonically non-increasing

def test_counts_monotonicity(sierpinski_carpet_243):
    """Larger boxes -> fewer (or equal) occupied boxes."""
    res = box_counting_2d(sierpinski_carpet_243)
    counts = res["counts"]
    for i in range(len(counts) - 1):
        assert counts[i] >= counts[i + 1]
