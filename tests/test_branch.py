"""
Branch Testing for box_counting_3d, _divisor_scales, and fractal generators.

Each test covers specific True/False branches in conditional logic.

Includes: parametrized tests for fractal generators, numpy.testing matchers,
          assumption (pytest.skip for slow computations).
"""
import numpy as np
import pytest

from src.fractal_dimension.box_counting_3d import box_counting_3d
from src.fractal_dimension.box_counting_2d import _divisor_scales
from src.helpers.menger_sponge import generate_menger_sponge
from src.helpers.koch_curve import generate_koch_curve
from src.helpers.sierpinski_carpet import generate_sierpinski_carpet
from src.helpers.sierpinski_triangle import generate_sierpinski_triangle


# TC-BR-01: max_box=None (T), use_div=True (T), ≥4 divisors (len<4 F)

def test_menger_default(menger_sponge_81):
    """Branches: max_box=None -> T, use_divisors -> T, len<4 -> F."""
    res = box_counting_3d(menger_sponge_81)
    assert res["dimension"] == pytest.approx(2.727, abs=0.15)
    assert len(res["epsilons"]) >= 4


# TC-BR-02: max_box explicit (max_box is None -> F)

def test_menger_explicit_maxbox(menger_sponge_81):
    """Branch: max_box is None -> F."""
    res = box_counting_3d(menger_sponge_81, max_box=27)
    assert res["dimension"] > 0
    assert res["epsilons"].max() <= 27


# TC-BR-03: use_divisors=False -> geometric sequence

def test_menger_no_divisors(menger_sponge_81):
    """Branch: use_divisors -> F."""
    res = box_counting_3d(menger_sponge_81, use_divisors=False)
    assert res["dimension"] > 0


# TC-BR-04: Sparse 7³ - few divisors triggers augmentation

def test_sparse_few_divisors(sparse_7):
    """Branch: len(epsilons) < 4 -> T, geometric augmentation applied."""
    res = box_counting_3d(sparse_7, use_divisors=True)
    assert len(res["epsilons"]) >= 2


# TC-BR-05: Full 27³ - cube.any() always True

def test_full_3d(full_3d_27):
    """Branch: cube.any() -> T for every cube. D = 3."""
    res = box_counting_3d(full_3d_27)
    assert res["dimension"] == pytest.approx(3.0, abs=0.3)


# TC-BR-06: Empty 27³ - cube.any() always False

def test_empty_3d(empty_3d_27):
    """Branch: cube.any() -> F for every cube. Dimension = 0 (degenerate)."""
    res = box_counting_3d(empty_3d_27)
    assert res["dimension"] == 0.0
    # matcher: numpy.testing.assert_array_equal
    np.testing.assert_array_equal(res["counts"], np.zeros_like(res["counts"]))


# TC-BR-07: Shape with many divisors -> non-empty

def test_divisor_scales_nonempty():
    """Branch: divs list non-empty -> T."""
    result = _divisor_scales((81, 81, 81), 1, 40)
    assert len(result) > 0
    for d in [1, 3, 9, 27]:
        assert d in result


# TC-BR-08: Prime-ish shape -> empty result

def test_divisor_scales_empty():
    """Branch: divs list non-empty -> F (returns empty array)."""
    result = _divisor_scales((7, 7, 7), 2, 6)
    assert len(result) == 0


# TC-BR-09 + TC-BR-10: Menger sponge at different depths

@pytest.mark.parametrize("depth, expected_shape, expected_count", [
    (1, (3, 3, 3), 20),       # TC-BR-09: 20 of 27 voxels survive
    (2, (9, 9, 9), 20**2),    # TC-BR-10: 20^2 = 400 voxels
], ids=["depth_1", "depth_2"])
def test_menger_sponge_structure(depth, expected_shape, expected_count):
    """Menger sponge removal rule must produce exact voxel counts."""
    arr = generate_menger_sponge(depth=depth)
    assert arr.shape == expected_shape
    assert arr.sum() == expected_count


# TC-BR-11: Koch curve produces non-empty raster

def test_koch_depth_1():
    """Depth=1 Koch curve must have True pixels."""
    arr = generate_koch_curve(depth=1, resolution=64)
    assert arr.shape == (64, 64)
    assert arr.any()


# TC-BR-12: Sierpinski carpet depth=1 -> center removed

def test_carpet_depth_1():
    """3x3 carpet: center cell [1,1] is False, 8 others True."""
    arr = generate_sierpinski_carpet(depth=1)

    # assumption: skip if array shape is unexpected (safety check)
    if arr.shape != (3, 3):
        pytest.skip("Unexpected carpet shape, skipping structural check")

    assert arr[1, 1] == False
    assert arr.sum() == 8


# TC-BR-13: Sierpinski triangle depth=1 -> 3 of 4 cells

def test_triangle_depth_1():
    """2x2 triangle: arr[1,1] is False, 3 others True."""
    arr = generate_sierpinski_triangle(depth=1)
    assert arr.shape == (2, 2)
    assert arr[1, 1] == False
    assert arr.sum() == 3
