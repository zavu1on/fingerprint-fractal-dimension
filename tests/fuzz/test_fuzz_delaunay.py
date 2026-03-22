
import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from app.fractal_dimension.delaunay_dimension import delaunay_dimension


REQUIRED_KEYS = {"dimension", "deltas", "num_triangles", "slope",
                 "intercept", "r_squared", "triangulations",
                 "subsampled_pts", "kept_masks"}


sparse_bool_array = st.tuples(
    st.integers(min_value=8, max_value=64),
    st.integers(min_value=8, max_value=64),
    st.floats(min_value=0.001, max_value=0.05),
).map(
    lambda hwp: np.random.default_rng(42).random((hwp[0], hwp[1])) < hwp[2]
)

small_point_set = st.integers(min_value=1, max_value=5).map(
    lambda n: np.random.default_rng(42).random((n, 2)).astype(np.float64) * 100
)

medium_point_set = st.integers(min_value=3, max_value=200).map(
    lambda n: np.random.default_rng(42).random((n, 2)).astype(np.float64) * 500
)


@given(arr=sparse_bool_array)
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_delaunay_sparse_array(arr):
    n_true = arr.sum()
    assume(n_true >= 1)

    result = delaunay_dimension(arr)

    assert REQUIRED_KEYS <= set(result.keys()), \
        f"Отсутствуют ключи: {REQUIRED_KEYS - set(result.keys())}"
    assert np.isfinite(result["dimension"]), \
        f"dimension = {result['dimension']}"


@given(pts=small_point_set)
@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_delaunay_small_point_set(pts):
    result = delaunay_dimension(pts)

    assert REQUIRED_KEYS <= set(result.keys())
    assert np.isfinite(result["dimension"]), \
        f"dimension = {result['dimension']} при {len(pts)} точках"


@given(
    pts=medium_point_set,
    max_frac=st.floats(min_value=0.5, max_value=5.0),
    edge_factor=st.floats(min_value=0.1, max_value=20.0),
)
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_delaunay_extreme_params(pts, max_frac, edge_factor):
    result = delaunay_dimension(pts, max_frac=max_frac,
                                edge_factor=edge_factor)

    assert REQUIRED_KEYS <= set(result.keys())
    assert np.isfinite(result["dimension"]), \
        f"dimension={result['dimension']} при max_frac={max_frac}, " \
        f"edge_factor={edge_factor}"


@given(data=st.data())
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_delaunay_consistency(data):
    size = data.draw(st.integers(min_value=16, max_value=64))
    density = data.draw(st.floats(min_value=0.05, max_value=0.8))
    arr = np.random.default_rng(42).random((size, size)) < density

    assume(arr.sum() >= 3)

    result = delaunay_dimension(arr)

    n = len(result["deltas"])
    assert len(result["num_triangles"]) == n, \
        f"Длина num_triangles ({len(result['num_triangles'])}) != deltas ({n})"
    assert len(result["triangulations"]) == n
    assert len(result["subsampled_pts"]) == n
    assert len(result["kept_masks"]) == n


def test_two_points_triggers_bug():
    pts = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float64)

    try:
        result = delaunay_dimension(pts)
        assert np.isfinite(result["dimension"]), \
            f"dimension = {result['dimension']}"
    except (ValueError, Exception) as e:
        pytest.fail(
            f"BUG FOUND: необработанное исключение {type(e).__name__}: {e}. "
            f"Причина: отсутствует проверка len(sub_pts) < 3 перед Delaunay()."
        )
