import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from app.fractal_dimension.box_counting_2d import box_counting_2d


random_bool_array_2d = st.tuples(
    st.integers(min_value=4, max_value=128),
    st.integers(min_value=4, max_value=128),
).flatmap(
    lambda hw: arrays(dtype=np.bool_, shape=hw,
                      elements=st.booleans())
)

sparse_bool_array_2d = st.tuples(
    st.integers(min_value=4, max_value=64),
    st.integers(min_value=4, max_value=64),
    st.floats(min_value=0.0, max_value=1.0),
).map(
    lambda hwp: np.random.default_rng(42).random((hwp[0], hwp[1])) < hwp[2]
)

REQUIRED_KEYS = {"dimension", "epsilons",
                 "counts", "slope", "intercept", "r_squared"}


@given(arr=random_bool_array_2d)
@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_bc2d_random_array(arr):
    result = box_counting_2d(arr)

    assert REQUIRED_KEYS <= set(result.keys()), \
        f"Отсутствуют ключи: {REQUIRED_KEYS - set(result.keys())}"

    assert np.isfinite(result["dimension"]), \
        f"dimension = {result['dimension']} (ожидается конечное число)"

    assert (result["counts"] >= 0).all(), "Обнаружены отрицательные counts"

    assert (result["epsilons"] > 0).all(
    ), "Обнаружены неположительные epsilons"


@given(arr=sparse_bool_array_2d)
@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_bc2d_sparse_array(arr):
    result = box_counting_2d(arr)

    assert REQUIRED_KEYS <= set(result.keys())
    assert np.isfinite(result["dimension"]), \
        f"dimension = {result['dimension']} при массиве плотности " \
        f"{arr.sum() / arr.size:.2%}"
    assert np.isfinite(result["r_squared"]), \
        f"r_squared = {result['r_squared']}"


@given(
    size=st.integers(min_value=4, max_value=64),
    min_box=st.integers(min_value=0, max_value=10),
    use_divisors=st.booleans(),
)
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_bc2d_parameters(size, min_box, use_divisors):
    """Фаззинг параметров min_box и use_divisors на пустом массиве."""
    arr = np.zeros((size, size), dtype=bool)

    if min_box == 0:
        min_box = 1

    result = box_counting_2d(arr, min_box=min_box, use_divisors=use_divisors)

    assert np.isfinite(result["dimension"]), \
        f"dimension = {result['dimension']} при size={size}, " \
        f"min_box={min_box}, use_divisors={use_divisors}"


@given(data=st.data())
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_bc2d_single_pixel(data):
    """Фаззинг: один пиксель в случайной позиции."""
    size = data.draw(st.integers(min_value=8, max_value=64))
    r = data.draw(st.integers(min_value=0, max_value=size - 1))
    c = data.draw(st.integers(min_value=0, max_value=size - 1))

    arr = np.zeros((size, size), dtype=bool)
    arr[r, c] = True

    result = box_counting_2d(arr)

    assert np.isfinite(result["dimension"]), \
        f"dimension = {result['dimension']} для одного пикселя"


def test_empty_array_triggers_bug():
    """
    Прямой тест: полностью пустой массив.
    Ожидание: dimension должна быть конечным числом.
    Реальность (с ошибкой): log(0)=-inf -> dimension=NaN.
    """
    arr = np.zeros((16, 16), dtype=bool)
    result = box_counting_2d(arr)

    assert np.isfinite(result["dimension"]), \
        f"BUG FOUND: dimension={result['dimension']} на пустом массиве. " \
        f"Причина: log(0) при отсутствии фильтрации нулевых counts."
