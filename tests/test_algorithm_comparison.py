import numpy as np
import pytest

from app.fractal_dimension.box_counting_2d import box_counting_2d
from app.fractal_dimension.delaunay_dimension import delaunay_dimension
from app.helpers.sierpinski_carpet import generate_sierpinski_carpet
from app.helpers.koch_curve import generate_koch_curve

ANALYTICAL = {
    "sierpinski":   np.log(8) / np.log(3),
    "koch":     np.log(4) / np.log(3),
}

ABS_TOL_EACH = 0.15
ABS_TOL_DIFF = 0.20


# TC-CMP-01: Оба метода дают значение в одном диапазоне на ковре Серпинского.
# Проверяется, что |D_bc - D_td| < ABS_TOL_DIFF.

def test_carpet_methods_agree():
    """Box-counting и Делоне дают близкие оценки на ковре Серпинского."""
    arr = generate_sierpinski_carpet(depth=5)
    res_bc = box_counting_2d(arr)
    res_td = delaunay_dimension(arr)

    assert abs(res_bc["dimension"] - res_td["dimension"]) < ABS_TOL_DIFF, (
        f"Расхождение методов: BC={res_bc['dimension']:.4f}, "
        f"TD={res_td['dimension']:.4f}"
    )


# TC-CMP-02: Оба метода независимо приближают аналитическое значение
# на кривой Коха (сравнение с эталоном, а не друг с другом).

@pytest.mark.parametrize("method,key", [
    ("bc", "dimension"),
    ("td", "dimension"),
], ids=["box_counting", "delaunay"])
def test_triangle_each_near_analytical(method, key):
    """Каждый метод должен приближать аналитическую размерность треугольника."""
    arr = generate_koch_curve(depth=7)
    expected = ANALYTICAL["koch"]

    if method == "bc":
        res = box_counting_2d(arr)
    else:
        res = delaunay_dimension(arr)

    assert abs(res[key] - expected) < ABS_TOL_EACH, (
        f"Метод {method}: D={res[key]:.4f}, аналитика={expected:.4f}"
    )


# TC-CMP-03: Оба метода возвращают R^2 > 0 на кривой Коха,
# подтверждая, что линейная регрессия в обоих случаях состоялась.

def test_koch_both_r_squared_positive():
    """R^2 > 0 у обоих методов на кривой Коха - регрессия состоятельна."""
    arr = generate_koch_curve(depth=5, resolution=512)
    res_bc = box_counting_2d(arr)
    res_td = delaunay_dimension(arr)

    assert res_bc["r_squared"] > 0, \
        f"Box-counting R^2={res_bc['r_squared']:.4f} не положителен"
    assert res_td["r_squared"] > 0, \
        f"Делоне R^2={res_td['r_squared']:.4f} не положителен"
