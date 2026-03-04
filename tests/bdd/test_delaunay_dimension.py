import pytest
import numpy as np
from pytest_bdd import scenarios, given, when, then, parsers

from app.fractal_dimension.delaunay_dimension import delaunay_dimension
from app.helpers.sierpinski_carpet import generate_sierpinski_carpet

scenarios("../../features/delaunay_dimension.feature")


# Given

@given("модуль delaunay_dimension импортирован")
def step_module_imported():
    assert delaunay_dimension is not None


@given("ковер Серпинского глубины 5 как 2D булева маска", target_fixture="context")
def step_carpet_5():
    return {"array": generate_sierpinski_carpet(depth=5)}


@given("набор из 3 неколлинеарных точек", target_fixture="context")
def step_three_points():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    return {"array": pts}


@given("набор из 2 точек", target_fixture="context")
def step_two_points():
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    return {"array": pts}


@given("набор из 10 коллинеарных точек", target_fixture="context")
def step_collinear_10():
    pts = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])
    return {"array": pts}


# When

@when("я оцениваю фрактальную размерность методом Делоне")
def step_compute_delaunay(context):
    context["result"] = delaunay_dimension(context["array"])


@when(parsers.parse("я оцениваю размерность Делоне с edge_factor={edge_factor:f}"))
def step_compute_edge_factor(context, edge_factor):
    context["result"] = delaunay_dimension(
        context["array"], edge_factor=edge_factor)


@when("я оцениваю размерность Делоне с параметрами по умолчанию")
def step_compute_default(context):
    context["result_default"] = delaunay_dimension(context["array"])


@when(parsers.parse("я также оцениваю размерность Делоне с edge_factor={edge_factor:f}"))
def step_compute_extra(context, edge_factor):
    context["result_extra"] = delaunay_dimension(
        context["array"],
        edge_factor=edge_factor
    )


@when(parsers.parse("я оцениваю размерность Делоне с num_scales={num_scales:d}"))
def step_compute_ns(context, num_scales):
    context["result"] = delaunay_dimension(
        context["array"], num_scales=num_scales
    )


@when(parsers.parse("я также оцениваю размерность Делоне с num_scales={num_scales:d}"))
def step_compute_ns(context, num_scales):
    context["result_extra"] = delaunay_dimension(
        context["array"], num_scales=num_scales
    )


# Then (module-specific)

@then("длины deltas и num_triangles должны совпадать")
def step_deltas_eq_nt(context):
    res = context["result"]
    assert len(res["deltas"]) == len(res["num_triangles"])


@then("длина triangulations должна совпадать с deltas")
def step_tri_len(context):
    res = context["result"]
    assert len(res["triangulations"]) == len(res["deltas"])


@then("длина subsampled_pts должна совпадать с deltas")
def step_sub_len(context):
    res = context["result"]
    assert len(res["subsampled_pts"]) == len(res["deltas"])


@then("массив num_triangles должен быть пустым")
def step_nt_empty(context):
    assert len(context["result"]["num_triangles"]) == 0


@then(parsers.parse("две оценки размерности должны различаться более чем на {epsilon:f}"))
def step_dims_differ(context, epsilon):
    if "result_extra" not in context:
        pytest.skip("Extra result not computed. Nothing to compare")

    d1 = context["result_default"]["dimension"]
    d2 = context["result_extra"]["dimension"]
    assert abs(d1 - d2) > epsilon


@then('результат должен быть словарем с ключом "dimension"')
def step_dict_with_dim(context):
    assert isinstance(context["result"], dict)
    assert "dimension" in context["result"]


@then(parsers.parse('первое значение должно быть ближе к {expected:f} чем второе'))
def step_close_two_value(context, expected):
    if "result_extra" not in context:
        pytest.skip("Extra result not computed. Nothing to compare")

    d1 = context["result"]["dimension"]
    d2 = context["result_extra"]["dimension"]

    assert np.abs(expected - d1) < np.abs(expected - d2)
