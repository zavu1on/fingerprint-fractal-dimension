import numpy as np
from pytest_bdd import scenarios, given, when, then, parsers

from app.fractal_dimension.box_counting_3d import box_counting_3d
from app.helpers.menger_sponge import generate_menger_sponge

scenarios("../../features/box_counting_3d.feature")

REQUIRED_KEYS = {"dimension", "epsilons",
                 "counts", "slope", "intercept", "r_squared"}


# Given

@given("модуль box_counting_3d импортирован")
def step_module_imported():
    assert box_counting_3d is not None


@given(parsers.parse("губка Менгера глубины {depth:d}"), target_fixture="context")
def step_menger_sponge(depth):
    return {"array": generate_menger_sponge(depth=depth)}


@given("полностью заполненный объем 27x27x27", target_fixture="context")
def step_full_3d():
    return {"array": np.ones((27, 27, 27), dtype=bool)}


@given("пустой объем 27x27x27", target_fixture="context")
def step_empty_3d():
    return {"array": np.zeros((27, 27, 27), dtype=bool)}


@given("разреженный объем 7x7x7 с 3 занятыми вокселями", target_fixture="context")
def step_sparse_7():
    arr = np.zeros((7, 7, 7), dtype=bool)
    arr[0, 0, 0] = True
    arr[3, 3, 3] = True
    arr[6, 6, 6] = True
    return {"array": arr}


# When

@when("я вычисляю размерность методом box-counting 3D")
def step_compute_bc3d(context):
    context["result"] = box_counting_3d(context["array"])


@when(parsers.parse("я вычисляю размерность box-counting 3D с max_box={max_box:d}"))
def step_compute_bc3d_maxbox(context, max_box):
    context["result"] = box_counting_3d(context["array"], max_box=max_box)


@when("я вычисляю размерность box-counting 3D с use_divisors=False")
def step_compute_bc3d_no_div(context):
    context["result"] = box_counting_3d(context["array"], use_divisors=False)


# Then (module-specific)

@then(parsers.parse("количество масштабов должно быть не менее {n:d}"))
def step_n_scales_ge(context, n):
    assert len(context["result"]["epsilons"]) >= n


@then("все счетчики должны быть нулевыми")
def step_counts_zero(context):
    np.testing.assert_array_equal(
        context["result"]["counts"],
        np.zeros_like(context["result"]["counts"]),
    )


@then("результат должен содержать все обязательные ключи")
def step_check_keys_3d(context):
    assert set(context["result"].keys()) == REQUIRED_KEYS


@then("размерность должна быть больше 0")
def step_dim_positive(context):
    assert context["result"]["dimension"] > 0


@then(parsers.parse("максимальный epsilon не должен превышать {limit:d}"))
def step_max_eps(context, limit):
    assert context["result"]["epsilons"].max() <= limit
