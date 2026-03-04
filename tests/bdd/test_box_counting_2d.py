import numpy as np
from pytest_bdd import scenarios, given, when, then, parsers

from app.fractal_dimension.box_counting_2d import box_counting_2d
from app.helpers.sierpinski_carpet import generate_sierpinski_carpet
from app.helpers.sierpinski_triangle import generate_sierpinski_triangle

scenarios("../../features/box_counting_2d.feature")

FRACTAL_GENERATORS = {
    "sierpinski_carpet": generate_sierpinski_carpet,
    "sierpinski_triangle": generate_sierpinski_triangle,
}

REQUIRED_KEYS = {"dimension", "epsilons",
                 "counts", "slope", "intercept", "r_squared"}


# Given

@given("модуль box_counting_2d импортирован")
def step_module_imported():
    assert box_counting_2d is not None


@given(
    parsers.parse('2D фрактал "{fractal}" глубины {depth:d}'),
    target_fixture="context",
)
def step_fractal_2d(fractal, depth):
    gen = FRACTAL_GENERATORS[fractal]
    return {"array": gen(depth=depth)}


@given("полностью заполненный массив 64x64", target_fixture="context")
def step_full_64():
    return {"array": np.ones((64, 64), dtype=bool)}


@given("массив 64x64 с одним пикселем в центре", target_fixture="context")
def step_single_pixel():
    arr = np.zeros((64, 64), dtype=bool)
    arr[32, 32] = True
    return {"array": arr}


# When

@when("я вычисляю размерность методом box-counting 2D")
def step_compute_bc2d(context):
    context["result"] = box_counting_2d(context["array"])


@when("я вычисляю размерность box-counting 2D с use_divisors=False")
def step_compute_bc2d_no_div(context):
    context["result"] = box_counting_2d(context["array"], use_divisors=False)


@when(parsers.parse("я вычисляю размерность box-counting 2D с min_box={min_box:d}"))
def step_compute_bc2d_min_box(context, min_box):
    context["result"] = box_counting_2d(context["array"], min_box=min_box)


# Then (module-specific)

@then(parsers.parse("размерность должна быть меньше {bound:f}"))
def step_dim_less(context, bound):
    assert abs(context["result"]["dimension"]) < bound


@then(parsers.parse("все epsilons должны быть не менее {min_val:d}"))
def step_epsilons_min(context, min_val):
    assert context["result"]["epsilons"].min() >= min_val


@then(
    "результат должен содержать ключи: "
    "dimension, epsilons, counts, slope, intercept, r_squared"
)
def step_check_keys_2d(context):
    assert set(context["result"].keys()) == REQUIRED_KEYS


@then("счетчики должны монотонно не возрастать")
def step_counts_monotonic(context):
    counts = context["result"]["counts"]
    for i in range(len(counts) - 1):
        assert counts[i] >= counts[i + 1]
