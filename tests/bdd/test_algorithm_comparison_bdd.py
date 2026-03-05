"""
Step definitions для нагрузочного BDD-тестирования алгоритмов
вычисления фрактальной размерности с профилированием cProfile

Профилирование включает:
  - замер времени каждого этапа (генерация, box-counting, Делоне)
  - сохранение cProfile-профилей (.prof бинарный, .txt текстовый)
  - сохранение сводки замеров (summary.json)
  - вывод таблицы замеров и топ-функций в консоль

Запуск всех нагрузочных тестов:
    pytest tests/bdd/test_algorithm_comparison_bdd.py -v -s

    Флаг -s обязателен для вывода профилей и замеров в консоль.

Запуск только средних наборов:
    pytest tests/bdd/test_algorithm_comparison_bdd.py -v -s -k "S2916"

Запуск только больших наборов:
    pytest tests/bdd/test_algorithm_comparison_bdd.py -v -s -k "S5832"

Запуск только сравнения с unit-тестом:
    pytest tests/bdd/test_algorithm_comparison_bdd.py -v -s -k "carpet"

Визуализация профилей после запуска:
    snakeviz results/perc_S2916_f10/box_counting_2d.prof
"""

import os
import sys
import time
import json
import cProfile
import pstats
import io

from pytest_bdd import scenarios, given, when, then, parsers

from app.fractal_dimension.box_counting_2d import box_counting_2d
from app.fractal_dimension.delaunay_dimension import delaunay_dimension
from app.helpers.sierpinski_carpet import generate_sierpinski_carpet

from .dataset_generator import load_or_generate

sys.path.insert(0, os.path.dirname(__file__))


scenarios("../../features/algorithm_comparison.feature")

RESULTS_DIR = "results"


# Profiling

def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    t_start = time.perf_counter()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    elapsed = time.perf_counter() - t_start
    return result, profiler, elapsed


def save_profile_report(profiler, filepath_base):
    profiler.dump_stats(filepath_base + ".prof")

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(40)
    report_text = stream.getvalue()

    with open(filepath_base + ".txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def print_timing_table(ctx):
    t = ctx["timings"]
    ds = ctx.get("dataset_name", "N/A")
    r_bc = ctx["result_bc"]
    r_td = ctx["result_td"]
    t_total = t["box_counting"] + t["delaunay"]
    diff = abs(r_bc["dimension"] - r_td["dimension"])

    print(f"\n{'=' * 70}")
    print(f"  Набор данных: {ds}")
    print(f"{'=' * 70}")
    print(f"  {'Этап':<35} {'Время, сек':>12}")
    print(f"  {'-' * 35} {'-' * 12}")
    print(f"  {'Генерация данных':<35} {t['generation']:>12.3f}")
    print(f"  {'Box-counting 2D':<35} {t['box_counting']:>12.3f}")
    print(f"  {'Триангуляция Делоне':<35} {t['delaunay']:>12.3f}")
    print(f"  {'ИТОГО (оба алгоритма)':<35} {t_total:>12.3f}")
    print(f"  {'-' * 35} {'-' * 12}")
    print(f"  {'Box-counting D':<35} {r_bc['dimension']:>12.4f}")
    print(f"  {'Box-counting R^2':<35} {r_bc['r_squared']:>12.4f}")
    print(f"  {'Box-counting масштабов':<35} {len(r_bc['epsilons']):>12d}")
    print(f"  {'Делоне D':<35} {r_td['dimension']:>12.4f}")
    print(f"  {'Делоне R^2':<35} {r_td['r_squared']:>12.4f}")
    print(f"  {'Делоне масштабов':<35} {len(r_td['deltas']):>12d}")
    print(f"  {'-' * 35} {'-' * 12}")
    print(f"  {'|D_bc - D_td|':<35} {diff:>12.4f}")
    print(f"{'=' * 70}\n")


# Given

@given("модули box_counting_2d и delaunay_dimension импортированы")
def step_modules_imported():
    assert box_counting_2d is not None
    assert delaunay_dimension is not None


@given("каталог results существует для сохранения профилей")
def step_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


@given(
    parsers.parse(
        'сгенерирован перколяционный набор "{dataset}" '
        'размером {resolution:d} с заполнением {fill_ratio:f}'
    ),
    target_fixture="context",
)
def step_generate_dataset(dataset, resolution, fill_ratio):
    ctx = {
        "dataset_name": dataset,
        "resolution": resolution,
        "fill_ratio": fill_ratio,
        "timings": {},
    }

    data_dir = "datasets"
    os.makedirs(data_dir, exist_ok=True)

    t0 = time.perf_counter()
    arr = load_or_generate(dataset, resolution, fill_ratio, data_dir)
    ctx["timings"]["generation"] = time.perf_counter() - t0
    ctx["array"] = arr
    ctx["n_points"] = int(arr.sum())

    return ctx


@given("время генерации набора зафиксировано")
def step_gen_time_logged(context):
    t = context["timings"]["generation"]
    npy_path = os.path.join("datasets", f"{context['dataset_name']}.npy")
    mode = "загружен" if os.path.exists(npy_path) else "сгенерирован"
    print(f"\n  [{mode.capitalize()}] {context['dataset_name']}: "
          f"{context['resolution']}x{context['resolution']}, "
          f"fill={context['fill_ratio']}, "
          f"{context['n_points']} точек, "
          f"{t:.3f} сек")


@given(
    "ковер Серпинского глубины 5 для воспроизведения unit-теста",
    target_fixture="context",
)
def step_carpet_for_unit_test():
    ctx = {
        "dataset_name": "sierpinski_carpet_d5",
        "timings": {},
    }
    t0 = time.perf_counter()
    ctx["array"] = generate_sierpinski_carpet(depth=5)
    ctx["timings"]["generation"] = time.perf_counter() - t0
    ctx["n_points"] = int(ctx["array"].sum())
    return ctx


# When

@when("я вычисляю размерность методом box-counting 2D с профилированием cProfile")
def step_bc2d_profiled(context):
    result, profiler, elapsed = profile_function(
        box_counting_2d, context["array"]
    )
    context["result_bc"] = result
    context["profile_bc"] = profiler
    context["timings"]["box_counting"] = elapsed
    print(f"  [Box-counting] D={result['dimension']:.4f}, "
          f"R^2={result['r_squared']:.4f}, "
          f"scales={len(result['epsilons'])}, "
          f"t={elapsed:.3f} сек")


@when("я вычисляю размерность методом триангуляции Делоне с профилированием cProfile")
def step_delaunay_profiled(context):
    result, profiler, elapsed = profile_function(
        delaunay_dimension, context["array"]
    )
    context["result_td"] = result
    context["profile_td"] = profiler
    context["timings"]["delaunay"] = elapsed
    print(f"  [Делоне]       D={result['dimension']:.4f}, "
          f"R^2={result['r_squared']:.4f}, "
          f"scales={len(result['deltas'])}, "
          f"t={elapsed:.3f} сек")


@when("я вычисляю размерность обоими методами с профилированием cProfile")
def step_both_profiled(context):
    res_bc, prof_bc, t_bc = profile_function(box_counting_2d, context["array"])
    context["result_bc"] = res_bc
    context["profile_bc"] = prof_bc
    context["timings"]["box_counting"] = t_bc

    res_td, prof_td, t_td = profile_function(
        delaunay_dimension, context["array"])
    context["result_td"] = res_td
    context["profile_td"] = prof_td
    context["timings"]["delaunay"] = t_td

    print(f"  [Box-counting] D={res_bc['dimension']:.4f}, t={t_bc:.3f} сек")
    print(f"  [Делоне]       D={res_td['dimension']:.4f}, t={t_td:.3f} сек")


# Then

@then(parsers.parse(
    "оба метода должны дать размерность в диапазоне от {dim_min:f} до {dim_max:f}"
))
def step_dim_in_range(context, dim_min, dim_max):
    d_bc = context["result_bc"]["dimension"]
    d_td = context["result_td"]["dimension"]
    assert dim_min <= d_bc <= dim_max, (
        f"box-counting D={d_bc:.4f} вне диапазона [{dim_min}, {dim_max}]"
    )
    assert dim_min <= d_td <= dim_max, (
        f"Делоне D={d_td:.4f} вне диапазона [{dim_min}, {dim_max}]"
    )


@then(parsers.parse(
    "расхождение между методами не должно превышать {max_diff:f}"
))
def step_methods_agree(context, max_diff):
    d_bc = context["result_bc"]["dimension"]
    d_td = context["result_td"]["dimension"]
    diff = abs(d_bc - d_td)
    assert diff < max_diff, (
        f"|D_bc - D_td| = {diff:.4f} >= {max_diff}"
    )


@then(parsers.parse(
    "R-squared обоих методов должен быть больше {threshold:f}"
))
def step_r2_positive(context, threshold):
    r2_bc = context["result_bc"]["r_squared"]
    r2_td = context["result_td"]["r_squared"]
    assert r2_bc > threshold, f"Box-counting R^2 = {r2_bc:.4f} <= {threshold}"
    assert r2_td > threshold, f"Делоне R^2 = {r2_td:.4f} <= {threshold}"


@then(parsers.parse('результаты профилирования сохранены в results для "{dataset}"'))
def step_save_profiles(context, dataset):
    ds_dir = os.path.join(RESULTS_DIR, dataset)
    os.makedirs(ds_dir, exist_ok=True)

    bc_report = save_profile_report(
        context["profile_bc"],
        os.path.join(ds_dir, "box_counting_2d"),
    )
    td_report = save_profile_report(
        context["profile_td"],
        os.path.join(ds_dir, "delaunay_dimension"),
    )

    t_total = context["timings"]["box_counting"] + \
        context["timings"]["delaunay"]
    summary = {
        "dataset": dataset,
        "resolution": context.get("resolution", "N/A"),
        "fill_ratio": context.get("fill_ratio", "N/A"),
        "n_points": context.get("n_points", "N/A"),
        "timings": {
            "generation_sec": round(context["timings"]["generation"], 4),
            "box_counting_sec": round(context["timings"]["box_counting"], 4),
            "delaunay_sec": round(context["timings"]["delaunay"], 4),
            "total_algorithms_sec": round(t_total, 4),
        },
        "box_counting": {
            "dimension": round(context["result_bc"]["dimension"], 6),
            "r_squared": round(context["result_bc"]["r_squared"], 6),
            "num_scales": len(context["result_bc"]["epsilons"]),
        },
        "delaunay": {
            "dimension": round(context["result_td"]["dimension"], 6),
            "r_squared": round(context["result_td"]["r_squared"], 6),
            "num_scales": len(context["result_td"]["deltas"]),
        },
        "dimension_diff": round(abs(
            context["result_bc"]["dimension"] -
            context["result_td"]["dimension"]
        ), 6),
    }

    with open(os.path.join(ds_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_timing_table(context)

    print("  --- Box-counting 2D: топ функций по cumtime ---")
    for line in bc_report.strip().split("\n")[:15]:
        print(f"    {line}")

    print("\n  --- Делоне: топ функций по cumtime ---")
    for line in td_report.strip().split("\n")[:15]:
        print(f"    {line}")
    print()

    print(f"  Профили сохранены:")
    print(f"    {ds_dir}/box_counting_2d.prof    (.txt)")
    print(f"    {ds_dir}/delaunay_dimension.prof  (.txt)")
    print(f"    {ds_dir}/summary.json")


@then(parsers.parse(
    "оба метода приближают аналитическое значение {expected:f} с допуском {tol:f}"
))
def step_both_near_analytical(context, expected, tol):
    d_bc = context["result_bc"]["dimension"]
    d_td = context["result_td"]["dimension"]
    assert abs(d_bc - expected) < tol, (
        f"Box-counting D={d_bc:.4f}, ожидание={expected:.4f}, допуск={tol}"
    )
    assert abs(d_td - expected) < tol, (
        f"Делоне D={d_td:.4f}, ожидание={expected:.4f}, допуск={tol}"
    )


@then(parsers.parse(
    "это воспроизводит утверждения модульного теста test_carpet_methods_agree с допуском {tol:f}"
))
def step_reproduces_unit_test(context, tol):
    d_bc = context["result_bc"]["dimension"]
    d_td = context["result_td"]["dimension"]
    diff = abs(d_bc - d_td)

    assert diff < tol, (
        f"BDD: |D_bc - D_td| = {diff:.4f} >= {tol}  "
        f"(не совпадает с unit-тестом test_carpet_methods_agree)"
    )

    print(f"\n  [Сравнение с unit-тестом test_carpet_methods_agree]")
    print(f"  Unit-тест: assert abs(D_bc - D_td) < {tol}")
    print(
        f"  BDD:       |{d_bc:.4f} - {d_td:.4f}| = {diff:.4f} < {tol}  OK")
    print(f"  Результаты BDD и модульного тестирования согласованы.")

    # Сохраняем сравнение
    comp_dir = os.path.join(RESULTS_DIR, "unit_test_comparison")
    os.makedirs(comp_dir, exist_ok=True)
    comparison = {
        "unit_test": "test_carpet_methods_agree",
        "unit_test_file": "tests/test_algorithm_comparison.py",
        f"assertion": "abs(D_bc - D_td) < {tol}",
        "bdd_d_bc": round(d_bc, 6),
        "bdd_d_td": round(d_td, 6),
        "bdd_diff": round(diff, 6),
        "passed": bool(diff < tol),
    }

    with open(
        os.path.join(comp_dir, "comparison.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
