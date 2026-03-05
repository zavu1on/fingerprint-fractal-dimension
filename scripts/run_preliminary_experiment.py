"""
Предварительный эксперимент: определение "максимальных" границ
производительности алгоритмов box-counting 2D и триангуляции Делоне.

Цель: измерить время работы обоих алгоритмов на массивах разного
размера и плотности, чтобы подобрать параметры наборов данных
для нагрузочного тестирования.

По результатам предварительного эксперимента корректируются
значения resolution и fill_ratio в DATASETS_CONFIG
(файл tests/bdd/dataset_generator.py).

Запуск:
    python scripts/run_preliminary_experiment.py

Результат:
    Таблица зависимости времени от размера и плотности.
    Файл results/preliminary_experiment.json с полными результатами.
"""

# autopep8: off

import sys
import os
import time
import json
import platform
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.fractal_dimension.box_counting_2d import box_counting_2d
from app.fractal_dimension.delaunay_dimension import delaunay_dimension
from tests.bdd.dataset_generator import generate_percolation_array


# Размеры для предварительного эксперимента (степени и кратные тройки)
# 243=3^5, 729=3^6, 1458=2*3^6, 2916=4*3^6, 5832=8*3^6
RESOLUTIONS = [243, 729, 1458, 2916, 5832]
FILL_RATIOS = [0.10, 0.30]
MAX_TIME_PER_TEST = 1800


def get_system_info() -> dict:
    info = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    try:
        import psutil
        info["ram_total_gb"] = round(
            psutil.virtual_memory().total / (1024 ** 3), 1)
        info["cpu_percent"] = psutil.cpu_percent(interval=1)
    except ImportError:
        info["ram_total_gb"] = "psutil не установлен"
        info["cpu_percent"] = "N/A"
    return info


def run_single_test(resolution, fill_ratio, seed=42):
    t0 = time.perf_counter()
    arr = generate_percolation_array(resolution, fill_ratio, seed)
    t_gen = time.perf_counter() - t0
    n_points = int(arr.sum())

    t0 = time.perf_counter()
    res_bc = box_counting_2d(arr)
    t_bc = time.perf_counter() - t0

    if t_bc > MAX_TIME_PER_TEST:
        return {
            "resolution": resolution,
            "fill_ratio": fill_ratio,
            "n_points": n_points,
            "t_generation": round(t_gen, 3),
            "t_box_counting": round(t_bc, 3),
            "t_delaunay": None,
            "t_total": round(t_bc, 3),
            "note": "BC превысил лимит, Делоне пропущен",
        }

    t0 = time.perf_counter()
    res_td = delaunay_dimension(arr)
    t_td = time.perf_counter() - t0

    return {
        "resolution": resolution,
        "fill_ratio": fill_ratio,
        "n_points": n_points,
        "t_generation": round(t_gen, 3),
        "t_box_counting": round(t_bc, 3),
        "t_delaunay": round(t_td, 3),
        "t_total": round(t_bc + t_td, 3),
        "dim_bc": round(res_bc["dimension"], 4),
        "dim_td": round(res_td["dimension"], 4),
        "r2_bc": round(res_bc["r_squared"], 4),
        "r2_td": round(res_td["r_squared"], 4),
        "n_scales_bc": len(res_bc["epsilons"]),
        "n_scales_td": len(res_td["deltas"]),
    }


def main():
    print("=" * 78)
    print("ПРЕДВАРИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ")
    print("Определение временных границ производительности алгоритмов")
    print("=" * 78)

    sys_info = get_system_info()
    print(f"\nСистема: {sys_info['os']}")
    print(f"Процессор: {sys_info['processor']}")
    print(f"Python: {sys_info['python']}")
    if isinstance(sys_info.get("ram_total_gb"), float):
        print(f"RAM: {sys_info['ram_total_gb']} GB")
        print(f"CPU загрузка: {sys_info['cpu_percent']}%")
    print()

    all_results = {"system_info": sys_info, "tests": []}

    header = (f"{'Размер':>8} {'Fill':>5} {'Точки':>10} "
              f"{'Gen,с':>7} {'BC,с':>8} {'TD,с':>8} {'Итого,с':>9} "
              f"{'D_bc':>7} {'D_td':>7}")
    print(header)
    print("-" * len(header))

    for resolution in RESOLUTIONS:
        for fill_ratio in FILL_RATIOS:
            print(f"{resolution:>8} {fill_ratio:>5.2f} ", end="", flush=True)

            try:
                r = run_single_test(resolution, fill_ratio)
                all_results["tests"].append(r)

                t_td_str = (f"{r['t_delaunay']:>8.3f}"
                            if r["t_delaunay"] is not None else "    SKIP")
                t_total = r.get("t_total", r["t_box_counting"])
                dim_bc = r.get("dim_bc", 0)
                dim_td = r.get("dim_td", 0)

                print(f"{r['n_points']:>10} {r['t_generation']:>7.3f} "
                      f"{r['t_box_counting']:>8.3f} {t_td_str} "
                      f"{t_total:>9.3f} {dim_bc:>7.4f} {dim_td:>7.4f}")

                if t_total > MAX_TIME_PER_TEST:
                    print(f"  -> Превышен лимит, пропуск для fill={fill_ratio}")
                    break

            except Exception as e:
                print(f"  ОШИБКА: {e}")
                all_results["tests"].append({
                    "resolution": resolution,
                    "fill_ratio": fill_ratio,
                    "error": str(e),
                })

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "preliminary_experiment.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nРезультаты сохранены: {out_path}")
    print()
    print("РЕКОМЕНДАЦИИ ПО ПОДБОРУ РАЗМЕРОВ:")
    print("-" * 50)
    print("  Средний набор (40с--1.5мин):  resolution,")
    print("    при котором t_total попадает в [40, 90] секунд.")
    print("  Большой набор (2--5мин):  resolution,")
    print("    при котором t_total попадает в [120, 300] секунд.")
    print()
    print("  Если время НЕ попадает в нужный диапазон,")
    print("  скорректируйте DATASETS_CONFIG в:")
    print("  tests/bdd/dataset_generator.py")


if __name__ == "__main__":
    main()