"""
Генератор тестовых наборов данных для нагрузочного тестирования
алгоритмов вычисления фрактальной размерности.

Метод генерации: фрактальная перколяция Мандельброта
====================================================

В отличие от сайт-перколяции (случайный шум без самоподобия),
фрактальная перколяция Мандельброта создает НАСТОЯЩИЙ фрактальный
объект с доказанным свойством самоподобия.

Алгоритм:
  1. Начинаем с полностью заполненного массива 3^d x 3^d (d = BASE_DEPTH).
  2. На каждом уровне рекурсии (0..d-1) делим массив на 3x3 блоков.
  3. Каждый блок независимо выживает с вероятностью p (retain_prob)
     или удаляется целиком (все пиксели -> False).
  4. Удаление каскадно: если блок удален на уровне k,
     все его потомки на уровнях k+1..d-1 тоже удалены.
  5. Результат масштабируется до целевого resolution (кратного 3^d).

Аналитическая размерность:
    D = 2 + ln(p) / ln(3)

    где p = retain_prob -- вероятность выживания блока.
    Параметр p вычисляется из целевой плотности fill_ratio:
        p = fill_ratio^(1/d)

    Примеры:
        fill=0.10 -> p=0.681 -> D=1.651
        fill=0.35 -> p=0.840 -> D=1.841
        fill=0.08 -> p=0.657 -> D=1.617
        fill=0.25 -> p=0.794 -> D=1.790

Ключевые характеристики, влияющие на производительность алгоритмов
=================================================================

1. РАЗМЕР МАССИВА (resolution, NxN):
   - box_counting_2d: сложность O(N^2 * K), где K -- число масштабов.
     Внутренний цикл Python перебирает все (N/eps)^2 блоков для каждого eps.
     При eps=1 выполняется N^2 итераций -- это основное узкое место.
   - delaunay_dimension: сложность O(P * K + T(S)*K),
     где P = число занятых пикселей, S = размер подвыборки, K = num_scales,
     T(S) = время триангуляции подвыборки (Scipy Delaunay).
     Основное узкое место -- Python-цикл по P точкам при построении
     словаря ячеек на каждом масштабе.

2. ПЛОТНОСТЬ ЗАПОЛНЕНИЯ (fill_ratio, она же p^d):
   - box_counting_2d: СЛАБОЕ влияние. Метод box.any() на numpy-срезе
     возвращает True за O(1) при наличии хотя бы одного пикселя,
     поэтому плотность почти не влияет на время box-counting.
   - delaunay_dimension: СИЛЬНОЕ влияние. Количество точек P = N^2 * fill.
     Каждая точка обрабатывается в Python-цикле на каждом масштабе,
     что напрямую увеличивает время работы Делоне.

3. ФРАКТАЛЬНАЯ СТРУКТУРА (самоподобие):
   Фрактальная перколяция Мандельброта обладает точным статистическим
   самоподобием: структура на масштабе 3^k выглядит так же, как на 3^(k+1).
   Это обеспечивает линейную зависимость log N(eps) vs log(1/eps)
   с высоким R^2 и размерностью D < 2.

   В отличие от сайт-перколяции (случайный шум), Мандельброт-перколяция:
   - имеет аналитически известную D
   - дает D < 2 при любом fill < 1
   - обеспечивает согласованность оценок box-counting и Делоне

Кодировка наименований наборов данных
=====================================
    perc_S{resolution}_f{fill_percent}

    perc  - метод генерации (фрактальная перколяция Мандельброта)
    S2916 - размер массива 2916x2916
    f10   - целевая плотность заполнения ~10%

Расшифровка наборов:
    perc_S2916_f10 - 2916x2916, fill~10%, D_anal=1.651 (разреженный, средний)
    perc_S2916_f35 - 2916x2916, fill~35%, D_anal=1.841 (плотный, средний)
    perc_S5832_f08 - 5832x5832, fill~8%,  D_anal=1.617 (разреженный, большой)
    perc_S5832_f25 - 5832x5832, fill~25%, D_anal=1.790 (плотный, большой)

Выбор размеров:
    2916 = 2^2 * 3^6 -> 20 делителей -> 20 масштабов box-counting
    5832 = 2^3 * 3^6 -> 27 делителей -> 27 масштабов box-counting

    Базовый размер фрактала: 3^6 = 729 (6 уровней рекурсии).
    Масштабирование до целевого resolution: 2916/729 = 4x, 5832/729 = 8x.

Калибровка по предварительному эксперименту (HP Pavilion, Ryzen 5 5000):
    2916x2916: BC~33-42с, TD~5-12с. Итого ~40-55с.   [medium]
    5832x5832: BC~134-229с, TD~13-46с. Итого ~150-270с. [large]
"""

import numpy as np
import json
import os
import time

BASE_DEPTH = 6
BASE_SIZE = 3 ** BASE_DEPTH  # 729


def _mandelbrot_percolation(depth: int,
                            retain_prob: float,
                            seed: int = 42) -> np.ndarray:
    """
    Генерирует фрактальную перколяцию Мандельброта на сетке 3^depth x 3^depth.

    На каждом уровне рекурсии (0..depth-1) массив разбивается на блоки,
    и каждый блок независимо удаляется с вероятностью (1 - retain_prob).
    Удаление каскадно: потомки удаленного блока тоже удалены.

    Parameters
    ----------
    depth : int
        Глубина рекурсии. Размер выхода = 3^depth x 3^depth.
    retain_prob : float
        Вероятность выживания блока на каждом уровне (0..1).
    seed : int
        Зерно генератора для воспроизводимости.

    Returns
    -------
    np.ndarray
        Булев массив shape (3^depth, 3^depth).
    """
    rng = np.random.default_rng(seed)
    size = 3 ** depth
    arr = np.ones((size, size), dtype=bool)

    for level in range(depth):
        # На уровне level массив делится на n*n блоков размером block x block
        n = 3 ** (level + 1)
        block = size // n

        # Каждый блок выживает с вероятностью retain_prob
        survive = rng.random((n, n)) < retain_prob

        # Расширяем маску выживания до полного разрешения
        survive_full = np.repeat(
            np.repeat(survive, block, axis=0),
            block, axis=1,
        )

        # Каскадное удаление: AND с текущим состоянием
        # Если родительский блок уже удален (False), потомки остаются False
        arr &= survive_full

    return arr


def generate_percolation_array(resolution: int,
                               fill_ratio: float,
                               seed: int = 42) -> np.ndarray:
    """
    Генерирует самоподобный фрактальный объект методом перколяции Мандельброта.

    Параметр fill_ratio задает целевую плотность заполнения.
    Из него вычисляется retain_prob (вероятность выживания блока):
        retain_prob = fill_ratio^(1 / depth)

    Аналитическая фрактальная размерность:
        D = 2 + ln(retain_prob) / ln(3)

    Алгоритм автоматически определяет глубину рекурсии depth
    и коэффициент масштабирования scale из resolution:
        - находит наибольшую степень 3, делящую resolution
        - depth = показатель этой степени (но не более BASE_DEPTH)
        - scale = resolution / 3^depth
    Например: 2916 = 4 * 3^6 -> depth=6, scale=4.
              243 = 3^5       -> depth=5, scale=1.

    Parameters
    ----------
    resolution : int
        Размер стороны квадратного массива (пиксели).
    fill_ratio : float
        Целевая плотность заполнения (0..1).
    seed : int
        Зерно генератора для воспроизводимости.

    Returns
    -------
    np.ndarray
        Булев массив shape (resolution, resolution).
    """
    # Определяем глубину рекурсии: наибольшая степень 3, делящая resolution
    depth = 0
    temp = resolution
    while temp % 3 == 0 and depth < BASE_DEPTH:
        temp //= 3
        depth += 1

    if depth < 2:
        raise ValueError(
            f"resolution ({resolution}) должен делиться хотя бы на 3^2=9. "
            f"Рекомендуемые значения: 243, 729, 1458, 2916, 5832, ..."
        )

    base_size = 3 ** depth
    scale = resolution // base_size

    if resolution != base_size * scale:
        raise ValueError(
            f"resolution ({resolution}) не разлагается как scale * 3^depth. "
            f"Попробуйте кратное {base_size}."
        )

    # retain_prob из целевой плотности для данной глубины
    retain_prob = fill_ratio ** (1.0 / depth)

    # Генерируем фрактал на базовой сетке
    base_arr = _mandelbrot_percolation(depth, retain_prob, seed)

    # Масштабируем до целевого resolution
    if scale > 1:
        arr = np.repeat(
            np.repeat(base_arr, scale, axis=0),
            scale, axis=1,
        )
    else:
        arr = base_arr

    return arr


def analytical_dimension(fill_ratio: float, depth: int = BASE_DEPTH) -> float:
    """
    Вычисляет аналитическую фрактальную размерность для
    перколяции Мандельброта с заданной целевой плотностью.

    D = 2 + ln(retain_prob) / ln(3)
      = 2 + ln(fill_ratio) / (depth * ln(3))
    """
    retain_prob = fill_ratio ** (1.0 / depth)
    return 2.0 + np.log(retain_prob) / np.log(3.0)


def dataset_name(resolution: int, fill_ratio: float) -> str:
    """Формирует кодированное название набора данных."""
    fill_pct = int(round(fill_ratio * 100))
    return f"perc_S{resolution}_f{fill_pct:02d}"


def save_dataset(arr: np.ndarray,
                 name: str,
                 fill_ratio: float = None,
                 output_dir: str = "datasets") -> str:
    """
    Сохраняет набор данных (.npy) и метаданные (.json).

    Returns
    -------
    str
        Путь к сохраненному .npy файлу.
    """
    os.makedirs(output_dir, exist_ok=True)

    npy_path = os.path.join(output_dir, f"{name}.npy")
    np.save(npy_path, arr)

    meta = {
        "name": name,
        "method": "mandelbrot_fractal_percolation",
        "base_depth": BASE_DEPTH,
        "base_size": BASE_SIZE,
        "resolution": int(arr.shape[0]),
        "scale_factor": int(arr.shape[0]) // BASE_SIZE,
        "shape": list(arr.shape),
        "fill_ratio_target": float(fill_ratio) if fill_ratio is not None else None,
        "fill_ratio_actual": round(float(arr.sum()) / float(arr.size), 6),
        "occupied_pixels": int(arr.sum()),
        "total_pixels": int(arr.size),
        "dtype": str(arr.dtype),
        "file_size_bytes": os.path.getsize(npy_path),
    }

    if fill_ratio is not None and fill_ratio > 0:
        rp = fill_ratio ** (1.0 / BASE_DEPTH)
        meta["retain_prob"] = round(float(rp), 6)
        meta["analytical_dimension"] = round(
            2.0 + float(np.log(rp) / np.log(3.0)), 6
        )

    json_path = os.path.join(output_dir, f"{name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return npy_path


def load_dataset(name: str, data_dir: str = "datasets") -> np.ndarray:
    """Загружает набор данных из .npy файла."""
    return np.load(os.path.join(data_dir, f"{name}.npy"))


def load_or_generate(name: str,
                     resolution: int,
                     fill_ratio: float,
                     data_dir: str = "datasets") -> np.ndarray:
    """
    Загружает набор из файла, если он существует,
    иначе генерирует и сохраняет.

    Parameters
    ----------
    name : str
        Кодированное имя набора (perc_S2916_f10).
    resolution : int
        Размер массива.
    fill_ratio : float
        Целевая плотность заполнения.
    data_dir : str
        Каталог с .npy файлами.

    Returns
    -------
    np.ndarray
        Булев массив shape (resolution, resolution).
    """
    npy_path = os.path.join(data_dir, f"{name}.npy")
    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        if arr.shape == (resolution, resolution):
            return arr
    # Файл не найден или не совпадает -> генерируем
    arr = generate_percolation_array(resolution, fill_ratio)
    save_dataset(arr, name, fill_ratio, data_dir)
    return arr


# =====================================================================
# Конфигурация наборов данных
# =====================================================================
# Калибровано по предварительному эксперименту
# (HP Pavilion, Ryzen 5 5000, ~90% CPU с фоновыми процессами):
#
#   medium (средний):  40 сек -- 1.5 мин  (оба алгоритма суммарно)
#     2916 = 2^2 * 3^6 -> 20 делителей -> BC ~42с
#
#   large  (большой):  2 -- 5 мин         (оба алгоритма суммарно)
#     5832 = 2^3 * 3^6 -> 27 делителей -> BC ~170с
#
# Два типа по плотности: разреженный (sparse) и плотный (dense).
# Два размера: средний (2916) и большой (5832).
# Итого 4 набора: 2 типа x 2 размера.
#
# retain_prob и D_analytical вычисляются из fill_ratio:
#   retain_prob = fill_ratio^(1/6)
#   D = 2 + ln(retain_prob) / ln(3)
#
#   fill=0.10 -> p=0.681 -> D=1.651
#   fill=0.35 -> p=0.840 -> D=1.841
#   fill=0.08 -> p=0.657 -> D=1.617
#   fill=0.25 -> p=0.794 -> D=1.790

DATASETS_CONFIG = [
    # Средние (2916 = 4*729 = 2^2 * 3^6)
    {
        "resolution": 2916,
        "fill_ratio": 0.10,
        "category": "medium",
        "type": "sparse",
        "description": "разреженный, средний размер, D_anal=1.651",
    },
    {
        "resolution": 2916,
        "fill_ratio": 0.35,
        "category": "medium",
        "type": "dense",
        "description": "плотный, средний размер, D_anal=1.841",
    },
    # Большие (5832 = 8*729 = 2^3 * 3^6, целевое время ~ 2--5мин)
    {
        "resolution": 5832,
        "fill_ratio": 0.08,
        "category": "large",
        "type": "sparse",
        "description": "разреженный, большой размер, D_anal=1.617",
    },
    {
        "resolution": 5832,
        "fill_ratio": 0.25,
        "category": "large",
        "type": "dense",
        "description": "плотный, большой размер, D_anal=1.790",
    },
]


def generate_all_datasets(output_dir: str = "datasets") -> list:
    """
    Генерирует и сохраняет все 4 набора данных.

    Returns
    -------
    list[dict]
        Список метаданных сгенерированных наборов.
    """
    results = []
    for cfg in DATASETS_CONFIG:
        name = dataset_name(cfg["resolution"], cfg["fill_ratio"])
        d_anal = analytical_dimension(cfg["fill_ratio"])
        rp = cfg["fill_ratio"] ** (1.0 / BASE_DEPTH)
        print(f"Генерация {name} ({cfg['description']})...")
        print(f"  retain_prob={rp:.4f}, D_analytical={d_anal:.4f}")

        t0 = time.perf_counter()
        arr = generate_percolation_array(cfg["resolution"], cfg["fill_ratio"])
        t_gen = time.perf_counter() - t0

        path = save_dataset(arr, name, cfg["fill_ratio"], output_dir)
        actual_fill = float(arr.sum()) / float(arr.size)
        info = {
            "name": name,
            "path": path,
            "resolution": cfg["resolution"],
            "fill_ratio_target": cfg["fill_ratio"],
            "fill_ratio_actual": round(actual_fill, 6),
            "retain_prob": round(float(rp), 6),
            "analytical_dimension": round(float(d_anal), 4),
            "category": cfg["category"],
            "type": cfg["type"],
            "occupied_pixels": int(arr.sum()),
            "generation_time_sec": round(t_gen, 3),
        }
        results.append(info)
        print(f"  -> {name}.npy  ({arr.sum()} px, fill_actual={actual_fill:.4f}, "
              f"{t_gen:.3f} сек)")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Генерация фрактальных наборов данных (перколяция Мандельброта)")
    print("=" * 70)
    print(f"Базовый размер: {BASE_SIZE} = 3^{BASE_DEPTH}")
    print()
    infos = generate_all_datasets()
    print("\nСводка:")
    print(f"{'Набор':<18} {'N':>5} {'fill_target':>10} {'fill_actual':>10} "
          f"{'D_anal':>7} {'px':>10} {'gen,с':>7}")
    print("-" * 75)
    for info in infos:
        print(f"  {info['name']:<16} {info['resolution']:>5} "
              f"{info['fill_ratio_target']:>10.2f} {info['fill_ratio_actual']:>10.4f} "
              f"{info['analytical_dimension']:>7.4f} "
              f"{info['occupied_pixels']:>10} {info['generation_time_sec']:>7.3f}")
