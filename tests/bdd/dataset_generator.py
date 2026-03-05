"""
Генератор тестовых наборов данных для нагрузочного тестирования
алгоритмов вычисления фрактальной размерности

Ключевые характеристики, влияющие на производительность алгоритмов
=================================================================

1. РАЗМЕР МАССИВА (resolution, NxN):
   - box_counting_2d: сложность O(N^2 * K), где K -число масштабов.
     Внутренний цикл Python перебирает все (N/eps)^2 блоков для каждого eps.
     При eps=1 выполняется N^2 итераций - это основное узкое место.
   - delaunay_dimension: сложность O(P * K + T(S)*K),
     где P = число занятых пикселей, S = размер подвыборки, K = num_scales,
     T(S) = время триангуляции подвыборки (Scipy Delaunay).
     Основное узкое место - Python-цикл по P точкам при построении
     словаря ячеек на каждом масштабе.

2. ПЛОТНОСТЬ ЗАПОЛНЕНИЯ (fill_ratio, p):
   - box_counting_2d: СЛАБОЕ влияние. Метод box.any() на numpy-срезе
     возвращает True за O(1) при наличии хотя бы одного пикселя,
     поэтому плотность почти не влияет на время box-counting.
   - delaunay_dimension: СИЛЬНОЕ влияние. Количество точек P = N^2 * p.
     При p=0.10 на 2916x2916 -> ~850 тыс. точек.
     При p=0.35 на 2916x2916 -> ~2.98 млн. точек (в 3.5 раза больше).
     Каждая точка обрабатывается в Python-цикле на каждом масштабе,
     что напрямую увеличивает время работы Делоне.

3. ПРОСТРАНСТВЕННАЯ СТРУКТУРА (метод генерации):
   Используется сайт-перколяция: каждый пиксель независимо занят
   с вероятностью p. При p вблизи критического порога p_c ~ 0.5927
   образуются фрактальные кластеры с размерностью D ~ 91/48 ~ 1.896.
   При p << p_c - изолированные мелкие кластеры (D < 2).
   При p >> p_c - почти сплошное заполнение (D -> 2).

   Это влияет на число уникальных ячеек в подвыборке Делоне:
   при разреженных кластерах подвыборка дает меньше точек -> быстрее,
   при плотном заполнении подвыборка менее эффективна -> медленнее.

Кодировка наименований наборов данных
=====================================
    perc_S{resolution}_f{fill_percent}

    perc - метод генерации (перколяция)
    S729 - размер массива 729x729
    f12  - плотность заполнения ~12%

Расшифровка наборов:
    perc_S2916_f10 - 2916x2916, 10% заполнения  ~850 тыс. точек (разреженный, средний)
    perc_S2916_f35 - 2916x2916, 35% заполнения  ~2.98 млн. точек (плотный, средний)
    perc_S5832_f08 - 5832x5832, 8% заполнения   ~2.72 млн. точек (разреженный, большой)
    perc_S5832_f25 - 5832x5832, 25% заполнения  ~8.50 млн. точек (плотный, большой)

Выбор размеров:
    2916 = 2^2 * 3^6 -> делители до 1458: {1,2,3,4,6,9,...,1458} - 20 масштабов
    5832 = 2^3 * 3^6 -> делители до 2916: {1,2,3,4,6,8,...,2916} - 27 масштабов
    Оба размера обеспечивают большое число делителей для
    box_counting_2d с use_divisors=True, исключая граничные артефакты.

Калибровка по предварительному эксперименту (HP Pavilion, Ryzen 5 5000):
    1458x1458: BC=10с, TD=5с (fill=0.30). Итого ~15с.
    2916x2916: прогноз BC~42с + TD~20с (fill=0.35). [medium]
    5832x5832: прогноз BC~170с + TD~50с (fill=0.25). [large]
"""

import numpy as np
import json
import os
import time


def generate_percolation_array(resolution: int,
                               fill_ratio: float,
                               seed: int = 42) -> np.ndarray:
    """
    Генерирует 2D булев массив методом сайт-перколяции

    Каждый пиксель независимо занят с вероятностью fill_ratio.
    Структура зависит от близости fill_ratio к критическому
    порогу перколяции p_c ~ 0.5927

    Parameters
    ----------
    resolution : int
        Размер стороны квадратного массива (пиксели).
    fill_ratio : float
        Вероятность занятости каждого пикселя (0..1).
    seed : int
        Зерно генератора для воспроизводимости.

    Returns
    -------
    np.ndarray
        Булев массив shape (resolution, resolution).
    """
    rng = np.random.default_rng(seed)
    return rng.random((resolution, resolution)) < fill_ratio


def dataset_name(resolution: int, fill_ratio: float) -> str:
    fill_pct = int(round(fill_ratio * 100))
    return f"perc_S{resolution}_f{fill_pct:02d}"


def save_dataset(arr: np.ndarray,
                 name: str,
                 output_dir: str = "datasets") -> str:
    os.makedirs(output_dir, exist_ok=True)

    npy_path = os.path.join(output_dir, f"{name}.npy")
    np.save(npy_path, arr)

    meta = {
        "name": name,
        "resolution": int(arr.shape[0]),
        "shape": list(arr.shape),
        "fill_ratio_actual": round(float(arr.sum()) / float(arr.size), 6),
        "occupied_pixels": int(arr.sum()),
        "total_pixels": int(arr.size),
        "dtype": str(arr.dtype),
        "file_size_bytes": os.path.getsize(npy_path),
    }

    json_path = os.path.join(output_dir, f"{name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return npy_path


def load_dataset(name: str, data_dir: str = "datasets") -> np.ndarray:
    return np.load(os.path.join(data_dir, f"{name}.npy"))


def load_or_generate(
    name: str,
    resolution: int,
    fill_ratio: float,
    data_dir: str = "datasets"
) -> np.ndarray:
    npy_path = os.path.join(data_dir, f"{name}.npy")
    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        if arr.shape == (resolution, resolution):
            return arr

    arr = generate_percolation_array(resolution, fill_ratio)
    save_dataset(arr, name, data_dir)
    return arr


DATASETS_CONFIG = [
    # Средние (2916 = 4*729 = 2^2 * 3^6)
    {
        "resolution": 2916,
        "fill_ratio": 0.10,
        "category": "medium",
        "type": "sparse",
        "description": "разреженный, средний размер",
    },
    {
        "resolution": 2916,
        "fill_ratio": 0.35,
        "category": "medium",
        "type": "dense",
        "description": "плотный, средний размер",
    },
    # Большие (5832 = 8*729 = 2^3 * 3^6)
    {
        "resolution": 5832,
        "fill_ratio": 0.08,
        "category": "large",
        "type": "sparse",
        "description": "разреженный, большой размер",
    },
    {
        "resolution": 5832,
        "fill_ratio": 0.25,
        "category": "large",
        "type": "dense",
        "description": "плотный, большой размер",
    },
]


def generate_all_datasets(output_dir: str = "datasets") -> list:
    results = []
    for cfg in DATASETS_CONFIG:
        name = dataset_name(cfg["resolution"], cfg["fill_ratio"])
        print(f"Генерация {name} ({cfg['description']})...")

        t0 = time.perf_counter()
        arr = generate_percolation_array(cfg["resolution"], cfg["fill_ratio"])
        t_gen = time.perf_counter() - t0

        path = save_dataset(arr, name, output_dir)
        info = {
            "name": name,
            "path": path,
            "resolution": cfg["resolution"],
            "fill_ratio": cfg["fill_ratio"],
            "category": cfg["category"],
            "type": cfg["type"],
            "occupied_pixels": int(arr.sum()),
            "generation_time_sec": round(t_gen, 3),
        }
        results.append(info)
        print(f"  -> {name}.npy  ({arr.sum()} px, {t_gen:.3f} сек)")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Генерация наборов данных для нагрузочного тестирования")
    print("=" * 60)
    infos = generate_all_datasets()
    print("\nСводка:")
    for info in infos:
        print(f"  {info['name']}: {info['resolution']}x{info['resolution']}, "
              f"fill={info['fill_ratio']}, "
              f"{info['occupied_pixels']} px, "
              f"gen={info['generation_time_sec']} сек")
