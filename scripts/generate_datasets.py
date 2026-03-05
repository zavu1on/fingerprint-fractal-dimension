"""
Генерация и сохранение всех наборов данных для нагрузочного тестирования

Запуск:
    python scripts/generate_datasets.py

Результат:
    datasets/perc_S2916_f10.npy + .json
    datasets/perc_S2916_f35.npy + .json
    datasets/perc_S5832_f08.npy + .json
    datasets/perc_S5832_f25.npy + .json

Общий размер: ~81 MB
"""

# autopep8: off

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tests.bdd.dataset_generator import generate_all_datasets


if __name__ == "__main__":
    generate_all_datasets(output_dir="datasets")

    print("\nВсе наборы данных сгенерированы в папке datasets/")
    print("Теперь можно запускать нагрузочные тесты:")
    print("  pytest tests/bdd/test_algorithm_comparison_bdd.py -v -s -k S2916")
    print("  pytest tests/bdd/test_algorithm_comparison_bdd.py -v -s -k S5832")
