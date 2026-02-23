import os
import numpy as np

from src.helpers.koch_curve import generate_koch_curve, visualize_koch_curve
from src.helpers.sierpinski_carpet import generate_sierpinski_carpet, visualize_sierpinski_carpet
from src.helpers.sierpinski_triangle import generate_sierpinski_triangle, visualize_sierpinski_triangle
from src.helpers.menger_sponge import generate_menger_sponge, visualize_menger_sponge

from src.fractal_dimension.box_counting_2d import box_counting_2d, visualize_box_counting_2d
from src.fractal_dimension.box_counting_3d import box_counting_3d, visualize_box_counting_3d
from src.fractal_dimension.delaunay_dimension import delaunay_dimension, visualize_delaunay

from src.image.png_to_2d import png_to_2d
from src.image.png_to_3d import png_to_3d


DEMO_2D = {
    "\\kc2d": [
        generate_koch_curve,
        visualize_koch_curve,
        np.round(np.log(4) / np.log(3), 5),
    ],
    "\\st2d": [
        generate_sierpinski_triangle,
        visualize_sierpinski_triangle,
        np.round(np.log(3) / np.log(2), 5),

    ],
    "\\sc2d": [
        generate_sierpinski_carpet,
        visualize_sierpinski_carpet,
        np.round(np.log(8) / np.log(3), 5)
    ],
}
METHOD_2D = {
    "\\bc": [box_counting_2d, visualize_box_counting_2d],
    "\\td": [delaunay_dimension, visualize_delaunay],
}

DEMO_3D = {
    "\\ms3d": [
        generate_menger_sponge,
        visualize_menger_sponge,
        np.round(np.log(20) / np.log(3), 5),
    ],
}


def run_2d_demo(cmd: str, method: str):
    if cmd not in DEMO_2D:
        print("Неверная команда")
        return
    if method not in METHOD_2D:
        print("Неверная команда")
        return

    gen_func, plot_func, analytical_dim = DEMO_2D[cmd]
    method_func, plot_method_func = METHOD_2D[method]

    tensor = gen_func()

    estimated_res = method_func(tensor)
    estimated_dim = np.round(estimated_res["dimension"], 5)

    print("Аналитическая размерность:", analytical_dim)
    print("Расчетная размерность:    ", estimated_dim)
    print("Ошибка:                   ", np.round(
        np.abs(estimated_dim - analytical_dim, 5)))
    print(
        "Коэффициент детерминированности линейной регрессии:",
        np.round(estimated_res["r_squared"], 5)
    )

    print("Отрисовка 2d объекта...")
    figure = plot_func(tensor)
    figure.show()
    input("Нажмите Enter для продолжения")

    print("Отрисовка результатов метода...")
    plot = plot_method_func(
        tensor,
        estimated_res,
        # scale_indices
    )
    plot.show()
    input("Нажмите Enter для продолжения")


def run_3d_demo(cmd: str):
    if cmd not in DEMO_3D:
        print("Неверная команда")
        return

    gen_func, plot_func, analytical_dim = DEMO_3D[cmd]

    tensor = gen_func()

    estimated_res = box_counting_3d(tensor)
    estimated_dim = np.round(estimated_res["dimension"], 5)

    print("Аналитическая размерность:", analytical_dim)
    print("Расчетная размерность:    ", estimated_dim)
    print("Ошибка:                   ", np.round(
        np.abs(estimated_dim - analytical_dim, 5)))
    print(
        "Коэффициент детерминированности линейной регрессии:",
        np.round(estimated_res["r_squared"], 5)
    )

    print("Отрисовка 3d объекта...")
    figure = plot_func(tensor)
    figure.show()
    input("Нажмите Enter для продолжения")

    print("Отрисовка работы метода...")
    plot = visualize_box_counting_3d(
        tensor,
        estimated_res,
    )
    plot.show()
    input("Нажмите Enter для продолжения")


def main():
    print("Вычисление фрактальной размерности фрактального объекта (папиллярного узора пальца руки)")

    cmd = ""

    while cmd != "\\e":
        print("Выберете операцию:")
        print("Проверить достоверность методов:")
        print("\\kc2d - рассчитать размерность кривой Коха")
        print("\\st2d - рассчитать размерность треугольника Серпинского")
        print("\\sc2d - рассчитать размерность ковра Серпинского")
        print("\\ms3d - рассчитать размерность губки Менгера")
        print("-" * 50)
        print("Рассчитать фрактальную размерность:")
        print("\\bc2d - методом подсчета квадратами")
        print("\\bc3d - методом подсчета кубами")
        print("\\td2d - триангуляцией Делоне")
        print("-" * 50)
        print("\\e - выход")

        cmd = input(">> ").lower()

        if cmd in DEMO_2D.keys():
            print("Выберете метод вычисления размерности:")
            print("\\bc - метод подсчета квадратами")
            print("\\td - метод триангуляции Делоне")

            method = input(">> ").lower()

            run_2d_demo(cmd, method)
        elif cmd in DEMO_3D.keys():
            run_3d_demo(cmd)
        elif cmd in ["\\bc2d", "\\bc3d", "\\td2d"]:
            print("Введите путь к изображению: ")
            path = input(">> ")

            if not os.path.exists(path):
                print("Файл не найден")
                continue
            if not path.endswith(".png"):
                print("Неверное расширение файла")
                continue

            method_func = None
            method_plot_func = None
            load_png_func = None
            kwargs = {}

            if cmd == "\\bc2d":
                method_func = box_counting_2d
                method_plot_func = visualize_box_counting_2d
                load_png_func = png_to_2d
                kwargs = {
                    "min_box": 2
                }
            elif cmd == "\\bc3d":
                method_func = box_counting_3d
                method_plot_func = visualize_box_counting_3d
                load_png_func = png_to_3d
                kwargs = {
                    "min_box": 4
                }
            elif cmd == "\\td2d":
                method_func = delaunay_dimension
                method_plot_func = visualize_delaunay
                load_png_func = png_to_2d

            image = load_png_func(path)

            estimated_res = method_func(image, **kwargs)
            estimated_dim = np.round(estimated_res["dimension"], 5)

            print("Расчетная размерность: ", estimated_dim)
            print(
                "Коэффициент детерминированности линейной регрессии:",
                np.round(estimated_res["r_squared"], 5)
            )

            print("Отрисовка работы метода...")
            plot = method_plot_func(image, estimated_res)
            plot.show()
            input("Нажмите Enter для продолжения")
        elif cmd == "\\e":
            break
        else:
            print("Неверная команда")


if __name__ == "__main__":
    main()
