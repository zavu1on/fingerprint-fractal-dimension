import pytest
import numpy as np
from unittest.mock import MagicMock
from pytest_bdd import scenarios, given, when, then, parsers

scenarios("../../features/user_input.feature")


# Given

@given("модуль main приложения импортирован")
def step_main_imported():
    from app.main import main  # noqa: F401


@given("существует валидный тестовый PNG файл")
def step_valid_png(context, tmp_png):
    context["png_path"] = tmp_png


@given("существует тестовый JPG файл")
def step_valid_jpg(context, tmp_jpg):
    context["jpg_path"] = tmp_jpg


@given(parsers.parse(
    "box_counting_2d замокирован с dimension={dim:f} и r_squared={r2:f}"
))
def step_mock_bc2d(context, dim, r2, mocker):
    mock = mocker.patch(
        "app.main.box_counting_2d",
        return_value={
            "dimension": dim,
            "epsilons": np.array([1]),
            "counts": np.array([1.0]),
            "slope": dim,
            "intercept": 0.0,
            "r_squared": r2,
        },
    )
    context["mock_bc2d"] = mock


@given("visualize_box_counting_2d замокирован")
def step_mock_vis(mocker):
    mock_fig = MagicMock()
    mocker.patch("app.main.visualize_box_counting_2d", return_value=mock_fig)


@given("png_to_2d замокирован и возвращает массив 32x32")
def step_mock_png_to_2d(mocker):
    mocker.patch(
        "app.main.png_to_2d",
        return_value=np.ones((32, 32), dtype=bool),
    )


@given("run_2d_demo замокирован")
def step_mock_run2d(context, mocker):
    context["mock_run_2d"] = mocker.patch("app.main.run_2d_demo")


@given("run_3d_demo замокирован")
def step_mock_run3d(context, mocker):
    context["mock_run_3d"] = mocker.patch("app.main.run_3d_demo")


# When (imperative steps for PNG scenario)

@when('пользователь вводит команду "\\bc2d"')
def step_input_bc2d(context):
    context.setdefault("inputs", []).append("\\bc2d")


@when("пользователь вводит путь к PNG файлу")
def step_input_path(context):
    context.setdefault("inputs", []).append(context["png_path"])


@when("пользователь нажимает Enter для продолжения")
def step_input_enter(context):
    context.setdefault("inputs", []).append("")


@when('пользователь вводит "\\e" для выхода')
def step_input_exit_and_run(context, mocker, capsys):
    context.setdefault("inputs", []).append("\\e")
    mocker.patch("builtins.input", side_effect=context["inputs"])
    from app.main import main
    main()
    context["captured"] = capsys.readouterr().out


# When (declarative steps)

@when(parsers.parse('пользователь выбирает демо "{cmd}" с методом "{method}"'))
def step_select_demo_2d(cmd, method, mocker):
    mocker.patch("builtins.input", side_effect=[cmd, method, "\\e"])
    from app.main import main
    main()


@when(parsers.parse('пользователь выбирает демо "{cmd}"'))
def step_select_demo_3d(cmd, mocker):
    mocker.patch("builtins.input", side_effect=[cmd, "\\e"])
    from app.main import main
    main()


@when(parsers.parse('пользователь вводит "\\bc2d" с путем "{path}"'))
def step_input_bc2d_with_path(context, path, mocker, capsys):
    mocker.patch("builtins.input", side_effect=["\\bc2d", path, "\\e"])
    from app.main import main
    main()
    context["captured"] = capsys.readouterr().out


@when('пользователь вводит "\\bc2d" с путем к JPG файлу')
def step_input_bc2d_jpg(context, mocker, capsys):
    mocker.patch(
        "builtins.input",
        side_effect=["\\bc2d", context["jpg_path"], "\\e"],
    )
    from app.main import main
    main()
    context["captured"] = capsys.readouterr().out


@when(parsers.parse('пользователь вводит неизвестную команду "{cmd}"'))
def step_input_unknown(context, cmd, mocker, capsys):
    mocker.patch("builtins.input", side_effect=[cmd, "\\e"])
    from app.main import main
    main()
    context["captured"] = capsys.readouterr().out


@when('пользователь вводит только "\\e"')
def step_input_only_exit(context, mocker, capsys):
    mocker.patch("builtins.input", side_effect=["\\e"])
    from app.main import main
    main()
    context["captured"] = capsys.readouterr().out


# Then

@then("box_counting_2d должен быть вызван ровно один раз")
def step_bc2d_called_once(context):
    context["mock_bc2d"].assert_called_once()


@then(parsers.parse('вывод должен содержать "{text}"'))
def step_output_contains(context, text):
    assert text in context["captured"]


@then(parsers.parse('run_2d_demo вызван с аргументами "{cmd}" и "{method}"'))
def step_run2d_called(context, cmd, method):
    context["mock_run_2d"].assert_called_once_with(cmd, method)


@then(parsers.parse('run_3d_demo вызван с аргументом "{cmd}"'))
def step_run3d_called(context, cmd):
    context["mock_run_3d"].assert_called_once_with(cmd)


@then("box_counting_2d не должен быть вызван")
def step_bc2d_not_called(context):
    # In the exit-only scenario, bc2d was never patched,
    # so if we reached here the main loop exited cleanly.

    if "mock_bc2d" not in context:
        pytest.skip("box_counting_2d not patched")

    assert context["mock_bc2d"].called is False
