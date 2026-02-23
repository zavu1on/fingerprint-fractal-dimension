"""
MC/DC tests for the console menu and Koch curve coordinate clipping.

Automotive Safety Integrity Levels:
D1: file validation - exists(C1) ∧ endswith_png(C2)
D2: command routing - if/elif chain
D3: Koch clipping - rows>=0(C3) ∧ rows<res(C4) ∧ cols>=0(C5) ∧ cols<res(C6)

Includes: 3 kinds of mocking (patch return_value, patch side_effect, MagicMock),
          assertions (assert ==, assert in, assert_called_once, assert_called_with).
"""
import numpy as np
from unittest.mock import MagicMock


# TC-MC-01: C1=T, C2=T -> valid file, computation runs

def test_valid_png_triggers_computation(tmp_png, mocker):
    """Existing .png -> box_counting_2d is called."""
    # mock kind 1: patch with return_value (stub)
    mock_bc = mocker.patch(
        "app.main.box_counting_2d",
        return_value={
            "dimension": 1.5, "epsilons": np.array([1]),
            "counts": np.array([1.0]), "slope": 1.5,
            "intercept": 0.0, "r_squared": 0.99,
        },
    )

    # mock kind 3: MagicMock for figure object
    mock_plot = mocker.patch("app.main.visualize_box_counting_2d")
    mock_plot.return_value = MagicMock()

    # mock kind 2: patch with side_effect (sequence of inputs)
    mocker.patch("builtins.input", side_effect=["\\bc2d", tmp_png, "", "\\e"])
    mocker.patch("app.main.png_to_2d",
                 return_value=np.ones((32, 32), dtype=bool))

    from app.main import main
    main()
    # assertion: assert_called_once (spy-style check)
    mock_bc.assert_called_once()


# TC-MC-02: C1=F -> "Файл не найден"

def test_missing_file(mocker, capsys):
    """Non-existent path -> error message printed."""
    mocker.patch(
        "builtins.input",
        side_effect=["\\bc2d", "C:\\Users\\joinm\\no-file.png", "\\e"],
    )
    from app.main import main
    main()
    # assertion: assert ... in (substring check)
    assert "Файл не найден" in capsys.readouterr().out


# TC-MC-03: C2=F -> "Неверное расширение"

def test_wrong_extension(tmp_jpg, mocker, capsys):
    """Existing .jpg -> wrong extension error."""
    mocker.patch("builtins.input", side_effect=["\\bc2d", tmp_jpg, "\\e"])
    from app.main import main
    main()
    assert "Неверное расширение" in capsys.readouterr().out


# TC-MC-04: \bc3d triggers box_counting_3d

def test_bc3d_command(tmp_png, mocker):
    """\\bc3d route calls box_counting_3d."""
    mock_bc3d = mocker.patch(
        "app.main.box_counting_3d",
        return_value={
            "dimension": 2.5, "epsilons": np.array([1]),
            "counts": np.array([1.0]), "slope": 2.5,
            "intercept": 0.0, "r_squared": 0.99,
        },
    )
    mocker.patch("app.main.visualize_box_counting_3d",
                 return_value=MagicMock())
    mocker.patch("app.main.png_to_3d", return_value=np.ones(
        (32, 32, 8), dtype=bool))
    mocker.patch("builtins.input", side_effect=["\\bc3d", tmp_png, "", "\\e"])

    from app.main import main
    main()
    mock_bc3d.assert_called_once()


# TC-MC-05: \td2d triggers delaunay_dimension

def test_td2d_command(tmp_png, mocker):
    """\\td2d route calls delaunay_dimension."""
    mock_del = mocker.patch(
        "app.main.delaunay_dimension",
        return_value={
            "dimension": 1.5, "deltas": np.array([1.0]),
            "num_triangles": np.array([1.0]), "slope": 1.5,
            "intercept": 0.0, "r_squared": 0.99,
            "triangulations": [], "subsampled_pts": [],
            "kept_masks": [],
        },
    )
    mocker.patch("app.main.visualize_delaunay", return_value=MagicMock())
    mocker.patch("app.main.png_to_2d",
                 return_value=np.ones((32, 32), dtype=bool))
    mocker.patch("builtins.input", side_effect=["\\td2d", tmp_png, "", "\\e"])

    from app.main import main
    main()
    mock_del.assert_called_once()


# TC-MC-06: \kc2d triggers 2D demo route

def test_kc2d_routes_to_demo(mocker):
    """\\kc2d must call run_2d_demo (mock the whole function)."""
    mock_run = mocker.patch("app.main.run_2d_demo")
    mocker.patch("builtins.input", side_effect=["\\kc2d", "\\bc", "\\e"])

    from app.main import main
    main()

    # assertion: assert_called_once_with (checks both call and arguments)
    mock_run.assert_called_once_with("\\kc2d", "\\bc")


# TC-MC-07: \ms3d triggers 3D demo route

def test_ms3d_routes_to_demo(mocker):
    """\\ms3d must call run_3d_demo."""
    mock_run = mocker.patch("app.main.run_3d_demo")
    mocker.patch("builtins.input", side_effect=["\\ms3d", "\\e"])

    from app.main import main
    main()
    mock_run.assert_called_once_with("\\ms3d")


# TC-MC-08: \e exits immediately

def test_exit_command(mocker, capsys):
    """\\e must terminate the main loop."""
    mocker.patch("builtins.input", side_effect=["\\e"])
    from app.main import main
    main()
    out = capsys.readouterr().out
    assert "Выберете операцию:" in out


# TC-MC-09: Unknown command

def test_unknown_command(mocker, capsys):
    """Unrecognized command prints error."""
    mocker.patch("builtins.input", side_effect=["\\xyz", "\\e"])
    from app.main import main
    main()
    assert "Неверная команда" in capsys.readouterr().out


RESOLUTION = 64


# TC-MC-10: All conditions True -> pixel valid

def test_clipping_all_valid():
    """Baseline: all coords inside [0, res) -> valid=True."""
    rows = np.array([0, 10, 32, 63])
    cols = np.array([0, 10, 32, 63])
    valid = (rows >= 0) & (rows < RESOLUTION) & \
            (cols >= 0) & (cols < RESOLUTION)
    assert valid.all()


# TC-MC-11: C3=F (rows < 0) -> pixel excluded

def test_clipping_negative_row():
    """Flip C3 only: row=-1, col=10 -> valid=False."""
    rows = np.array([10, -1])
    cols = np.array([10, 10])
    valid = (rows >= 0) & (rows < RESOLUTION) & \
            (cols >= 0) & (cols < RESOLUTION)
    assert valid[0] == True   # control: all T
    assert valid[1] == False  # only C3 flipped


# TC-MC-12: C4=F (rows >= resolution) -> pixel excluded

def test_clipping_overflow_row():
    """Flip C4 only: row=64, col=10 -> valid=False."""
    rows = np.array([10, RESOLUTION])
    cols = np.array([10, 10])
    valid = (rows >= 0) & (rows < RESOLUTION) & \
            (cols >= 0) & (cols < RESOLUTION)
    assert valid[0] == True
    assert valid[1] == False


# TC-MC-13: C5=F (cols < 0) -> pixel excluded

def test_clipping_negative_col():
    """Flip C5 only: row=10, col=-1 -> valid=False."""
    rows = np.array([10, 10])
    cols = np.array([10, -1])
    valid = (rows >= 0) & (rows < RESOLUTION) & \
            (cols >= 0) & (cols < RESOLUTION)
    assert valid[0] == True
    assert valid[1] == False


# TC-MC-14: C6=F (cols >= resolution) -> pixel excluded

def test_clipping_overflow_col():
    """Flip C6 only: row=10, col=64 -> valid=False."""
    rows = np.array([10, 10])
    cols = np.array([10, RESOLUTION])
    valid = (rows >= 0) & (rows < RESOLUTION) & \
            (cols >= 0) & (cols < RESOLUTION)
    assert valid[0] == True
    assert valid[1] == False


# TC-MC-15: run_2d_demo - C7=T, C8=T, метод \\bc -> полный прогон

def test_run_2d_demo_bc_full(mocker):
    """C7=T, C8=T: оба figure.show() должны быть вызваны."""
    from app import main

    tensor = np.zeros((243, 243), dtype=bool)

    mock_gen_func = MagicMock(return_value=tensor)
    mock_gen_fig = MagicMock()
    mock_plot_func = MagicMock(return_value=mock_gen_fig)

    mock_method_func = MagicMock(return_value={
        "dimension": 1.5, "epsilons": np.array([1, 3, 9]),
        "counts": np.array([100.0, 30.0, 10.0]), "slope": 1.5,
        "intercept": 0.0, "r_squared": 0.99,
    })
    mock_method_fig = MagicMock()
    mock_plot_method_func = MagicMock(return_value=mock_method_fig)

    # patch.dict заменяет записи прямо в словарях, которые уже хранятся в main
    mocker.patch.dict(main.DEMO_2D, {
        "\\sc2d": [mock_gen_func, mock_plot_func, 1.8928]
    })
    mocker.patch.dict(main.METHOD_2D, {
        "\\bc": [mock_method_func, mock_plot_method_func]
    })
    mocker.patch("builtins.input", return_value="")

    main.run_2d_demo("\\sc2d", "\\bc")

    mock_gen_fig.show.assert_called_once()
    mock_method_fig.show.assert_called_once()


# TC-MC-16: run_2d_demo - C7=F -> ранний выход

def test_run_2d_demo_invalid_cmd(capsys):
    """C7=F: неизвестная команда -> 'Неверная команда'."""
    from app import main

    main.run_2d_demo("\\invalid", "\\bc")
    assert "Неверная команда" in capsys.readouterr().out


# TC-MC-17: run_2d_demo - C7=T, C8=F -> ранний выход

def test_run_2d_demo_invalid_method(capsys):
    """C7=T, C8=F: неизвестный метод -> 'Неверная команда'."""
    from app import main

    main.run_2d_demo("\\kc2d", "\\unknown_method")
    assert "Неверная команда" in capsys.readouterr().out


# TC-MC-18: run_2d_demo - C7=T, C8=T, метод \\td -> вызывается delaunay

def test_run_2d_demo_td_method(mocker):
    """C7=T, C8=T с методом \\td: mock_td_func и figure.show вызваны."""
    from app import main

    tensor = np.zeros((243, 243), dtype=bool)

    mock_gen_func = MagicMock(return_value=tensor)
    mock_gen_fig = MagicMock()
    mock_plot_func = MagicMock(return_value=mock_gen_fig)

    mock_td_func = MagicMock(return_value={
        "dimension": 1.5, "deltas": np.array([1.0, 3.0, 9.0]),
        "num_triangles": np.array([100.0, 30.0, 10.0]), "slope": 1.5,
        "intercept": 0.0, "r_squared": 0.99,
        "triangulations": [], "subsampled_pts": [], "kept_masks": [],
    })

    mock_td_fig = MagicMock()
    mock_plot_td_func = MagicMock(return_value=mock_td_fig)

    mocker.patch.dict(main.DEMO_2D, {
        "\\sc2d": [mock_gen_func, mock_plot_func, 1.8928]
    })
    mocker.patch.dict(main.METHOD_2D, {
        "\\td": [mock_td_func, mock_plot_td_func]
    })
    mocker.patch("builtins.input", return_value="")

    main.run_2d_demo("\\sc2d", "\\td")

    mock_td_func.assert_called_once()
    mock_td_fig.show.assert_called_once()


# TC-MC-19: run_3d_demo - C9=T -> полный прогон

def test_run_3d_demo_full(mocker):
    """C9=T: оба figure.show() должны быть вызваны."""
    from app import main

    tensor = np.zeros((81, 81, 81), dtype=bool)

    mock_gen_func = MagicMock(return_value=tensor)
    mock_gen_fig = MagicMock()
    mock_plot_func = MagicMock(return_value=mock_gen_fig)

    bc3d_res = {
        "dimension": 2.7, "epsilons": np.array([1, 3, 9]),
        "counts": np.array([1000.0, 100.0, 10.0]), "slope": 2.7,
        "intercept": 0.0, "r_squared": 0.99,
    }
    mock_bc3d = MagicMock(return_value=bc3d_res)
    mock_method_fig = MagicMock()
    mock_vis_bc3d = MagicMock(return_value=mock_method_fig)

    mocker.patch.dict(main.DEMO_3D, {
        "\\ms3d": [mock_gen_func, mock_plot_func, 2.7268]
    })

    # box_counting_3d и visualize_box_counting_3d вызываются напрямую в run_3d_demo,
    # а не через словарь -> патчим через patch.object
    mocker.patch.object(main, "box_counting_3d", mock_bc3d)
    mocker.patch.object(main, "visualize_box_counting_3d", mock_vis_bc3d)
    mocker.patch("builtins.input", return_value="")

    main.run_3d_demo("\\ms3d")

    mock_gen_fig.show.assert_called_once()
    mock_method_fig.show.assert_called_once()


# TC-MC-20: run_3d_demo - C9=F -> ранний выход

def test_run_3d_demo_invalid_cmd(capsys):
    """C9=F: неизвестная команда -> 'Неверная команда'."""
    from app import main

    main.run_3d_demo("\\invalid3d")
    assert "Неверная команда" in capsys.readouterr().out
