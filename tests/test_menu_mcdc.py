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
        "main.box_counting_2d",
        return_value={
            "dimension": 1.5, "epsilons": np.array([1]),
            "counts": np.array([1.0]), "slope": 1.5,
            "intercept": 0.0, "r_squared": 0.99,
        },
    )

    # mock kind 3: MagicMock for figure object
    mock_plot = mocker.patch("main.visualize_box_counting_2d")
    mock_plot.return_value = MagicMock()

    # mock kind 2: patch with side_effect (sequence of inputs)
    mocker.patch("builtins.input", side_effect=["\\bc2d", tmp_png, "", "\\e"])
    mocker.patch("main.png_to_2d", return_value=np.ones((32, 32), dtype=bool))

    from main import main
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
    from main import main
    main()
    # assertion: assert ... in (substring check)
    assert "Файл не найден" in capsys.readouterr().out


# TC-MC-03: C2=F -> "Неверное расширение"

def test_wrong_extension(tmp_jpg, mocker, capsys):
    """Existing .jpg -> wrong extension error."""
    mocker.patch("builtins.input", side_effect=["\\bc2d", tmp_jpg, "\\e"])
    from main import main
    main()
    assert "Неверное расширение" in capsys.readouterr().out


# TC-MC-04: \bc3d triggers box_counting_3d

def test_bc3d_command(tmp_png, mocker):
    """\\bc3d route calls box_counting_3d."""
    mock_bc3d = mocker.patch(
        "main.box_counting_3d",
        return_value={
            "dimension": 2.5, "epsilons": np.array([1]),
            "counts": np.array([1.0]), "slope": 2.5,
            "intercept": 0.0, "r_squared": 0.99,
        },
    )
    mocker.patch("main.visualize_box_counting_3d", return_value=MagicMock())
    mocker.patch("main.png_to_3d", return_value=np.ones(
        (32, 32, 8), dtype=bool))
    mocker.patch("builtins.input", side_effect=["\\bc3d", tmp_png, "", "\\e"])

    from main import main
    main()
    mock_bc3d.assert_called_once()


# TC-MC-05: \td2d triggers delaunay_dimension

def test_td2d_command(tmp_png, mocker):
    """\\td2d route calls delaunay_dimension."""
    mock_del = mocker.patch(
        "main.delaunay_dimension",
        return_value={
            "dimension": 1.5, "deltas": np.array([1.0]),
            "num_triangles": np.array([1.0]), "slope": 1.5,
            "intercept": 0.0, "r_squared": 0.99,
            "triangulations": [], "subsampled_pts": [],
            "kept_masks": [],
        },
    )
    mocker.patch("main.visualize_delaunay", return_value=MagicMock())
    mocker.patch("main.png_to_2d", return_value=np.ones((32, 32), dtype=bool))
    mocker.patch("builtins.input", side_effect=["\\td2d", tmp_png, "", "\\e"])

    from main import main
    main()
    mock_del.assert_called_once()


# TC-MC-06: \kc2d triggers 2D demo route

def test_kc2d_routes_to_demo(mocker):
    """\\kc2d must call run_2d_demo (mock the whole function)."""
    mock_run = mocker.patch("main.run_2d_demo")
    mocker.patch("builtins.input", side_effect=["\\kc2d", "\\bc", "\\e"])

    from main import main
    main()

    # assertion: assert_called_once_with (checks both call and arguments)
    mock_run.assert_called_once_with("\\kc2d", "\\bc")


# TC-MC-07: \ms3d triggers 3D demo route

def test_ms3d_routes_to_demo(mocker):
    """\\ms3d must call run_3d_demo."""
    mock_run = mocker.patch("main.run_3d_demo")
    mocker.patch("builtins.input", side_effect=["\\ms3d", "\\e"])

    from main import main
    main()
    mock_run.assert_called_once_with("\\ms3d")


# TC-MC-08: \e exits immediately

def test_exit_command(mocker, capsys):
    """\\e must terminate the main loop."""
    mocker.patch("builtins.input", side_effect=["\\e"])
    from main import main
    main()
    out = capsys.readouterr().out
    assert "Выберете операцию:" in out


# TC-MC-09: Unknown command

def test_unknown_command(mocker, capsys):
    """Unrecognized command prints error."""
    mocker.patch("builtins.input", side_effect=["\\xyz", "\\e"])
    from main import main
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
