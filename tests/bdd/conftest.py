import sys
from pathlib import Path

import numpy as np
import pytest
from pytest_bdd import then, parsers
from PIL import Image

# Ensure project root is on sys.path
_here = Path(__file__).resolve()
for parent in _here.parents:
    if (parent / "app").is_dir():
        sys.path.insert(0, str(parent))
        break


# ===================== Shared fixtures =====================

@pytest.fixture
def context():
    """Shared mutable dict for passing data between BDD steps."""
    return {}


@pytest.fixture
def tmp_png(tmp_path):
    """Small white-on-black PNG for menu tests."""
    img = Image.new("L", (32, 32), 0)
    img.putpixel((16, 16), 255)
    path = tmp_path / "test.png"
    img.save(path)
    return str(path)


@pytest.fixture
def tmp_jpg(tmp_path):
    """A .jpg file (wrong extension for the app)."""
    img = Image.new("L", (32, 32), 0)
    path = tmp_path / "test.jpg"
    img.save(path)
    return str(path)


# =================== Shared Then steps ====================
# These Then steps appear in multiple feature files and MUST be
# defined once to avoid pytest-bdd step registration conflicts.

@then(parsers.parse(
    "размерность должна быть близка к {expected:f} с допуском {tol:f}"
))
def step_dim_close_to(context, expected, tol):
    assert context["result"]["dimension"] == pytest.approx(expected, abs=tol)


@then(parsers.parse("R-squared должен превышать {threshold:f}"))
def step_r_squared_exceeds(context, threshold):
    assert context["result"]["r_squared"] > threshold


@then("размерность должна равняться 0.0")
def step_dim_equals_zero(context):
    assert context["result"]["dimension"] == 0.0
