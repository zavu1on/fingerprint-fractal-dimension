# autopep8: off

import sys
import sys
from pathlib import Path

_here = Path(__file__).resolve()
for parent in _here.parents:
    if (parent / "app").is_dir():
        sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError(f"Can't find project root containing 'app' from {_here}")

from app.helpers.menger_sponge import generate_menger_sponge
from app.helpers.sierpinski_triangle import generate_sierpinski_triangle
from app.helpers.sierpinski_carpet import generate_sierpinski_carpet
from PIL import Image
import numpy as np
import pytest



@pytest.fixture
def sierpinski_carpet_243():
    """Sierpinski carpet 243x243 (depth=5), analytical D = 1.8928."""
    return generate_sierpinski_carpet(depth=5)


@pytest.fixture
def sierpinski_triangle_128():
    """Sierpinski triangle 128x128 (depth=7), analytical D = 1.585."""
    return generate_sierpinski_triangle(depth=7)


@pytest.fixture
def full_array_64():
    """Fully occupied 64x64 array."""
    return np.ones((64, 64), dtype=bool)


@pytest.fixture
def single_pixel_64():
    """64x64 array with a single True pixel."""
    arr = np.zeros((64, 64), dtype=bool)
    arr[32, 32] = True
    return arr


@pytest.fixture
def rectangular_array():
    """256x512 array with a diagonal line."""
    arr = np.zeros((256, 512), dtype=bool)
    for i in range(256):
        arr[i, i * 2] = True
    return arr


@pytest.fixture
def menger_sponge_81():
    """Menger sponge 81^3 (depth=4), analytical D = 2.727."""
    return generate_menger_sponge(depth=4)


@pytest.fixture
def full_3d_27():
    return np.ones((27, 27, 27), dtype=bool)


@pytest.fixture
def empty_3d_27():
    return np.zeros((27, 27, 27), dtype=bool)


@pytest.fixture
def sparse_7():
    """Sparse 7x7x7 array - few divisors in range."""
    arr = np.zeros((7, 7, 7), dtype=bool)
    arr[0, 0, 0] = True
    arr[3, 3, 3] = True
    arr[6, 6, 6] = True
    return arr


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
