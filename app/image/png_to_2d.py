import numpy as np
from PIL import Image


def png_to_2d(path: str,
              threshold: float = 0.5,
              invert: bool = False) -> np.ndarray:
    """
    Loads a PNG image and converts it to a 2D boolean matrix.

    Algorithm:
      1. Load the image, convert it to grayscale (0..255).
      2. Normalize the brightness to the range [0, 1].
      3. Pixels with brightness >= threshold -> True, others -> False.
      4. If invert=True, the result is inverted (dark pixels -> True).

    Parameters
    - path : str
        Path to the PNG file.
    - threshold : float
        Binarization threshold (0..1). Defaults to 0.5.
    - invert : bool
        Invert the result (useful when the fractal is drawn in a dark color on a light background).

    Returns
    - np.ndarray
        A 2D boolean matrix of shape (H, W).
    """
    img = Image.open(path).convert("L")  # gray shade
    arr = np.array(img, dtype=np.float64) / 255.0  # normalize
    binary = arr >= threshold
    if invert:
        binary = ~binary
    return binary
