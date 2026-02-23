import numpy as np
from PIL import Image


def png_to_3d(path: str,
              max_depth: int = 64,
              threshold: float = 0.01) -> np.ndarray:
    """
    Loads a PNG image and builds a 3D boolean voxel matrix.

    Algorithm:
      The intensity of each pixel (0..1) is interpreted as height.
      A voxel (r, c, z) is considered occupied if z < intensity(r, c) * max_depth.
      The result is a "relief" projected along the Z axis.

    Parameters
    - path : str
        Path to the PNG file.
    - max_depth : int
        Number of layers along the Z axis.
    - threshold : float
        Minimum intensity below which a pixel is considered empty.

    Returns
    - np.ndarray
        Boolean matrix of shape (H, W, max_depth).
    """
    img = Image.open(path).convert("L")
    gray = np.array(img, dtype=np.float64) / 255.0

    # Voxels are occupied if their z-coordinate is below their height
    heights = (gray * max_depth).astype(int)
    heights[gray < threshold] = 0

    z_levels = np.arange(max_depth)[None, None, :]  # (1, 1, D)
    arr_3d = z_levels < heights[:, :, None]  # (H, W, D)

    return arr_3d
