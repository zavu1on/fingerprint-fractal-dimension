import numpy as np
import matplotlib.pyplot as plt


def generate_sierpinski_triangle(depth: int = 7) -> np.ndarray:
    """
    Generates a Sierpinski triangle using Pascal's triangle / XOR method.

    The triangle is represented by a 2d boolean array of shape (2**depth, 2**depth).
    A cell (r, c) is part of the triangle if and only if (r & c) == 0 in the
    complement.
    https://mathworld.wolfram.com/SierpinskiSieve.html

    Analytical dimension: log(3)/log(2) = 1.585

    Parameters:
    - depth (int): The number of iterations to perform.

    Returns:
    - arr (np.ndarray): A 2d boolean array representing the Sierpinski triangle.
    """
    size = 2 ** depth

    # Each element is a element-vector
    r = np.arange(size, dtype=np.int64)[:, None]
    c = np.arange(size, dtype=np.int64)[None, :]

    # Cell is IN the triangle if bitwise AND of row & col is zero
    arr = (r & c) == 0

    return arr


def visualize_sierpinski_triangle(arr: np.ndarray) -> plt.Figure:
    """
    Visualizes a Sierpinski triangle by plotting a 2D boolean array as an image.

    Parameters:
    - arr (np.ndarray): A 2d boolean array representing the Sierpinski triangle.

    Returns:
    - fig (plt.Figure): The generated figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr, cmap="inferno", origin="lower", interpolation="nearest")

    n = arr.shape[0]
    depth = int(np.log2(n))

    ax.set_title(f"Sierpinski Triangle  (depth={depth})\n"
                 f"Analytical dim = log(3)/log(2) = 1.585", color="white", pad=10)
    ax.axis("off")

    fig.patch.set_facecolor("#0d0d0d")

    return fig
