import numpy as np
import matplotlib.pyplot as plt


def generate_sierpinski_carpet(depth: int = 5) -> np.ndarray:
    """
    Generates a 2d boolean array of shape (3**depth, 3**depth).

    The Sierpinski carpet is a fractal curve constructed by recursively
    removing the middle third of each segment. Cell (r,c) is REMOVED
    if at any base-3 digit position both digits are 1.
    https://mathworld.wolfram.com/SierpinskiCarpet.html

    Analytical dimension: log(8)/log(3) = 1.8928

    Parameters:
    - depth (int): The number of iterations to perform.

    Returns:
    - arr (np.ndarray): A 2d boolean array representing the Sierpinski carpet.
    """
    size = 3 ** depth

    # Create a 2d index array for each cell
    r = np.arange(size, dtype=np.int64)[:, None]
    c = np.arange(size, dtype=np.int64)[None, :]
    arr = np.ones((size, size), dtype=bool)

    # Iteratively check each base-3 level
    for _ in range(depth):
        # Remove cells where both base-3 digits are 1
        arr &= ~((r % 3 == 1) & (c % 3 == 1))

        # Move to the next base-3 level
        r //= 3
        c //= 3

    return arr


def visualize_sierpinski_carpet(arr: np.ndarray) -> plt.Figure:
    """
    Visualize a Sierpinski carpet by plotting a 2D boolean array as an image.

    Parameters:
    - arr (np.ndarray): A 2d boolean array representing the Sierpinski carpet.

    Returns:
    - fig (plt.Figure): The generated figure.
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr, cmap="viridis", origin="lower", interpolation="nearest")

    # Compute the depth of the carpet
    n = arr.shape[0]
    depth = int(np.log(n) / np.log(3) + 0.5)

    # Set the title
    ax.set_title(f"Sierpinski Carpet  (depth={depth})\n"
                 f"Analytical dim = log(8)/log(3) = 1.8928", color="white", pad=10)
    ax.axis("off")  # Turn off axis labels

    fig.patch.set_facecolor("#0d0d0d")  # Set the background color

    return fig
