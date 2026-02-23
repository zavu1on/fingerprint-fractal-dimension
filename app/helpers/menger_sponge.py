import numpy as np
import matplotlib.pyplot as plt


def generate_menger_sponge(depth: int = 4) -> np.ndarray:
    """
    Generates a 3d boolean array of shape (3**depth,)*3 representing the Menger sponge.

    The Menger sponge is a fractal curve constructed by recursively removing cells
    based on the following rule: a cell is removed if at any base-3 digit level,
    at least two of the three coordinates have digit 1 simultaneously.
    https://mathworld.wolfram.com/MengerSponge.html

    Analytical dimension: log(20)/log(3) = 2.7268

    Parameters:
    - depth (int): The number of iterations to perform. Defaults to 3.

    Returns:
    - arr (np.ndarray): A 3d boolean array representing the Menger sponge.
    """
    size = 3 ** depth
    idx = np.arange(size, dtype=np.int64)
    xx, yy, zz = np.meshgrid(idx, idx, idx, indexing="ij")
    arr = np.ones((size, size, size), dtype=bool)

    for _ in range(depth):
        # Calculate base-3 coordinates
        dx, dy, dz = xx % 3, yy % 3, zz % 3
        # Cells are removed if at least two of the three coordinates have digit 1
        removed = ((dx == 1) & (dy == 1)) | \
                  ((dy == 1) & (dz == 1)) | \
                  ((dx == 1) & (dz == 1))
        arr &= ~removed
        # Move to the next base-3 level
        xx //= 3
        yy //= 3
        zz //= 3

    return arr


def visualize_menger_sponge(arr: np.ndarray, depth: int = None) -> plt.Figure:
    """
    Voxel plot of the Menger sponge.
    For depth > 2 downsamples for rendering speed.

    Parameters:
    - arr (np.ndarray): A 3d boolean array representing the Menger sponge.
    - depth (int): The number of iterations to perform. If None, defaults to int(np.log(size) / np.log(3) + 0.5).

    Returns:
    - fig (plt.Figure): The generated figure.
    """
    size = arr.shape[0]
    if depth is None:
        depth = int(np.log(size) / np.log(3) + 0.5)

    # Downsample if too large for voxel rendering
    render = arr
    if size > 27:
        factor = size // 27
        s = 27
        render = arr[::factor, ::factor, ::factor][:s, :s, :s]

    fig = plt.figure(figsize=(9, 9))
    fig.patch.set_facecolor("#0d0d0d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d0d0d")

    # Color voxels based on presence in the Menger sponge
    colors = np.empty(render.shape + (4,))
    colors[render] = [0.2, 0.6, 1.0, 0.85]
    colors[~render] = [0, 0, 0, 0]

    ax.voxels(render, facecolors=colors, edgecolor=[0.1, 0.3, 0.5, 0.3],
              linewidth=0.3)
    ax.set_title(f"Menger Sponge  (depth={depth})\n"
                 f"Analytical dim = log(20)/log(3) = 2.7268",
                 color="white", pad=10)
    ax.tick_params(colors="white")

    # Hide axis panes for a cleaner visual
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
    ax.grid(False)
    return fig
