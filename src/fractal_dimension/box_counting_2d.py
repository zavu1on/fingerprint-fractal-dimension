import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _divisor_scales(shape: tuple, min_box: int = 1, max_box: int = None) -> np.ndarray:
    """
    Compute a list of divisor scales for a given shape.

    Parameters
    - shape : tuple
        Shape of the array (height, width).
    - min_box : int, optional
        Minimum box size (default is 1).
    - max_box : int, optional
        Maximum box size (default is None, which means the minimum side of the array).

    Returns
    - np.ndarray
        Array of divisor scales.
    """
    side = min(shape)
    if max_box is None:
        max_box = side

    # Get all divisors of the shape
    divs = [d for d in range(min_box, max_box + 1)
            if all(s % d == 0 for s in shape)]

    return np.array(divs, dtype=int) if divs else np.array([], dtype=int)


def box_counting_2d(arr: np.ndarray,
                    min_box: int = 1,
                    max_box: int = None,
                    num_scales: int = 20,
                    use_divisors: bool = True) -> dict:
    """
    Estimate the fractal (box-counting) dimension of a 2D set using the box-counting method.

    The method covers the binary image with square boxes of side length eps and counts
    N(eps): the number of boxes that contain at least one occupied pixel. The fractal
    dimension D is estimated as the slope of a linear regression in log-log coordinates:

        log N(eps) = D * log(1/eps) + b

    This implementation performs an explicit scan over all eps*eps boxes for clarity and
    uses `scipy.stats.linregress` for the regression.

    Parameters:
    - arr : np.ndarray
        2D array interpreted as a binary mask; nonzero/True pixels are treated as occupied.
    - min_box : int, optional
        Minimum box size eps in pixels (default: 1).
    - max_box : int | None, optional
        Maximum box size eps in pixels. If None, defaults to half of the smaller image side.
    - num_scales : int, optional
        Number of scales in the geometric grid when it is used (default: 20).
    - use_divisors : bool, optional
        If True, prefer box sizes that evenly divide both image dimensions and optionally
        augment with a geometric grid. If False, use only the geometric grid.

    Returns
    - dict
        A dictionary with:
        - 'dimension'  : estimated fractal dimension (same as 'slope')
        - 'epsilons'   : array of box sizes eps (in pixels)
        - 'counts'     : array of N(eps) values for each eps
        - 'slope'      : regression slope D
        - 'intercept'  : regression intercept b
        - 'r_squared'  : coefficient of determination (R^2)
    """

    arr = arr.astype(bool)
    H, W = arr.shape
    side = min(H, W)

    if max_box is None:
        max_box = side // 2

    if use_divisors:
        epsilons = _divisor_scales((H, W), min_box, max_box)
        if len(epsilons) < 4:
            # Geometric sequence of logarithmic steps
            geo = np.unique(np.geomspace(max(min_box, 1), max_box,
                                         num=num_scales).astype(int))
            epsilons = np.unique(np.concatenate([epsilons, geo]))
    else:
        epsilons = np.unique(
            np.geomspace(min_box, max_box, num=num_scales).astype(int)
        )

    counts = []
    for eps in epsilons:
        count = 0
        # Iterate over all eps*eps squares covering the array
        for r0 in range(0, H, eps):
            for c0 in range(0, W, eps):
                # Crop square area
                box = arr[r0: min(r0 + eps, H),
                          c0: min(c0 + eps, W)]
                # If any pixel is occupied in the box, it counts
                if box.any():
                    count += 1
        counts.append(count)

    counts = np.array(counts, dtype=np.float64)

    # Perform linear regression
    log_inv_eps = np.log(1.0 / epsilons.astype(np.float64))
    log_counts = np.log(counts)
    slope, intercept, r_value, _, _ = linregress(log_inv_eps, log_counts)

    return {
        "dimension":  slope,
        "epsilons":   epsilons,
        "counts":     counts,
        "slope":      slope,
        "intercept":  intercept,
        "r_squared":  r_value ** 2,
    }


_BG = "#0d0d0d"
_FG = "white"


def _pick_indices(n: int, target: int = 3) -> list:
    """Uniformly picks target indices from the range [0, n)."""
    if n <= target:
        return list(range(n))
    step = (n - 1) / (target - 1)
    return [int(round(i * step)) for i in range(target)]
    # """Last target indices."""
    # return list(range(n - target, n))[::-1]


def visualize_box_counting_2d(arr: np.ndarray,
                              result: dict,
                              scale_indices: list = None,
                              title: str = "") -> plt.Figure:
    """
    Visualization of the box counting method.

    Several panels side by side: on each - a fractal with an overlaying grid
    of one scale eps. Partially occupied squares are highlighted in red,
    fully occupied - in blue. Under each panel - the total number of occupied squares.

    Parameters:
    - arr : np.ndarray          2D-boolean array.
    - result : dict             Output of box_counting_2d.
    - scale_indices : list|None Indices of scales to plot
                              (by default - 3 evenly spaced).
    - title : str               Overall title of the figure.

    Returns:
    - plt.Figure
    """
    epsilons = result["epsilons"]

    if scale_indices is None:
        # Choose 3 evenly spaced indices
        scale_indices = _pick_indices(len(epsilons), 3)

    n_panels = len(scale_indices)
    H, W = arr.shape

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.5))
    fig.patch.set_facecolor(_BG)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        title or "Box Counting 2d",
        color=_FG, fontsize=15, y=0.98,
    )

    for panel, si in enumerate(scale_indices):
        ax = axes[panel]
        ax.set_facecolor(_BG)
        eps = int(epsilons[si])

        ax.imshow(arr, cmap="gray", origin="upper",
                  extent=[0, W, 0, H], interpolation="nearest")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Box size: {eps}*{eps}", color=_FG, fontsize=11)

        count = 0
        for ry in range(0, H, eps):
            for cx in range(0, W, eps):
                r_end = min(ry + eps, H)
                c_end = min(cx + eps, W)
                box = arr[ry:r_end, cx:c_end]

                has_any = box.any()
                has_all = box.all()

                if has_any:
                    count += 1

                    if not has_all:
                        fc, ec = "#ff1744", "blue"  # partially occupied
                    else:
                        fc, ec = "#2196f3", "#1565c0"  # no boxes

                    rect = patches.Rectangle(
                        (cx, H - r_end),
                        width=c_end - cx, height=r_end - ry,
                        linewidth=0.6,
                        edgecolor=ec,
                        facecolor=fc,
                        alpha=0.30,
                    )
                    ax.add_patch(rect)

        ax.text(
            0.5, 0.03,
            f"Box count: {count}",
            transform=ax.transAxes, ha="center",
            color="blue", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7,
                      edgecolor="none", boxstyle="round,pad=0.3"),
        )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    return fig
