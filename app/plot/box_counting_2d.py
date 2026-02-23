import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


_BG = "#0d0d0d"
_FG = "white"


def _pick_indices(n: int, target: int = 3) -> list:
    """Uniformly picks target indices from the range [0, n)."""
    if n <= target:
        return list(range(n))
    step = (n - 1) / (target - 1)
    return [int(round(i * step)) for i in range(target)]


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
