import numpy as np
from scipy.stats import linregress
from matplotlib import pyplot as plt

from .box_counting_2d import _divisor_scales, _pick_indices, _BG, _FG


def box_counting_3d(arr: np.ndarray,
                    min_box: int = 1,
                    max_box: int = None,
                    num_scales: int = 12,
                    use_divisors: bool = True) -> dict:
    """
    Estimate the fractal dimension of a 3D set using the box-counting method.

    Algorithm:
      For each box size eps, cover the space with cubes of size eps*eps*eps and
      count how many cubes contain at least one occupied voxel. D - the slope
      of the line log N(eps) vs log(1/eps).

    Parameters
    - arr : np.ndarray
        3D boolean array (True = occupied).
    - min_box : int
        Minimum box size (in voxels).
    - max_box : int | None
        Maximum box size.
    - num_scales : int
        Number of scales (used when use_divisors=False).
    - use_divisors : bool
        True -> use only box sizes that divide all sides of the shape.

    Returns
    - dict  – same as the output of box_counting_2d.
    """
    arr = arr.astype(bool)
    D1, D2, D3 = arr.shape
    side = min(D1, D2, D3)

    if max_box is None:
        max_box = side // 2

    if use_divisors:
        epsilons = _divisor_scales((D1, D2, D3), min_box, max_box)
        if len(epsilons) < 4:
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
        for x0 in range(0, D1, eps):
            for y0 in range(0, D2, eps):
                for z0 in range(0, D3, eps):
                    cube = arr[x0: min(x0 + eps, D1),
                               y0: min(y0 + eps, D2),
                               z0: min(z0 + eps, D3)]
                    if cube.any():
                        count += 1
        counts.append(count)

    counts = np.array(counts, dtype=np.float64)

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


def visualize_box_counting_3d(arr: np.ndarray,
                              result: dict,
                              scale_indices: list = None,
                              max_cubes_draw: int = 4000,
                              title: str = "") -> plt.Figure:
    """
    Visualization of the box counting method.

    Several 3D panels side by side: on each - a voxel object,
    covered with semi-transparent cubes eps*eps*eps. The color of the cube
    encodes its height along the Z axis (coolwarm). Under each panel -
    the number of found cubes.

    Parameters:
    - arr : np.ndarray          3D boolean array.
    - result : dict             Output of box_counting_3d.
    - scale_indices : list|None Indices of scales to plot.
    - max_cubes_draw : int      Limit of drawn cubes (for responsiveness).
    - title : str               Overall title of the figure.

    Returns:
    - plt.Figure
    """
    epsilons = result["epsilons"]

    if scale_indices is None:
        scale_indices = _pick_indices(len(epsilons), 3)

    n_panels = len(scale_indices)
    D1, D2, D3 = arr.shape
    cmap = plt.cm.coolwarm

    fig = plt.figure(figsize=(6.5 * n_panels, 7))
    fig.patch.set_facecolor(_BG)
    fig.suptitle(
        title or "Box Counting 3D",
        color=_FG, fontsize=15, y=0.98,
    )

    for panel, si in enumerate(scale_indices):
        ax = fig.add_subplot(1, n_panels, panel + 1, projection="3d")
        ax.set_facecolor(_BG)
        eps = int(epsilons[si])

        cubes = []
        for x0 in range(0, D1, eps):
            for y0 in range(0, D2, eps):
                for z0 in range(0, D3, eps):
                    blk = arr[x0: min(x0 + eps, D1),
                              y0: min(y0 + eps, D2),
                              z0: min(z0 + eps, D3)]
                    if blk.any():
                        cubes.append((x0, y0, z0))

        count = len(cubes)

        if count > max_cubes_draw:
            rng = np.random.default_rng(42)
            idx = rng.choice(count, max_cubes_draw, replace=False)
            cubes_draw = [cubes[i] for i in idx]
        else:
            cubes_draw = cubes

        z_max = max(D1, D2, D3) or 1
        for (x0, y0, z0) in cubes_draw:
            c = cmap(z0 / z_max)
            sx = min(eps, D1 - x0)
            sy = min(eps, D2 - y0)
            sz = min(eps, D3 - z0)
            ax.bar3d(x0, y0, z0, sx, sy, sz,
                     color=c, alpha=0.7, shade=True,
                     edgecolor="black", linewidth=0.5)

        ax.set_xlabel("X", color=_FG, fontsize=9)
        ax.set_ylabel("Y", color=_FG, fontsize=9)
        ax.set_zlabel("Z",  color=_FG, fontsize=9)
        ax.tick_params(colors=_FG, labelsize=7)
        ax.set_xlim(0, D1)
        ax.set_ylim(0, D2)
        ax.set_zlim(0, D3)

        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
        ax.grid(False)

        ax.set_title(
            f"Cube size: {eps}*{eps}*{eps}\n"
            f"Cube count: {count}",
            color=_FG, fontsize=11, pad=8,
        )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    return fig
