import numpy as np
from scipy.spatial import Delaunay, QhullError
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from .box_counting_2d import _BG, _FG, _pick_indices


def delaunay_dimension(arr_or_pts: np.ndarray,
                       num_scales: int = 12,
                       min_frac: float = 0.003,
                       max_frac: float = 0.20,
                       edge_factor: float = 3.0) -> dict:
    """
    Estimate the fractal dimension of a 2D set using the Delaunay triangulation method.

    Algorithm:
      1. From a binary matrix, extract coordinates of occupied pixels
         (or take a ready array of points (N, 2)).
      2. For each scale delta, perform a sampling: the space is divided into delta*delta
         squares, and one point-representative is taken from each occupied square.
      3. Build a Delaunay triangulation from the sampling (scipy.spatial.Delaunay).
      4. Select only "local" triangles, whose maximum edge length is < edge_factor * delta.
      5. Count the number of remaining triangles N(delta) ~ delta^(-D).
      6. Perform a linear regression in log-log coordinates: log N vs log(1/delta) -> D.

    Parameters
    ----------
    arr_or_pts : np.ndarray
        2D binary matrix *or* array of points shape (N, 2).
    num_scales : int
        Number of scales.
    min_frac, max_frac : float
        Range of scales as a fraction of the diagonal of the bounding box.
    edge_factor : float
        Filtering factor for local triangles: max edge length < edge_factor * delta.

    Returns
    -------
    dict
        'dimension'      – estimated fractal dimension D
        'deltas'         – array of scales delta
        'num_triangles'  – array of N(delta) values for each delta
        'slope',         - slope of the linear regression
        'intercept',     - intercept of the linear regression
        'r_squared'      - coefficient of determination
        'triangulations' – list of Delaunay objects (for visualization)
        'subsampled_pts' – list of arrays of sampled points
        'kept_masks'     – masks of "local" triangles
    """
    if arr_or_pts.ndim == 2 and arr_or_pts.shape[1] == 2 and arr_or_pts.dtype.kind == "f":
        # Already an array of points (N, 2)
        pts = arr_or_pts.astype(np.float64)
    else:
        # Coordinates of occupied pixels
        rows, cols = np.where(arr_or_pts.astype(bool))
        pts = np.column_stack([cols.astype(np.float64),
                               rows.astype(np.float64)])

    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    diag = np.linalg.norm(pmax - pmin)

    deltas = np.geomspace(min_frac * diag, max_frac * diag, num=num_scales)

    num_triangles = []
    triangulations = []
    subsampled_list = []
    kept_masks = []

    for delta in deltas:
        cell_ids = ((pts - pmin) / delta).astype(np.int64)
        cells = {}
        for idx in range(len(cell_ids)):
            key = (cell_ids[idx, 0], cell_ids[idx, 1])
            if key not in cells:
                cells[key] = idx
        sub_pts = pts[list(cells.values())]

        if len(sub_pts) < 3:
            continue

        try:
            tri = Delaunay(sub_pts)
        except QhullError:
            # degenerate case, cannot build triangulation
            continue

        simplices = tri.simplices  # (M, 3)
        verts = sub_pts[simplices]  # (M, 3, 2)

        # Estimate ribs
        e0 = np.linalg.norm(verts[:, 1] - verts[:, 0], axis=1)
        e1 = np.linalg.norm(verts[:, 2] - verts[:, 1], axis=1)
        e2 = np.linalg.norm(verts[:, 0] - verts[:, 2], axis=1)
        max_edge = np.maximum(e0, np.maximum(e1, e2))
        mask = max_edge < edge_factor * delta

        n_kept = mask.sum()
        if n_kept == 0:
            continue

        num_triangles.append(n_kept)
        triangulations.append(tri)
        subsampled_list.append(sub_pts)
        kept_masks.append(mask)

    deltas = deltas[: len(num_triangles)]
    num_triangles = np.array(num_triangles, dtype=np.float64)

    if len(num_triangles) < 2:
        return {
            "dimension": 0.0,
            "deltas": deltas,
            "num_triangles": num_triangles,
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "triangulations": triangulations,
            "subsampled_pts": subsampled_list,
            "kept_masks": kept_masks,
        }

    log_inv_d = np.log(1.0 / deltas)
    log_nt = np.log(num_triangles)
    slope, intercept, r_value, _, _ = linregress(log_inv_d, log_nt)

    return {
        "dimension":       slope,
        "deltas":          deltas,
        "num_triangles":   num_triangles,
        "slope":           slope,
        "intercept":       intercept,
        "r_squared":       r_value ** 2,
        "triangulations":  triangulations,
        "subsampled_pts":  subsampled_list,
        "kept_masks":      kept_masks,
    }


def visualize_delaunay(
        arr_or_pts: np.ndarray,
        result: dict,
        scale_indices: list = None,
        title: str = "") -> plt.Figure:
    """
    Visualization of the Delaunay triangulation method.

    Several panels side by side: on each - a subset of points with an
    overlaying Delaunay triangulation. "Local" triangles (edge < edge_factor *
    delta) are colored according to their area, other triangles are shown by
    thin gray lines. Under each panel - statistics (points/triangles/area).

    Parameters:
    - arr_or_pts : np.ndarray   Boolean array or points array (N,2).
    - result : dict             Output of delaunay_dimension.
    - scale_indices : list|None Indices of scales to plot.
    - title : str               Overall title of the figure.

    Returns:
    - plt.Figure
    """
    deltas = result["deltas"]
    num_tri = result["num_triangles"]

    if scale_indices is None:
        scale_indices = _pick_indices(len(deltas), 3)

    n_panels = len(scale_indices)
    cmap = plt.cm.coolwarm

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 6))
    fig.patch.set_facecolor(_BG)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        title or "Delaunay Triangulation",
        color=_FG, fontsize=15, y=0.98,
    )

    all_pts_list = result["subsampled_pts"]
    global_xmin = min(p[:, 0].min() for p in all_pts_list)
    global_xmax = max(p[:, 0].max() for p in all_pts_list)
    global_ymin = min(p[:, 1].min() for p in all_pts_list)
    global_ymax = max(p[:, 1].max() for p in all_pts_list)
    margin = max(global_xmax - global_xmin, global_ymax - global_ymin) * 0.02

    for panel, si in enumerate(scale_indices):
        ax = axes[panel]
        ax.set_facecolor(_BG)

        tri = result["triangulations"][si]
        sub_pts = result["subsampled_pts"][si]
        mask = result["kept_masks"][si]
        delta = deltas[si]
        simplices = tri.simplices
        all_verts = sub_pts[simplices]

        # Not considered triangles
        rejected = all_verts[~mask]
        if len(rejected):
            pc_rej = PolyCollection(rejected, closed=True,
                                    facecolors="none",
                                    edgecolors="#2a2a2a",
                                    linewidths=0.15)
            ax.add_collection(pc_rej)

        # Considered triangles
        kept = all_verts[mask]
        total_area = 0.0
        if len(kept):
            # Cross product
            areas = 0.5 * np.abs(
                (kept[:, 1, 0] - kept[:, 0, 0]) *
                (kept[:, 2, 1] - kept[:, 0, 1]) -
                (kept[:, 2, 0] - kept[:, 0, 0]) *
                (kept[:, 1, 1] - kept[:, 0, 1])
            )
            total_area = areas.sum()

            # Color by area
            norm = plt.Normalize(areas.min(), areas.max() + 1e-12)
            face_colors = cmap(norm(areas))
            face_colors[:, 3] = 0.55

            pc_ok = PolyCollection(kept, closed=True,
                                   facecolors=face_colors,
                                   edgecolors="black",
                                   linewidths=0.4)
            ax.add_collection(pc_ok)

        ax.scatter(sub_pts[:, 0], sub_pts[:, 1],
                   s=0.5, color="#ffeb3b", alpha=0.6, zorder=3)

        ax.set_xlim(global_xmin - margin, global_xmax + margin)
        ax.set_ylim(global_ymin - margin, global_ymax + margin)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Масштаб delta = {delta:.4g}", color=_FG, fontsize=11)

        ax.text(
            0.5, 0.03,
            f"Dots: {len(sub_pts)}  │  "
            f"Triangles: {int(num_tri[si])}  │  "
            f"Square: {total_area:.1f}",
            transform=ax.transAxes, ha="center",
            color="blue", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7,
                      edgecolor="none", boxstyle="round,pad=0.3"),
        )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    return fig
