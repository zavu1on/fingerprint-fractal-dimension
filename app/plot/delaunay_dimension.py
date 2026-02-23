import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from .box_counting_2d import _BG, _FG, _pick_indices


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
