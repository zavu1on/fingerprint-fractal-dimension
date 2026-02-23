import numpy as np
import matplotlib.pyplot as plt


def _generate_koch_pts(depth: int = 5) -> np.ndarray:
    """
    Internal helper: generates ordered (N, 2) float32 vertices of the Koch curve.
    The curve starts as a single unit segment [0,0]->[1,0] and is subdivided.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    for _ in range(depth):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            p1 = a + (b - a) / 3
            p2 = a + (b - a) * 2 / 3
            mid = (p1 + p2) / 2
            perp = np.array([-(p2 - p1)[1], (p2 - p1)[0]])
            apex = mid + perp * (np.sqrt(3) / 2)
            new_pts += [p1, apex, p2, b]
        pts = np.array(new_pts, dtype=np.float32)

    return pts


def generate_koch_curve(depth: int = 5, resolution: int = 1024) -> np.ndarray:
    """
    Generates the Koch curve as a 2D boolean raster matrix.

    The Koch curve is a fractal curve constructed by recursively
    subdividing a line segment into four parts and offsetting the middle
    two parts to form an equilateral triangle bump.
    https://mathworld.wolfram.com/KochSnowflake.html

    Segments between successive vertices are rasterized with
    Bresenham-style linear interpolation so every pixel on each segment
    is marked True - no extra transformations are needed before calling
    box_counting_2d.

    Parameters:
    - depth : int
        Recursion depth (default 5). Each step multiplies the segment
        count by 4; depth=7 gives 4^7 = 16 384 segments.
    - resolution : int
        Side length of the output square matrix in pixels (default 1024).
        The curve is scaled to fill [0, resolution-1] x [0, resolution-1].

    Returns:
    - np.ndarray
        Boolean array of shape (resolution, resolution) where True pixels
        belong to the curve.  Row 0 is the *top* of the image (standard
        array / image convention used by box_counting_2d / imshow).

    Analytical fractal dimension: log(4)/log(3) = 1.2619
    """
    pts = _generate_koch_pts(depth)

    # Normalise to [0, resolution-1] pixel coordinates
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    coords = ((pts - lo) / (hi - lo) * (resolution - 1)).astype(np.int32)

    arr = np.zeros((resolution, resolution), dtype=bool)

    for i in range(len(coords) - 1):
        c0, r0 = coords[i]       # x -> column, y -> row
        c1, r1 = coords[i + 1]

        # Number of interpolation steps = Chebyshev distance (>= 1)
        steps = max(abs(c1 - c0), abs(r1 - r0), 1)
        ts = np.linspace(0.0, 1.0, steps + 1)

        rows = np.round(r0 + ts * (r1 - r0)).astype(np.int32)
        cols = np.round(c0 + ts * (c1 - c0)).astype(np.int32)

        # Clip to stay inside the matrix (shouldn't be needed, but safe)
        valid = (rows >= 0) & (rows < resolution) & \
                (cols >= 0) & (cols < resolution)
        arr[rows[valid], cols[valid]] = True

    return arr


def visualize_koch_curve(arr: np.ndarray, depth: int = None) -> plt.Figure:
    """
    Plots the Koch curve stored as a 2D boolean raster.

    The fractal is rendered as a gradient polyline by re-tracing the True
    pixels in scan order, which preserves the visual appearance of the
    original vector plot while accepting the rasterized form used by
    box_counting_2d.

    Parameters:
    - arr : np.ndarray
        2D boolean array returned by generate_koch_curve.
    - depth : int | None
        Recursion depth used to generate the curve.  When None it is
        inferred from the number of True pixels (approximate).

    Returns:
    - plt.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    # Display the boolean matrix directly - fast and exact
    H, W = arr.shape
    ax.imshow(arr, cmap="gray", origin="upper",
              extent=[0, W, 0, H], interpolation="nearest",
              aspect="equal")

    ax.axis("off")

    n_true = arr.sum()
    depth_str = f"depth={depth}" if depth is not None else f"~{n_true:,} px"
    ax.set_title(
        f"Koch Curve  ({depth_str},  {W}*{H} px)\n"
        f"Analytical dim = log(4)/log(3) = 1.2619",
        color="white", pad=10,
    )

    return fig
