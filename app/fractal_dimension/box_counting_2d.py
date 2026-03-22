import numpy as np
from scipy.stats import linregress


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

    BUG INTRODUCED: removed filtering of zero counts before log().
    When array is empty or very sparse, counts contain zeros ->
    log(0) = -inf -> linregress returns NaN slope.

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

    # Guard: need at least 2 scales with non-zero counts for regression

    # BUG
    # valid_mask = counts > 0
    # if valid_mask.sum() < 2:
    #     return {
    #         "dimension": 0.0,
    #         "epsilons": epsilons,
    #         "counts": counts,
    #         "slope": 0.0,
    #         "intercept": 0.0,
    #         "r_squared": 0.0,
    #     }

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
