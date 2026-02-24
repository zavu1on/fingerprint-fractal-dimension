import numpy as np
from scipy.stats import linregress

from .box_counting_2d import _divisor_scales


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
    - dict  - same as the output of box_counting_2d.
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

    valid_mask = counts > 0
    if valid_mask.sum() < 2:
        return {
            "dimension": 0.0,
            "epsilons": epsilons,
            "counts": counts,
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
        }

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
