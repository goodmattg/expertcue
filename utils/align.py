import numpy as np
import pdb

from typing import List, Tuple


def interpolate_fill(
    boundaries: List[Tuple[int, int, List[int]]], array: np.ndarray
) -> np.ndarray:
    """
    Fill duplicate duplicate indices with interpolated data

    Input:
        indices: index array
        array: [N,...]
    """
    for s, e, idx in boundaries:
        array[idx] = interpolate_linear(array[s], array[e], n=len(idx))

    return array


def interpolate_linear(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    return np.stack(
        [np.linspace(xr, yr, n + 1, endpoint=False)[1:] for xr, yr in zip(x, y)], axis=1
    )


def run_boundaries(
    run_starts: np.ndarray, run_lengths: np.ndarray
) -> List[Tuple[int, int, List[int]]]:
    """Returns run boundaries (left bound, right_bound, in_run)"""

    assert len(run_starts) == len(run_lengths)
    # Ignore the last index start b/c no right boundary)
    boundaries = []
    for ix in range(len(run_starts) - 1):
        s, l = run_starts[ix], run_lengths[ix]
        if l > 1:
            boundaries.append(
                (s, run_starts[ix + 1], list(range(s + 1, run_starts[ix + 1])))
            )

    return boundaries


def find_runs(x: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find runs of consecutive items in an array.
    https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths
