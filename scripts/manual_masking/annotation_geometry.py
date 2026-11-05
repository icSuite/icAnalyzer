"""Evaluate annotation geometry on the native 46x46 IMAGE grid."""

import numpy as np


def line_side_mask(line, shape):
    """Return the selected side of an infinite line in native pixel coordinates."""
    rows, columns = np.indices(shape, dtype=float)
    points = line["points"]
    start, stop = np.asarray(points, dtype=float)
    dx, dy = stop - start
    if dx == 0 and dy == 0:
        raise ValueError("Line endpoints must differ")
    signed_side = dx * (rows - start[1]) - dy * (columns - start[0])
    return int(line.get("valid_side", 1)) * signed_side >= 0
