"""Native-grid line geometry selects the intended half-plane."""

import numpy as np

from scripts.manual_masking.annotation_geometry import line_side_mask


def test_vertical_line_keeps_its_valid_side():
    rows, columns = np.indices((46, 46), dtype=float)
    line = dict(points=[[25, 3], [25, 42]], valid_side=1)
    result = line_side_mask(line, (46, 46))
    np.testing.assert_array_equal(result, columns <= 25)
    assert result[0, 0] and result[45, 0]
    line["valid_side"] = -1
    np.testing.assert_array_equal(line_side_mask(line, (46, 46)), columns >= 25)


def test_two_lines_can_trim_the_native_grid():
    rows, columns = np.indices((46, 46), dtype=float)
    result = line_side_mask(dict(points=[[25, 3], [25, 42]], valid_side=1), (46, 46))
    result &= line_side_mask(dict(points=[[0, 10], [45, 10]], valid_side=1), (46, 46))
    np.testing.assert_array_equal(result, (columns <= 25) & (rows >= 10))
    assert result[45, 0]


