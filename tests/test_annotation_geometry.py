"""Full-grid class and line transfer must not add an old-grid boundary."""

import numpy as np
import pytest

from scripts.manual_masking.annotation_geometry import full_grid_keyframe_mask, line_side_mask
from scripts.manual_masking.annotation_store import encode_mask


def test_source_line_keeps_its_valid_side_outside_original_grid():
    rows, columns = np.indices((46, 46), dtype=float)
    source_coordinates = ((columns - 3) / 1.1, (rows - 3) / 1.1)
    line = dict(points=[[25, 3], [25, 42]], source_points=[[20, 0], [20, 35]], valid_side=1)
    result = line_side_mask(line, (46, 46), source_coordinates)
    np.testing.assert_array_equal(result, columns <= 25)
    assert result[0, 0] and result[45, 0]
    line["valid_side"] = -1
    np.testing.assert_array_equal(line_side_mask(line, (46, 46), source_coordinates), columns >= 25)


def test_old_and_new_lines_can_trim_the_same_full_grid():
    rows, columns = np.indices((46, 46), dtype=float)
    converted = dict(points=[[25, 3], [25, 42]], source_points=[[20, 0], [20, 35]], valid_side=1)
    native = dict(points=[[0, 10], [45, 10]], valid_side=1)
    payload = dict(geometry=dict(kind="line_set", lines=[converted, native]))
    footprint = np.zeros((46, 46), dtype=bool)
    footprint[3:44, 3:44] = True
    result = full_grid_keyframe_mask(payload, footprint, ((columns - 3) / 1.1, (rows - 3) / 1.1))
    np.testing.assert_array_equal(result, (columns <= 25) & (rows >= 10))
    assert result[45, 0]


def test_constant_masks_expand_without_creating_an_unreviewed_border():
    footprint = np.zeros((46, 46), dtype=bool)
    footprint[3:44, 3:44] = True
    assert full_grid_keyframe_mask(dict(valid_runs=encode_mask(footprint)), footprint).all()
    assert not full_grid_keyframe_mask(dict(valid_runs=[]), footprint).any()
    mixed = footprint.copy()
    mixed[:, 24:] = False
    with pytest.raises(ValueError, match="drawn geometry"):
        full_grid_keyframe_mask(dict(valid_runs=encode_mask(mixed)), footprint)


