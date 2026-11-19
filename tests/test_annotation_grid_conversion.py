"""Native-resolution annotation transfer checks."""

import numpy as np
from netCDF4 import Dataset
from secsy import cubedsphere

from scripts.manual_masking.annotation_store import decode_mask, encode_mask
from scripts.manual_masking.convert_annotation_grid import convert_document


def grid_coordinates(axis):
    projection = cubedsphere.CSprojection((0, 90), (0, 1))
    xi, eta = np.meshgrid(axis, axis)
    longitude, latitude = projection.cube2geo(xi, eta)
    return xi, eta, latitude, longitude / 15


def write_target_grid(path):
    axis = np.asarray([-0.375, -0.125, 0.125, 0.375])
    xi, eta, latitude, mlt = grid_coordinates(axis)
    with Dataset(path, "w") as product:
        grid = product.createGroup("grid")
        grid.createDimension("row", 4)
        grid.createDimension("column", 4)
        grid.position = np.asarray([0.0, 90.0])
        grid.orientation = np.asarray([0.0, 1.0])
        grid.radius_metres = 6501200.0
        grid.grid_id = "synthetic_4x4"
        for name, values in {
            "xi": xi, "eta": eta, "mlat": latitude, "mlt": mlt
        }.items():
            variable = grid.createVariable(name, "f8", ("row", "column"))
            variable[:] = values


def source_document():
    _, _, latitude, mlt = grid_coordinates(np.asarray([-0.25, 0.25]))
    footprint = np.ones((2, 2), dtype=bool)
    mask = np.asarray([[True, False], [True, False]])
    line = {"points": [[0.1, 0.0], [0.1, 1.0]], "valid_side": 1}
    return {
        "schema_version": 3,
        "mask_shape": [2, 2],
        "target_grid": {"mlat": latitude.tolist(), "mlt": mlt.tolist()},
        "target_grid_radius_metres": 6501200.0,
        "target_projection": {"position": [0.0, 90.0], "orientation": [0.0, 1.0]},
        "orbits": {
            "0001": {
                "mask_shape": [2, 2],
                "target_annotated_runs": encode_mask(footprint),
                "keyframes": {
                    "0": {
                        "valid_runs": encode_mask(mask),
                        "valid_fraction": float(mask.mean()),
                        "geometry": {"kind": "line_set", "lines": [line]},
                    }
                },
            }
        },
    }


def test_line_geometry_is_rasterized_on_target_grid(tmp_path):
    grid_file = tmp_path / "target.nc"
    write_target_grid(grid_file)
    converted = convert_document(source_document(), grid_file)
    payload = converted["orbits"]["0001"]["keyframes"]["0"]
    mask = decode_mask(payload["valid_runs"], (4, 4))
    expected = np.zeros((4, 4), dtype=bool)
    expected[:, 0] = True
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_allclose(
        payload["geometry"]["lines"][0]["points"],
        [[0.7, 0.5], [0.7, 2.5]], rtol=0, atol=1e-12
    )
    assert converted["grid_transfer"]["line_geometry_rasterized_keyframes"] == 1
    assert converted["grid_transfer"]["cells_changed_by_target_line_rasterization"] == 4


def test_source_cell_policy_remains_available(tmp_path):
    grid_file = tmp_path / "target.nc"
    write_target_grid(grid_file)
    converted = convert_document(
        source_document(), grid_file, rasterize_line_geometry=False
    )
    payload = converted["orbits"]["0001"]["keyframes"]["0"]
    mask = decode_mask(payload["valid_runs"], (4, 4))
    expected = np.zeros((4, 4), dtype=bool)
    expected[:, :2] = True
    np.testing.assert_array_equal(mask, expected)
    assert converted["grid_transfer"]["line_geometry_rasterized_keyframes"] == 0
