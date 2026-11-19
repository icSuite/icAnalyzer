"""Transfer annotations between aligned CS grids.

Editable line geometry can be rasterized on the target grid so a boundary
drawn at sub-pixel coordinates retains its intended location. Keyframes without
editable geometry inherit the source cell's Boolean decision.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
from secsy import cubedsphere

from scripts.manual_masking.annotation_store import decode_mask, encode_mask, utc_now
from scripts.manual_masking.annotation_geometry import line_side_mask


def convert_document(document, grid_file, rasterize_line_geometry=True):
    """Return a spatially transferred copy, preserving all temporal records."""

    old_shape = tuple(document["mask_shape"])
    projection = document["target_projection"]
    project = cubedsphere.CSprojection(projection["position"], projection["orientation"])
    old_x, old_y = project.geo2cube(
        15 * np.asarray(document["target_grid"]["mlt"]),
        np.asarray(document["target_grid"]["mlat"]),
    )
    with Dataset(grid_file) as product:
        grid = product.groups["grid"]
        for name in ("position", "orientation"):
            np.testing.assert_allclose(grid.getncattr(name), projection[name], rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            grid.radius_metres, document["target_grid_radius_metres"], rtol=0, atol=1e-6
        )
        new_x, new_y = np.asarray(grid["xi"][:]), np.asarray(grid["eta"][:])
        target_grid = {name: np.asarray(grid[name][:]).tolist() for name in ("mlat", "mlt")}
        grid_id = str(grid.grid_id)

    new_shape = new_x.shape
    axes = ((old_x, old_y), (new_x, new_y))
    for x, y in axes:
        np.testing.assert_allclose(x, np.broadcast_to(x[0], x.shape), rtol=0, atol=1e-10)
        np.testing.assert_allclose(y, np.broadcast_to(y[:, :1], y.shape), rtol=0, atol=1e-10)
        for axis in (x[0], y[:, 0]):
            steps = np.diff(axis)
            if np.any(steps <= 0):
                raise ValueError("conversion requires increasing rectilinear CS axes")
            np.testing.assert_allclose(steps, steps[0], rtol=0, atol=1e-10)

    scale, offset, indices = [], [], []
    for old_axis, new_axis in ((old_x[0], new_x[0]), (old_y[:, 0], new_y[:, 0])):
        old_step, new_step = np.diff(old_axis).mean(), np.diff(new_axis).mean()
        old_edges = [old_axis[0] - old_step / 2, old_axis[-1] + old_step / 2]
        new_edges = [new_axis[0] - new_step / 2, new_axis[-1] + new_step / 2]
        np.testing.assert_allclose(new_edges, old_edges, rtol=0, atol=1e-10)
        scale.append(float(old_step / new_step))
        offset.append(float((old_axis[0] - new_axis[0]) / new_step))
        positions = np.floor((new_axis - old_edges[0]) / old_step).astype(int)
        if np.any((positions < 0) | (positions >= len(old_axis))):
            raise ValueError("target cell centres extend outside the annotated area")
        indices.append(positions)

    def transfer_runs(runs):
        original = decode_mask(runs, shape=old_shape)
        transferred = original[np.ix_(indices[1], indices[0])]
        return encode_mask(transferred), float(transferred.mean())

    def transfer_points(points):
        return (np.asarray(points, dtype=float) * scale + offset).tolist()

    result = deepcopy(document)
    keyframes = 0
    line_keyframes = 0
    geometry_changed_cells = 0
    raster_fallback_keyframes = 0
    for record in result["orbits"].values():
        record["mask_shape"] = list(new_shape)
        record["target_annotated_runs"], _ = transfer_runs(record["target_annotated_runs"])
        footprint = decode_mask(record["target_annotated_runs"], shape=new_shape)
        for payload in record["keyframes"].values():
            inherited_runs, inherited_fraction = transfer_runs(payload["valid_runs"])
            if "vertices" in payload:
                payload["vertices"] = transfer_points(payload["vertices"])
            geometry = payload.get("geometry")
            if geometry:
                kind = geometry["kind"]
                if kind == "line_set":
                    for line in geometry["lines"]:
                        line["points"] = transfer_points(line["points"])
                elif kind == "line":
                    geometry["points"] = transfer_points(geometry["points"])
                elif kind == "lasso":
                    geometry["vertices"] = transfer_points(geometry["vertices"])
                else:
                    raise ValueError(f"unsupported drawing geometry: {kind}")
            if rasterize_line_geometry and geometry and geometry["kind"] == "line_set":
                mask = footprint.copy()
                for line in geometry["lines"]:
                    mask &= line_side_mask(line, new_shape)
                inherited = decode_mask(inherited_runs, shape=new_shape)
                geometry_changed_cells += int(np.count_nonzero(mask != inherited))
                payload["valid_runs"] = encode_mask(mask)
                payload["valid_fraction"] = float(mask.mean())
                line_keyframes += 1
            else:
                payload["valid_runs"] = inherited_runs
                payload["valid_fraction"] = inherited_fraction
                raster_fallback_keyframes += 1
            keyframes += 1

    result.update(
        mask_shape=list(new_shape), target_grid=target_grid,
        annotation_context="self-contained source_times_utc per orbit and embedded target_grid",
        line_side_reference="target-grid pixel coordinates; valid_runs is authoritative",
        outside_source_grid="Frame classes and drawn geometry apply across the entire target grid",
        updated_at=utc_now(),
    )
    result["grid_transfer"] = {
        "source_mask_shape": list(old_shape), "target_mask_shape": list(new_shape),
        "target_grid_id": grid_id,
        "pixel_coordinate_scale_xy": scale, "pixel_coordinate_offset_xy": offset,
        "mask_policy": (
            "affinely transferred line_set geometry is rasterized at target cell centres; "
            "keyframes without line_set geometry inherit the source cell Boolean decision; "
            "valid_runs remains authoritative"
            if rasterize_line_geometry else
            "target cell centre inherits source cell Boolean decision; valid_runs remains authoritative"
        ),
        "drawing_policy": (
            "affine coordinate transfer; line_set geometry replaces the inherited raster at target resolution"
            if rasterize_line_geometry else
            "affine coordinate transfer; drawings do not replace stored raster masks"
        ),
        "review_policy": "boundaries and reviews are inherited from the source annotation; no new manual decisions",
        "keyframes": keyframes,
        "line_geometry_rasterized_keyframes": line_keyframes,
        "source_cell_fallback_keyframes": raster_fallback_keyframes,
        "cells_changed_by_target_line_rasterization": geometry_changed_cells,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("grid_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preserve-source-cell-masks", action="store_true",
        help="do not rerasterize transferred line geometry on the target grid"
    )
    args = parser.parse_args()
    source = args.annotations.expanduser()
    grid_file = args.grid_file.expanduser()
    output = args.output.expanduser()
    if output.exists():
        raise FileExistsError(f"choose a new annotation output: {output}")
    source_bytes = source.read_bytes()
    result = convert_document(
        json.loads(source_bytes), grid_file,
        rasterize_line_geometry=not args.preserve_source_cell_masks
    )
    result["grid_transfer"]["source_annotation_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["grid_transfer"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
