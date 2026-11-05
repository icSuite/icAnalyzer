"""Evaluate transferred line sides across the full native IMAGE grid."""

#%% Source coordinates preserve the original line geometry after conversion

import numpy as np
from matplotlib.path import Path as PolygonPath
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from secsy import CSgrid, CSprojection

from scripts.manual_masking.annotation_store import decode_mask


def source_pixel_coordinates(record, target_grid_file):
    source = record.get("legacy_annotation", record)
    with Dataset(source["source_file"]) as nc:
        grid = nc.groups["grid"]
        old = CSgrid(CSprojection(grid.position, grid.orientation), L=float(grid.L), W=float(grid.W), Lres=float(grid.Lres), Wres=float(grid.Wres), R=float(grid.R))
    with Dataset(target_grid_file) as nc:
        grid = nc.groups["grid"]
        projection = CSprojection(grid.position, grid.orientation)
        lon, lat = projection.cube2geo(np.asarray(grid.variables["xi"][:]), np.asarray(grid.variables["eta"][:]))
    xi, eta = old.projection.geo2cube(lon, lat)
    columns = interp1d(old.xi[0], np.arange(old.shape[1]), fill_value="extrapolate")(xi)
    rows = interp1d(old.eta[:, 0], np.arange(old.shape[0]), fill_value="extrapolate")(eta)
    if not np.isfinite(columns).all() or not np.isfinite(rows).all():
        raise ValueError("Native grid cannot be mapped to the original line coordinates")
    return columns, rows


def line_side_mask(line, shape, source_coordinates=None):
    """An infinite half-plane; native lines use native pixels, converted lines their source."""
    rows, columns = np.indices(shape, dtype=float)
    points = line["points"]
    if "source_points" in line:
        if source_coordinates is None:
            raise ValueError("Converted line requires original source-grid coordinates")
        columns, rows = source_coordinates
        points = line["source_points"]
    start, stop = np.asarray(points, dtype=float)
    dx, dy = stop - start
    if dx == 0 and dy == 0:
        raise ValueError("Line endpoints must differ")
    signed_side = dx * (rows - start[1]) - dy * (columns - start[0])
    return int(line.get("valid_side", 1)) * signed_side >= 0


def full_grid_keyframe_mask(payload, previous_footprint, source_coordinates=None):
    """Transfer geometry without adding a restriction at the original grid edge."""
    shape = previous_footprint.shape
    geometry = payload.get("geometry", {})
    if geometry.get("kind") in ("line", "line_set"):
        lines = [geometry] if geometry["kind"] == "line" else geometry["lines"]
        mask = np.ones(shape, dtype=bool)
        for line in lines:
            mask &= line_side_mask(line, shape, source_coordinates)
        return mask
    vertices = payload.get("source_vertices")
    if vertices is not None:
        if source_coordinates is None:
            raise ValueError("Converted lasso requires original source-grid coordinates")
        columns, rows = source_coordinates
    else:
        vertices = payload.get("vertices", geometry.get("vertices"))
        rows, columns = np.indices(shape, dtype=float)
    if vertices is not None:
        cells = np.column_stack((columns.ravel(), rows.ravel()))
        return PolygonPath(vertices, closed=True).contains_points(cells, radius=0.15).reshape(shape)
    mask = decode_mask(payload["valid_runs"], shape=shape)
    if not mask.any():
        return mask
    if np.array_equal(mask, previous_footprint):
        return np.ones(shape, dtype=bool)
    raise ValueError("A nonconstant raster mask needs its original drawn geometry to extend it")
