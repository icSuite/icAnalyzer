"""Apply converted IMAGE annotations to either paired FUV or conductance data."""

#%% Read reviewed annotations and the original frame time axes

from datetime import datetime, timedelta
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from secsy import CSprojection

from scripts.manual_masking.annotation_store import decode_mask, frame_class, propagated_mask


def load_good_orbit_intervals(filename):
    """Read inclusive manually accepted WIC-frame intervals from the legacy CSV."""

    intervals = {}
    with Path(filename).open(newline="", encoding="utf-8") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 5:
                raise ValueError(f"annotation row {line_number} has {len(row)} columns; expected 5")
            try:
                orbit = int(row[0])
                start = int(float(row[3])) if row[3].strip() else 0
                stop = int(float(row[4])) if row[4].strip() else None
            except ValueError as error:
                raise ValueError(f"annotation row {line_number} contains a nonnumeric value") from error
            if stop in (None, 0):
                continue
            if start < 0 or stop < start:
                raise ValueError(f"annotation row {line_number} has invalid interval {start}:{stop}")
            if orbit in intervals:
                raise ValueError(f"orbit {orbit} occurs more than once in annotations")
            intervals[orbit] = (start, stop)
    if not intervals:
        raise ValueError("annotation file contains no accepted orbit intervals")
    return intervals


def annotation_times(record):
    """Read the time axis actually used to review this orbit (legacy or CS)."""
    if record.get("frame_index_space") == "native_cs":
        from icreader import load
        with load(record["source_file"]) as product:
            times = np.asarray(product.time, dtype="datetime64[us]")
    else:
        with Dataset(record["source_file"]) as nc:
            reference = datetime.fromisoformat(nc.reference_time)
            times = [reference + timedelta(seconds=float(t)) for t in nc.variables["time"][:]]
        times = np.asarray(times, dtype="datetime64[us]")
    if len(times) != record["n_frames"] or np.any(np.diff(times) <= np.timedelta64(0, "us")):
        raise ValueError(f"orbit {record.get('orbit')}: annotation source time axis changed or is not increasing")
    return times


def load_annotations(filename, segments=None):
    filename = Path(filename).expanduser().resolve()
    contents = filename.read_bytes()
    document = json.loads(contents)
    if document.get("schema_version") != 2 or document.get("mask_shape") != [46, 46]:
        raise ValueError("--annotations requires the converted 46x46 annotation JSON")
    reviewed_statuses = ("complete", "no_usable") if segments is not None else ("complete",)
    records = {int(orbit): record for orbit, record in document["orbits"].items() if record["review_status"] in reviewed_statuses}
    if not records:
        raise ValueError("annotation file has no completed orbit reviews")

    source_times = {}
    timeline_hash = hashlib.sha256()
    for orbit, record in sorted(records.items()):
        times = annotation_times(record)
        source_times[orbit] = times
        timeline_hash.update(str(orbit).encode())
        timeline_hash.update(times.tobytes())

    # Read the actual target grid to prevent using masks on a different grid.
    with Dataset(document["target_grid_file"]) as nc:
        grid = nc.groups["grid"]
        projection = CSprojection(grid.position, grid.orientation)
        lon, lat = projection.cube2geo(np.asarray(grid.variables["xi"][:]), np.asarray(grid.variables["eta"][:]))
    provenance = {
        "filename": str(filename), "sha256": hashlib.sha256(contents).hexdigest(),
        "source_times_sha256": timeline_hash.hexdigest(), "completed_orbits": len(records),
        "time_matching": "nearest original frame within 1 second; unmatched frames excluded",
        "partial_masks": "last earlier keyframe within the same partial interval",
        "outside_original_grid": "Frame classes and drawn geometry apply across the entire native grid" if all(decode_mask(record["target_annotated_runs"], shape=(46, 46)).all() for record in records.values()) else "per-orbit explicit target_annotated_runs domain",
        "missing_fraction_policy": "intrinsic product validity before manual masking"
    }
    if any(record.get("frame_index_space") == "native_cs" for record in records.values()):
        provenance["time_matching"] = "nearest per-orbit annotation source frame within 1 second; unmatched frames excluded"
    annotations = {"records": records, "source_times": source_times, "grid": {"mlat": lat, "mlt": (lon / 15) % 24}, "provenance": provenance}
    if segments is not None:
        segments = Path(segments).expanduser().resolve()
        annotations["csv_intervals"] = load_good_orbit_intervals(segments)
        provenance["csv_fallback"] = {"filename": str(segments), "sha256": hashlib.sha256(segments.read_bytes()).hexdigest()}
        provenance["selection_policy"] = "completed JSON reviews override whole orbits; CSV fallback elsewhere"
    return annotations


def annotation_intervals(annotations):
    """Read all JSON-orbit frames; apply CSV source-index cuts only elsewhere."""
    if "csv_intervals" not in annotations:
        return None
    intervals = dict(annotations["csv_intervals"])
    intervals.update({orbit: None for orbit in annotations["records"]})
    return intervals


def select_annotation_files(files, annotations):
    selected = []
    for path in files:
        orbit = int(Path(path).stem.split("_")[-1])
        record = annotations["records"].get(orbit)
        if record is not None:
            if record["review_status"] != "no_usable":
                selected.append(path)
        elif orbit in annotations.get("csv_intervals", {}):
            selected.append(path)
    if not selected:
        raise ValueError("no input orbits have completed annotations")
    return selected


#%% Match timestamps and construct manual masks

def annotation_mask_arrays(product, annotations):
    """Return manual masks, classes and source frames on a loaded product axis."""

    for name in ("mlat", "mlt"):
        difference = product["grid"][name] - annotations["grid"][name]
        if name == "mlt":
            difference = (difference + 12) % 24 - 12
        if not np.allclose(difference, 0, atol=1e-6):
            raise ValueError("annotation masks and input product use different grids")

    count, _, height, width = product["values"].shape
    masks = np.zeros((count, height, width), dtype=bool)
    classes = np.full(count, "excluded", dtype="U8")
    source_frames = np.full(count, -1, dtype=int)
    unmatched = 0
    for orbit in np.unique(product["orbit"]):
        indices = np.flatnonzero(product["orbit"] == orbit)
        record = annotations["records"].get(int(orbit))
        if record is None:
            interval = annotations.get("csv_intervals", {}).get(int(orbit))
            if interval is not None:
                start, stop = interval
                source_indices = product["source_indices"]["wic"][indices]
                accepted = indices[(source_indices >= start) & (source_indices <= stop)]
                masks[accepted] = True
                classes[accepted] = "csv"
            continue
        old_times = annotations["source_times"][int(orbit)]
        new_times = product["time"][indices].astype("datetime64[us]")
        distances = np.abs(new_times[:, None] - old_times[None, :])
        nearest = distances.argmin(axis=1)
        matched = distances[np.arange(len(indices)), nearest] <= np.timedelta64(1, "s")
        unmatched += int((~matched).sum())
        footprint = decode_mask(record["target_annotated_runs"], shape=(height, width))
        for index, source_frame in zip(indices[matched], nearest[matched]):
            classification = frame_class(record, int(source_frame))
            if classification not in ("full", "partial"):
                continue
            masks[index] = propagated_mask(record, int(source_frame), shape=(height, width)) & footprint
            classes[index] = classification
            source_frames[index] = source_frame
    return masks, classes, source_frames, unmatched


#%% Select good frames and mask their paired fields

def apply_annotations(product, annotations):
    if product is None:
        return None
    count, _, height, width = product["values"].shape
    masks, classes, source_frames, unmatched = annotation_mask_arrays(product, annotations)

    # Manual exclusions affect both encoder input and reconstruction targets.
    # Keep the intrinsic missing fraction: an intentional partial mask is not
    # an upstream missing-data failure under the existing 33% frame cutoff.
    valid = product["valid"] & masks[:, None]
    common = np.all(valid, axis=1)
    weighted = common & (np.min(product["quality"], axis=1) > 0)
    # CSV fallback keeps the original frame-selection semantics, including
    # zero-quality frames; the existing intrinsic cutoff is still applied later.
    keep = np.flatnonzero(np.any(weighted, axis=(1, 2)) | (classes == "csv"))
    if not len(keep):
        raise ValueError("no annotated frames retain valid, positively weighted paired pixels")
    result = dict(product)
    frame_fields = ("values", "standard_deviation", "quality", "frame_missing_fraction", "si13_corrected", "dsi13_corrected", "time", "orbit", "kp", "kp_interval_start")
    for name in frame_fields:
        if name in product:
            result[name] = product[name][keep]
    for name in ("source_indices", "source_times", "frame_quality"):
        if name in product:
            result[name] = {sensor: values[keep] for sensor, values in product[name].items()}
    result["valid"] = valid[keep]
    result["quality"] = np.where(result["valid"], result["quality"], 0)
    result["annotation_class"] = classes[keep]
    result["annotation_source_frame"] = source_frames[keep]
    result["annotation_selection"] = {
        "input_frames": count, "selected_full": int(np.sum(classes[keep] == "full")),
        "selected_partial": int(np.sum(classes[keep] == "partial")),
        "selected_csv": int(np.sum(classes[keep] == "csv")),
        "excluded_frames": count - len(keep), "unmatched_timestamps": unmatched,
        "manually_excluded_previously_valid_cells": int(np.sum(product["valid"][keep] & ~valid[keep]))
    }
    return result
