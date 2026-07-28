"""Persistent annotation state and temporal mask propagation.

The annotation file stores only orbit-level boundaries and sparse mask
keyframes. Full 36x36 masks are generated when processed training data is
built; annotators never need to label each image independently.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
MASK_SHAPE = (36, 36)
BOUNDARY_FIELDS = (
    "usable_start",
    "full_good_start",
    "full_good_stop",
    "usable_stop",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_document() -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": now,
        "updated_at": now,
        "cursor": {"orbit": None, "page_start": 0, "frame": 0},
        "orbits": {},
    }


def load_document(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return empty_document()

    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)

    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported annotation schema {document.get('schema_version')}; "
            f"expected {SCHEMA_VERSION}"
        )
    if not isinstance(document.get("orbits"), dict):
        raise ValueError("Annotation document is missing the 'orbits' mapping")
    document.setdefault("cursor", {"orbit": None, "page_start": 0, "frame": 0})
    return document


def save_document_atomic(document: dict[str, Any], path: str | Path) -> None:
    """Write a complete JSON document and atomically replace the prior file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document["updated_at"] = utc_now()

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def orbit_key(orbit: int) -> str:
    return f"{int(orbit):04d}"


def new_orbit_record(
    orbit: int,
    n_frames: int,
    seed_good_start: int | None = None,
    seed_good_stop: int | None = None,
) -> dict[str, Any]:
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")

    if seed_good_start is not None:
        seed_good_start = _validate_frame(seed_good_start, n_frames)
    if seed_good_stop is not None:
        seed_good_stop = _validate_frame(seed_good_stop, n_frames)
    if (
        seed_good_start is not None
        and seed_good_stop is not None
        and seed_good_start > seed_good_stop
    ):
        raise ValueError("Seed good interval has start after stop")

    now = utc_now()
    return {
        "orbit": int(orbit),
        "n_frames": int(n_frames),
        "review_status": "in_progress",
        "usable_start": seed_good_start,
        "full_good_start": seed_good_start,
        "full_good_stop": seed_good_stop,
        "usable_stop": seed_good_stop,
        "keyframes": {},
        "notes": "",
        "created_at": now,
        "updated_at": now,
    }


def ensure_orbit_record(
    document: dict[str, Any],
    orbit: int,
    n_frames: int,
    seed_good_start: int | None = None,
    seed_good_stop: int | None = None,
) -> dict[str, Any]:
    key = orbit_key(orbit)
    if key not in document["orbits"]:
        document["orbits"][key] = new_orbit_record(
            orbit,
            n_frames,
            seed_good_start=seed_good_start,
            seed_good_stop=seed_good_stop,
        )

    record = document["orbits"][key]
    if int(record["n_frames"]) != int(n_frames):
        raise ValueError(
            f"Orbit {orbit} has {n_frames} frames, but the annotation file "
            f"records {record['n_frames']}"
        )
    return record


def _validate_frame(frame: int, n_frames: int) -> int:
    frame = int(frame)
    if not 0 <= frame < int(n_frames):
        raise ValueError(f"Frame {frame} is outside [0, {n_frames - 1}]")
    return frame


def validate_boundaries(record: dict[str, Any]) -> None:
    n_frames = int(record["n_frames"])
    values: list[tuple[str, int]] = []
    for field in BOUNDARY_FIELDS:
        value = record.get(field)
        if value is not None:
            values.append((field, _validate_frame(value, n_frames)))

    for (left_name, left), (right_name, right) in zip(values, values[1:]):
        if left > right:
            raise ValueError(
                f"{left_name}={left} must not be after {right_name}={right}"
            )


def set_boundary(record: dict[str, Any], field: str, frame: int) -> None:
    if field not in BOUNDARY_FIELDS:
        raise ValueError(f"Unknown boundary field: {field}")

    previous = record.get(field)
    record[field] = _validate_frame(frame, int(record["n_frames"]))
    try:
        validate_boundaries(record)
    except ValueError:
        record[field] = previous
        raise
    record["review_status"] = "in_progress"
    record["updated_at"] = utc_now()


def clear_boundary(record: dict[str, Any], field: str) -> None:
    if field not in BOUNDARY_FIELDS:
        raise ValueError(f"Unknown boundary field: {field}")
    previous = record.get(field)
    record[field] = None
    try:
        validate_boundaries(record)
    except ValueError:
        record[field] = previous
        raise
    record["review_status"] = "in_progress"
    record["updated_at"] = utc_now()


def encode_mask(mask: np.ndarray) -> list[list[int]]:
    """Run-length encode True pixels in row-major order."""
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.r_[False, flat, False].astype(np.int8)
    changes = np.flatnonzero(np.diff(padded))
    starts = changes[::2]
    stops = changes[1::2]
    return [[int(start), int(stop - start)] for start, stop in zip(starts, stops)]


def decode_mask(
    runs: list[list[int]],
    shape: tuple[int, int] = MASK_SHAPE,
) -> np.ndarray:
    size = int(np.prod(shape))
    flat = np.zeros(size, dtype=bool)
    for start, length in runs:
        start = int(start)
        length = int(length)
        if start < 0 or length < 0 or start + length > size:
            raise ValueError("Invalid mask run")
        flat[start : start + length] = True
    return flat.reshape(shape)


def set_keyframe(
    record: dict[str, Any],
    frame: int,
    mask: np.ndarray,
    vertices: list[list[float]] | None = None,
    geometry: dict[str, Any] | None = None,
) -> None:
    frame = _validate_frame(frame, int(record["n_frames"]))
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != MASK_SHAPE:
        raise ValueError(f"Mask must have shape {MASK_SHAPE}")

    payload: dict[str, Any] = {
        "valid_runs": encode_mask(mask),
        "valid_fraction": float(mask.mean()),
        "updated_at": utc_now(),
    }
    if vertices is not None:
        payload["vertices"] = [
            [float(x_coordinate), float(y_coordinate)]
            for x_coordinate, y_coordinate in vertices
        ]
    if geometry is not None:
        payload["geometry"] = deepcopy(geometry)

    record["keyframes"][str(frame)] = payload
    record["review_status"] = "in_progress"
    record["updated_at"] = utc_now()


def delete_keyframe(record: dict[str, Any], frame: int) -> bool:
    removed = record["keyframes"].pop(str(int(frame)), None) is not None
    if removed:
        record["review_status"] = "in_progress"
        record["updated_at"] = utc_now()
    return removed


def exact_keyframe_mask(
    record: dict[str, Any],
    frame: int,
    shape: tuple[int, int] = MASK_SHAPE,
) -> np.ndarray | None:
    payload = record.get("keyframes", {}).get(str(int(frame)))
    if payload is None:
        return None
    return decode_mask(payload["valid_runs"], shape=shape)


def keyframe_frames(record: dict[str, Any]) -> list[int]:
    return sorted(int(frame) for frame in record.get("keyframes", {}))


def frame_class(record: dict[str, Any], frame: int) -> str:
    frame = _validate_frame(frame, int(record["n_frames"]))
    if record.get("review_status") == "no_usable":
        return "bad"

    usable_start = record.get("usable_start")
    usable_stop = record.get("usable_stop")
    if usable_start is None or usable_stop is None:
        return "unreviewed"
    if frame < usable_start or frame > usable_stop:
        return "bad"

    full_start = record.get("full_good_start")
    full_stop = record.get("full_good_stop")
    if (
        full_start is not None
        and full_stop is not None
        and full_start <= frame <= full_stop
    ):
        return "full"
    return "partial"


def propagated_mask(
    record: dict[str, Any],
    frame: int,
    shape: tuple[int, int] = MASK_SHAPE,
) -> np.ndarray:
    classification = frame_class(record, frame)
    if classification == "full":
        return np.ones(shape, dtype=bool)
    if classification in {"bad", "unreviewed"}:
        return np.zeros(shape, dtype=bool)

    exact = exact_keyframe_mask(record, frame, shape=shape)
    if exact is not None:
        return exact

    usable_start = int(record["usable_start"])
    usable_stop = int(record["usable_stop"])
    full_start = record.get("full_good_start")
    full_stop = record.get("full_good_stop")
    if full_start is not None and frame < full_start:
        interval_start, interval_stop = usable_start, int(full_start) - 1
    elif full_stop is not None and frame > full_stop:
        interval_start, interval_stop = int(full_stop) + 1, usable_stop
    else:
        interval_start, interval_stop = usable_start, usable_stop

    earlier_keyframes = [
        keyframe
        for keyframe in keyframe_frames(record)
        if interval_start <= keyframe <= frame <= interval_stop
    ]
    if not earlier_keyframes:
        return np.zeros(shape, dtype=bool)
    mask = exact_keyframe_mask(record, earlier_keyframes[-1], shape=shape)
    if mask is None:
        raise RuntimeError("Stored keyframe disappeared while propagating mask")
    return mask


def mark_no_usable(record: dict[str, Any]) -> None:
    for field in BOUNDARY_FIELDS:
        record[field] = None
    record["keyframes"] = {}
    record["review_status"] = "no_usable"
    record["updated_at"] = utc_now()


def mark_complete(record: dict[str, Any]) -> None:
    validate_boundaries(record)
    usable_start = record.get("usable_start")
    usable_stop = record.get("usable_stop")
    if usable_start is None or usable_stop is None:
        raise ValueError("Set a usable interval or mark the orbit as no usable data")

    full_start = record.get("full_good_start")
    full_stop = record.get("full_good_stop")
    if (full_start is None) != (full_stop is None):
        raise ValueError(
            "full_good_start and full_good_stop must either both be set or "
            "both be empty"
        )
    has_partial_frames = (
        full_start is None
        or usable_start < full_start
        or full_stop < usable_stop
    )
    partial_intervals: list[tuple[int, int]] = []
    if full_start is None:
        partial_intervals.append((usable_start, usable_stop))
    else:
        if usable_start < full_start:
            partial_intervals.append((usable_start, full_start - 1))
        if full_stop < usable_stop:
            partial_intervals.append((full_stop + 1, usable_stop))

    frames = keyframe_frames(record)
    if has_partial_frames:
        for interval_start, interval_stop in partial_intervals:
            if interval_start not in frames:
                raise ValueError(
                    f"Add the initial mask keyframe at frame {interval_start} "
                    f"for partial interval {interval_start}:{interval_stop}"
                )

    record["review_status"] = "complete"
    record["updated_at"] = utc_now()


def restore_record(target: dict[str, Any], snapshot: dict[str, Any]) -> None:
    target.clear()
    target.update(deepcopy(snapshot))


def load_good_segment_seeds(path: str | Path) -> dict[int, tuple[int, int]]:
    """Load only finite, positive-stop good intervals.

    A blank start is interpreted as frame zero, matching the corrected IMAGE
    loader. Blank stops and zero stops remain unseeded because their historical
    semantics are ambiguous.
    """
    seeds: dict[int, tuple[int, int]] = {}
    path = Path(path)
    if not path.exists():
        return seeds

    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) < 5:
                continue
            try:
                orbit = int(float(row[0]))
            except ValueError:
                continue

            start_text = row[3].strip()
            stop_text = row[4].strip()
            if not stop_text:
                continue
            try:
                stop = int(float(stop_text))
                start = int(float(start_text)) if start_text else 0
            except ValueError:
                continue
            if stop <= 0 or start < 0 or start > stop:
                continue
            seeds[orbit] = (start, stop)
    return seeds


def progress_counts(
    document: dict[str, Any],
    available_orbits: list[int],
) -> dict[str, int]:
    counts = {"unreviewed": 0, "in_progress": 0, "complete": 0, "no_usable": 0}
    for orbit in available_orbits:
        record = document["orbits"].get(orbit_key(orbit))
        status = "unreviewed" if record is None else record["review_status"]
        counts[status] = counts.get(status, 0) + 1
    return counts
