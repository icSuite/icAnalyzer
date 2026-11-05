"""Read native IMAGE CS images and resume converted reviews by timestamp."""

#%% Native images and their actual grid

from pathlib import Path
import warnings

import numpy as np
from icreader import load

from scripts.manual_masking.annotated_data import annotation_times
from scripts.manual_masking.annotation_store import BOUNDARY_FIELDS, decode_mask, encode_mask, frame_class, keyframe_frames, new_orbit_record, orbit_key, propagated_mask


def load_native_images(filename, document):
    filename = Path(filename).resolve()
    with load(filename) as product:
        if product.grid_id != "image_apex_130km_46x46_v1" or product.shape[1:] != (46, 46):
            raise ValueError("annotation tool requires the native 46x46 IMAGE CS grid")
        if product.product_type == "conductance_cs":
            fields = ("H", "P", "method_quality_weight", "coverage")
            names = ("Hall", "Pedersen", "auroral quality", "coverage")
            valid = np.asarray(product.read("conductance_valid"), dtype=bool)
        elif product.product_type == "precipitation_cs":
            fields = ("wic_corrected", "si12", "method_quality_weight", "wic_coverage")
            names = ("corrected WIC", "SI12", "auroral quality", "WIC coverage")
            valid = np.isfinite(product.read("wic_corrected"))
        else:
            raise ValueError("choose conductance_cs or precipitation_cs files")
        channels = tuple(np.asarray(product.read(field), dtype=float) for field in fields)
        channels = tuple(np.where(valid, channel, np.nan) for channel in channels)
        times = np.asarray(product.time, dtype="datetime64[us]")
        source_wic = np.asarray(product.source_indices["wic"])
        latitude = np.asarray(product.mlat)
        local_time = np.asarray(product.mlt)
        companion_attribute = product.attrs.get("companion_precipitation_cs")
        is_conductance = product.product_type == "conductance_cs"
        sza = None if is_conductance else np.asarray(product.read("sza"), dtype=float)

    if is_conductance:
        # Copied products retain the producer's absolute path. Prefer the
        # companion in the same local IMAGE_FUV tree.
        companion = filename.parents[4] / "precipitation_cs" / filename.parents[2].name / filename.parents[1].name / filename.name
        if not companion.exists() and companion_attribute:
            companion = Path(companion_attribute)
        if companion.exists():
            with load(companion) as precipitation:
                if precipitation.grid_id != "image_apex_130km_46x46_v1" or not np.array_equal(precipitation.time, times):
                    raise ValueError("WIC companion must match the conductance grid and frame times")
                if not np.allclose(precipitation.mlat, latitude) or not np.allclose(precipitation.mlt, local_time):
                    raise ValueError("WIC companion coordinates differ from conductance images")
                # Keep WIC's own finite pixels, including where retrieval failed.
                wic = np.asarray(precipitation.read("wic_corrected"), dtype=float)
                sza = np.asarray(precipitation.read("sza"), dtype=float)
                if wic.shape != channels[0].shape:
                    raise ValueError("WIC companion shape differs from conductance images")
            channels += (wic,)
            names += ("corrected WIC",)
        else:
            warnings.warn(f"WIC companion not found for {filename.name}: {companion}", stacklevel=2)

    if "target_grid" in document:
        lat = np.asarray(document["target_grid"]["mlat"], dtype=float)
        target_mlt = np.asarray(document["target_grid"]["mlt"], dtype=float)
        difference = (local_time - target_mlt + 12) % 24 - 12
        if not np.allclose(latitude, lat, atol=1e-6) or not np.allclose(difference, 0, atol=1e-6):
            raise ValueError("native images differ from the embedded annotation grid")
    else:
        document["target_grid"] = {"mlat": latitude.tolist(), "mlt": local_time.tolist()}
    if sza is not None and sza.shape != channels[0].shape:
        raise ValueError("SZA shape differs from annotation images")
    return channels, names, times, source_wic, sza


#%% Preserve old masks while moving an opened orbit to the native time axis

def native_orbit_record(document, orbit, times, source_wic, seed=None):
    times = np.asarray(times, dtype="datetime64[us]")
    if len(times) == 0 or np.any(np.diff(times) <= np.timedelta64(0, "us")):
        raise ValueError("native frame times must be nonempty and strictly increasing")
    key = orbit_key(orbit)
    old = document["orbits"].get(key)
    if old is not None and old.get("frame_index_space") == "native_cs":
        if not np.array_equal(annotation_times(old), times):
            raise ValueError(f"orbit {orbit}: native annotation time axis differs from images")
        return old

    record = new_orbit_record(orbit, len(times))
    record.update(
        mask_shape=[46, 46], frame_index_space="native_cs",
        source_times_utc=np.datetime_as_string(times, unit="us").tolist(),
        target_annotated_runs=encode_mask(np.ones((46, 46), dtype=bool))
    )
    if old is None:
        # CSV boundaries are detector WIC IDs, not CS array indices.
        if seed is not None:
            accepted = np.flatnonzero((source_wic >= seed[0]) & (source_wic <= seed[1]))
            if accepted.size:
                for name in ("usable_start", "full_good_start"):
                    record[name] = int(accepted[0])
                for name in ("usable_stop", "full_good_stop"):
                    record[name] = int(accepted[-1])
    else:
        old_times = annotation_times(old)
        distances = np.abs(times[:, None] - old_times[None, :])
        nearest = distances.argmin(axis=1)
        matched = distances[np.arange(len(times)), nearest] <= np.timedelta64(1, "s")
        footprint = decode_mask(old["target_annotated_runs"], shape=(46, 46))
        classes = [frame_class(old, int(frame)) if found else "bad" for frame, found in zip(nearest, matched)]
        accepted = np.flatnonzero(np.isin(classes, ("full", "partial")))
        full = np.flatnonzero(np.asarray(classes) == "full")
        record["target_annotated_runs"] = old["target_annotated_runs"]
        record["notes"] = old.get("notes", "")
        record["review_status"] = old["review_status"]
        if accepted.size:
            record["usable_start"], record["usable_stop"] = int(accepted[0]), int(accepted[-1])
            if full.size and np.all(np.diff(full) == 1):
                record["full_good_start"], record["full_good_stop"] = int(full[0]), int(full[-1])
            previous_mask = None
            for frame in range(int(accepted[0]), int(accepted[-1]) + 1):
                if frame_class(record, frame) == "full":
                    previous_mask = None
                    continue
                mask = np.zeros((46, 46), dtype=bool)
                if matched[frame]:
                    mask = propagated_mask(old, int(nearest[frame]), shape=(46, 46)) & footprint
                if previous_mask is None or not np.array_equal(mask, previous_mask):
                    earlier = [number for number in keyframe_frames(old) if number <= nearest[frame]]
                    payload = dict(old["keyframes"][str(earlier[-1])]) if earlier and classes[frame] == "partial" else {}
                    payload["human_reviewed"] = False
                    payload.update(valid_runs=encode_mask(mask), valid_fraction=float(mask.mean()), source_time=str(times[frame]))
                    record["keyframes"][str(frame)] = payload
                previous_mask = mask
        elif old["review_status"] != "no_usable":
            # Keep unfinished draft boundaries/keyframes where times survive.
            record["review_status"] = "in_progress"
            for name in BOUNDARY_FIELDS:
                value = old.get(name)
                if value is not None:
                    found = np.flatnonzero(matched & (nearest == value))
                    record[name] = int(found[0]) if found.size else None
            for frame, payload in old.get("keyframes", {}).items():
                found = np.flatnonzero(matched & (nearest == int(frame)))
                if found.size:
                    record["keyframes"][str(int(found[0]))] = deepcopy(payload)
    document["orbits"][key] = record
    return record


def stamp_native_record(record, times):
    """Keep human-readable timestamps current after boundary/mask edits."""
    record["boundary_times"] = {name: str(times[int(record[name])]) if record.get(name) is not None else None for name in BOUNDARY_FIELDS}
    for frame, payload in record["keyframes"].items():
        payload["source_time"] = str(times[int(frame)])
