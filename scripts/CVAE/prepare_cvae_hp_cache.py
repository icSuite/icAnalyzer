"""Prepare a disk-backed Hall/Pedersen CVAE cache from automatic WIC masks.

This is a separate workflow. Existing VAE/CVAE runners and annotations are
not modified. Preparation never trains a model.
"""

#%% Imports

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from tqdm.auto import tqdm
from icreader import load

from icanalyzer.conductance import load_conductance_cs
from icanalyzer.cvae.cache import (
    AUTOMATIC_CONDUCTANCE_CACHE_FORMAT,
    AutomaticConductanceDataset,
    AutomaticFieldDataset,
    file_sha256,
    read_cache
)
from icanalyzer.cvae.data import ChannelTransform, fit_precision_weighting, split_orbit_files


CACHE_FORMAT = AUTOMATIC_CONDUCTANCE_CACHE_FORMAT
REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path.home() / "Dropbox/work/data/IMAGE_FUV_new/conductance_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic1_si121_si131/robinson"
DEFAULT_MASKS = REPOSITORY / "data/training_data/automatic_masks_fuvpy_bs2d_sigma2_pointing_v1_e5c009e6713b"


#%% Align each orbit's masks to its native conductance images

def masked_orbit(filename, mask_directory, mask_record):
    mask_file = mask_directory / (filename.stem + ".npz")
    with np.load(mask_file, allow_pickle=False) as stored:
        times = stored["time"].astype("datetime64[us]")
        masks = stored["mask"]
    if masks.dtype != bool or masks.shape != (mask_record["frames"], 46, 46) or times.shape != (len(masks),):
        raise ValueError(f"{mask_file}: expected Boolean masks and matching native timestamps")
    if len(times) == 0 or np.isnat(times).any() or np.any(np.diff(times) <= np.timedelta64(0, "us")):
        raise ValueError(f"{mask_file}: timestamps must be nonempty and strictly increasing")

    # The manifest refers to precipitation_cs, not the conductance file.
    source = Path(mask_record["source"])
    evidence = source.stat()
    if evidence.st_size != mask_record["source_bytes"] or evidence.st_mtime_ns != mask_record["source_mtime_ns"]:
        raise ValueError(f"{source}: mask source changed; use masks from this product generation")
    product = load_conductance_cs(filename, include_kp=True)
    with load(source) as original:
        if original.grid_id != product["metadata"]["grid_id"] or not np.array_equal(original.time.astype("datetime64[us]"), times):
            raise ValueError(f"{source}: automatic mask time axis/grid identity differs from its source")
        for name in ("mlat", "mlt"):
            difference = np.asarray(getattr(original, name)) - product["grid"][name]
            if name == "mlt":
                difference = (difference + 12) % 24 - 12
            if not np.allclose(difference, 0, atol=1e-6):
                raise ValueError(f"{filename}: masks and conductance use different {name} grids")

    current_times = product["time"].astype("datetime64[us]")
    position = np.searchsorted(times, current_times)
    position = np.minimum(position, len(times) - 1)
    matched = times[position] == current_times
    common = np.all(product["valid"], axis=1) & masks[position] & matched[:, None, None]
    quality = np.min(product["quality"], axis=1)
    positive = np.any(common & np.isfinite(quality) & (quality > 0), axis=(1, 2))
    product["automatic_valid"] = common
    product["automatic_positive"] = positive
    product["automatic_evidence"] = {"filename": str(mask_file.resolve()), "sha256": file_sha256(mask_file), "unmatched_frames": int((~matched).sum()), "no_positive_masked_frames": int((matched & ~positive).sum())}
    return product


#%% Prepare one orbit at a time; never concatenate the image corpus

def prepare(arguments):
    source = arguments.source.expanduser().resolve()
    masks = arguments.masks.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("choose a new cache directory; existing caches are never overwritten")
    if not np.isfinite(arguments.maximum_missing_fraction) or not 0 <= arguments.maximum_missing_fraction <= 1:
        raise ValueError("maximum missing fraction must be within 0–1")
    minimum_valid = 1.0 - arguments.maximum_missing_fraction
    print(f"Frame eligibility: at least {100 * minimum_valid:g}% intrinsic valid H/P coverage before automatic masking.", flush=True)
    mask_manifest = masks / "manifest.json"
    mask_summary = json.loads(mask_manifest.read_text())
    records = {row["orbit"]: row for row in mask_summary["orbits"]}
    files = sorted(source.glob("or_*.nc"))
    if arguments.orbits is not None:
        requested = set(arguments.orbits)
        files = [path for path in files if int(path.stem[3:]) in requested]
        if {int(path.stem[3:]) for path in files} != requested:
            raise FileNotFoundError("some requested conductance orbits are missing")
    if not files:
        raise FileNotFoundError(f"no conductance_cs files in {source}")
    missing = [path.stem for path in files if path.stem not in records or not (masks / (path.stem + ".npz")).is_file()]
    if missing:
        raise FileNotFoundError(f"missing automatic masks for {missing[:8]}")
    split = split_orbit_files(files, arguments.validation_fraction, arguments.test_fraction, arguments.seed)

    # Conservative upper bound: five float32 maps + a Boolean map per frame.
    upper_bytes = sum(records[path.stem]["frames"] for path in files) * 46 * 46 * 21
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    free_bytes = shutil.disk_usage(ancestor).free
    print(f"Cache upper estimate: {upper_bytes / 1024**3:.2f} GiB; free disk: {free_bytes / 1024**3:.2f} GiB.", flush=True)
    if free_bytes < upper_bytes * 1.1:
        raise OSError("not enough free disk for the conservative cache estimate plus 10% margin")
    output.mkdir(parents=True)

    rows = []
    counts = np.zeros(2, dtype=np.int64)
    sums = np.zeros(2, dtype=np.float64)
    squared_sums = np.zeros(2, dtype=np.float64)
    training_support = np.zeros((46, 46), dtype=bool)
    reference_grid = None
    reference_metadata = None
    used_counts = {name: 0 for name in split}
    progress = tqdm(total=len(files), desc="Prepare CVAE cache", unit="orbit", dynamic_ncols=True)
    for split_name, paths in split.items():
        for filename in paths:
            product = masked_orbit(filename, masks, records[filename.stem])
            if reference_grid is None:
                reference_grid = product["grid"]
                reference_metadata = product["metadata"]
            for name in ("mlat", "mlt"):
                if not np.allclose(product["grid"][name], reference_grid[name]):
                    raise ValueError(f"{filename}: conductance grids differ across orbits")
            for name in ("proton_energy_model", "proton_flux_source", "nonlinear_ordering", "method_quality_weight_method"):
                if product["metadata"][name] != reference_metadata[name]:
                    raise ValueError(f"{filename}: incompatible upstream {name}")

            # Preserve current ordering: normalization/precision use positive
            # manually masked frames BEFORE the intrinsic 33% training cut.
            indices = np.flatnonzero(product["automatic_positive"])
            common = product["automatic_valid"][indices]
            eligible = product["frame_missing_fraction"][indices] <= arguments.maximum_missing_fraction
            kp = product["kp"][indices]
            if np.any(~np.isfinite(kp[eligible])) or np.any((kp[eligible] < 0) | (kp[eligible] > 9)):
                raise ValueError(f"{filename}: retained CVAE images need finite Kp within 0–9")
            used = int(eligible.sum())
            row = {"orbit": int(filename.stem[3:]), "split": split_name, "source": str(filename), "product": product["metadata"], "automatic_mask": product["automatic_evidence"], "input_frames": len(product["time"]), "prepared_frames": len(indices), "used_frames": used, "intrinsic_missing_dropped_frames": int((~eligible).sum()), "shard": None}
            if len(indices):
                shard = output / filename.stem
                shard.mkdir()
                values = product["values"][indices]
                errors = product["standard_deviation"][indices]
                quality = np.where(common, np.min(product["quality"][indices], axis=1), 0.0)
                physical = np.concatenate((values, errors, quality[:, None]), axis=1).astype(np.float32)
                np.save(shard / "physical.npy", physical, allow_pickle=False)
                np.save(shard / "valid.npy", common, allow_pickle=False)
                source_indices = {f"{sensor}_source_index": value[indices] for sensor, value in product["source_indices"].items()}
                np.savez(shard / "frames.npz", time=product["time"][indices].astype("datetime64[us]"), source_frame=indices, kp=kp, kp_interval_start=product["kp_interval_start"][indices], eligible=eligible, **source_indices)
                row["shard"] = filename.stem
                if split_name == "train":
                    for channel in range(2):
                        logarithms = np.log10(values[:, channel][common] + 1.0).astype(np.float64)
                        counts[channel] += logarithms.size
                        sums[channel] += logarithms.sum()
                        squared_sums[channel] += np.sum(logarithms**2)
                    if used:
                        training_support |= np.any(common[eligible], axis=0)
                del values, errors, quality, physical
            used_counts[split_name] += used
            rows.append(row)
            progress.update(1)
            progress.set_postfix(used=sum(used_counts.values()), refresh=False)
            del product, common
    progress.close()
    for name in split:
        if split[name] and used_counts[name] == 0:
            raise ValueError(f"{name} split has no eligible images; cache is incomplete")
    if np.any(counts == 0):
        raise ValueError("no valid training pixels for log normalization")
    center = sums / counts
    spread = np.sqrt(np.maximum(squared_sums / counts - center**2, 0.0))
    if not np.isfinite(spread).all() or np.any(spread <= 0):
        raise ValueError("training log spread must be finite and positive")
    transform = ChannelTransform([1, 1], center, spread, kind=ChannelTransform.LOG10_KIND)
    np.save(output / "training_support.npy", training_support, allow_pickle=False)
    metadata = {
        "cache_format": CACHE_FORMAT, "input_product_type": "conductance_cs", "field_names": ["Hall", "Pedersen"], "field_units": "S",
        "source_directory": str(source), "automatic_masks": {"directory": str(masks), "manifest_sha256": file_sha256(mask_manifest), "model_sha256": mask_summary["model_sha256"], "threshold": mask_summary["threshold"]},
        "alignment": "exact timestamp; unmatched conductance frames excluded; mask AND intrinsic common validity",
        "maximum_missing_fraction": arguments.maximum_missing_fraction, "minimum_valid_fraction": 1.0 - arguments.maximum_missing_fraction, "seed": arguments.seed, "validation_fraction": arguments.validation_fraction, "test_fraction": arguments.test_fraction,
        "normalization_population": "positive masked training frames before intrinsic missing-fraction cut; float64 log moment accumulation",
        "transform": transform.to_dict(), "grid": {name: value.tolist() for name, value in reference_grid.items()}, "grid_id": reference_metadata["grid_id"],
        "split_files": {name: [str(path) for path in paths] for name, paths in split.items()}, "used_frames": used_counts, "orbits": rows
    }
    # This manifest is the completion marker. Never load a partial preparation.
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Prepared {used_counts}; cache: {output}", flush=True)
    return metadata


def cached_coverage_weighting(cache_directory, sigma_pixels=1.5, maximum_boost=3., kp_bin_edges=(0, 9), balance_mode="capped"):
    """Stream only eligible training quality maps; validation/test never fit weights."""
    from icanalyzer.cvae.loss import coverage_bin_indices, smooth_coverage_weighting

    coverage_bin_indices([], kp_bin_edges)
    directory = Path(cache_directory)
    metadata = read_cache(directory)
    quality_sum = None
    frame_counts = np.zeros(len(kp_bin_edges) - 1, dtype=np.int64)
    records = [row for row in metadata["orbits"] if row["split"] == "train" and row["shard"] is not None]
    for row in tqdm(records, desc="Fit training coverage", unit="orbit", dynamic_ncols=True):
        shard = directory / row["shard"]
        with np.load(shard / "frames.npz", allow_pickle=False) as frames:
            selected = np.flatnonzero(frames["eligible"])
            bins = coverage_bin_indices(frames["kp"][selected], kp_bin_edges)
        physical = np.load(shard / "physical.npy", mmap_mode="r", allow_pickle=False)
        valid = np.load(shard / "valid.npy", mmap_mode="r", allow_pickle=False)
        try:
            if physical.shape != (len(valid), 5, *valid.shape[1:]) or valid.ndim != 3:
                raise ValueError(f"{shard}: invalid coverage cache layout")
            if quality_sum is None:
                quality_sum = np.zeros((len(kp_bin_edges) - 1, *valid.shape[1:]), dtype=np.float64)
            if quality_sum.shape[1:] != valid.shape[1:]:
                raise ValueError("coverage cache grids differ between training orbits")
            for start in range(0, len(selected), 64):
                indices, bin_ids = selected[start:start + 64], bins[start:start + 64]
                quality = physical[indices, 4]
                quality = np.where(valid[indices] & np.isfinite(quality) & (quality > 0), quality, 0.)
                for index in np.unique(bin_ids):
                    quality_sum[index] += quality[bin_ids == index].sum(axis=0, dtype=np.float64)
                    frame_counts[index] += int((bin_ids == index).sum())
        finally:
            physical._mmap.close()
            valid._mmap.close()
    if quality_sum is None:
        raise ValueError("no eligible training coverage to fit")
    if int(frame_counts.sum()) != metadata["used_frames"]["train"]:
        raise ValueError("training coverage frame count differs from the cache manifest")
    configuration, smoothed = smooth_coverage_weighting(quality_sum, sigma_pixels, maximum_boost, kp_bin_edges, balance_mode)
    configuration.update(training_frames_per_bin=frame_counts.tolist(), cache_manifest_sha256=metadata["manifest_sha256"])
    arrays = dict(quality_sum=quality_sum, smoothed_quality_sum=smoothed,
        factors=np.asarray(configuration["factors"], dtype=np.float32), kp_bin_edges=np.asarray(kp_bin_edges))
    return configuration, arrays


def cached_precision_configuration(cache_directory, variance_floor, maximum_relative_precision, maximum_samples):
    """Use the existing precision formula on its bounded, global frame sample."""
    if maximum_samples < 1:
        raise ValueError("maximum precision sample count must be positive")
    directory = Path(cache_directory)
    metadata = read_cache(directory)
    transform = ChannelTransform.from_dict(metadata["transform"])
    rows = [row for row in metadata["orbits"] if row["split"] == "train" and row["shard"]]
    count = sum(row["prepared_frames"] for row in rows)
    stride = max(1, int(np.ceil(count * 46 * 46 / maximum_samples)))
    physical_samples, masks = [], []
    offset = 0
    for row in rows:
        indices = np.arange(row["prepared_frames"])
        indices = indices[(offset + indices) % stride == 0]
        if len(indices):
            physical = np.load(directory / row["shard"] / "physical.npy", mmap_mode="r", allow_pickle=False)
            valid = np.load(directory / row["shard"] / "valid.npy", mmap_mode="r", allow_pickle=False)
            physical_samples.append(physical[indices].copy())
            masks.append(valid[indices].copy())
            physical._mmap.close()
            valid._mmap.close()
        offset += row["prepared_frames"]
    physical = np.concatenate(physical_samples)
    valid = np.broadcast_to(np.concatenate(masks)[:, None], physical[:, :2].shape)
    quality = np.broadcast_to(physical[:, 4:5], physical[:, :2].shape)
    sample = {"values": physical[:, :2], "standard_deviation": physical[:, 2:4], "valid": valid, "quality": quality}
    configuration = fit_precision_weighting(sample, transform, variance_floor, maximum_relative_precision, maximum_scale_samples=max(maximum_samples, len(physical) * 46 * 46))
    configuration.update(frame_stride=stride, maximum_scale_samples=maximum_samples)
    return configuration


#%% Preparation command; training is a different script

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--masks", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orbits", type=int, nargs="+", help="optional explicit orbit subset")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    coverage = parser.add_mutually_exclusive_group()
    coverage.add_argument("--minimum-valid-fraction", type=float, help="Minimum intrinsic common H/P coverage before U-net masking (default: 0.05)")
    coverage.add_argument("--maximum-missing-fraction", type=float, help="Equivalent maximum intrinsic missing fraction; use 0.33 to reproduce the original cutoff")
    arguments = parser.parse_args(argv)
    if arguments.maximum_missing_fraction is None:
        valid = 0.05 if arguments.minimum_valid_fraction is None else arguments.minimum_valid_fraction
        if not np.isfinite(valid) or not 0 <= valid <= 1:
            parser.error("--minimum-valid-fraction must be within 0–1")
        arguments.maximum_missing_fraction = 1.0 - valid
    if not np.isfinite(arguments.maximum_missing_fraction) or not 0 <= arguments.maximum_missing_fraction <= 1:
        parser.error("--maximum-missing-fraction must be within 0–1")
    arguments.minimum_valid_fraction = 1.0 - arguments.maximum_missing_fraction
    return arguments


if __name__ == "__main__":
    prepare(parse_arguments())
