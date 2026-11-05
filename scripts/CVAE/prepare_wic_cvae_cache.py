"""Prepare a disk-backed WIC-only Kp-CVAE cache.

The cache contains corrected WIC, its formal standard deviation, the Product-2
auroral-quality field, Kp, exact frame provenance and either manual annotations
or a frozen automatic-mask policy.
Preparation never trains a model and never overwrites an existing cache.
"""

#%% Imports and product contract

import argparse
import json
from pathlib import Path
import shutil
import warnings

import numpy as np
from tqdm.auto import tqdm
from icreader import load

from icanalyzer.cvae.cache import file_sha256, read_cache
from icanalyzer.cvae.data import ChannelTransform, split_orbit_files
from scripts.manual_masking.annotated_data import annotation_mask_arrays, load_annotations


WIC_CACHE_FORMAT = "wic_cvae_cache_v1"
LEGACY_WIC_CACHE_FORMAT = "automatic_wic_cache_v1"
SUPPORTED_WIC_CACHE_FORMATS = (WIC_CACHE_FORMAT, LEGACY_WIC_CACHE_FORMAT)
GRID_ID = "image_apex_130km_46x46_v1"
DEFAULT_SOURCE = Path.home() / "Dropbox/work/data/IMAGE_FUV_new/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic1_si121_si131"


def read_wic_cache(cache_directory):
    """Read current or pre-rename WIC caches without weakening format checks."""

    directory = Path(cache_directory)
    cache_format = json.loads((directory / "manifest.json").read_text()).get("cache_format")
    if cache_format not in SUPPORTED_WIC_CACHE_FORMATS:
        raise ValueError(f"unsupported WIC CVAE cache format {cache_format!r}")
    return read_cache(directory, expected_format=cache_format)


def _source_evidence(filename, record):
    recorded = Path(record["source"]).expanduser().resolve()
    filename = Path(filename).resolve()
    if recorded != filename:
        raise ValueError(f"{filename}: automatic mask belongs to {recorded}")
    evidence = filename.stat()
    if evidence.st_size != record["source_bytes"] or evidence.st_mtime_ns != record["source_mtime_ns"]:
        raise ValueError(f"{filename}: mask source changed; regenerate masks for this Product-2 generation")


def _load_wic_fields(filename, product_loader=load):
    """Load and validate the fields shared by manual and automatic selection."""

    with product_loader(filename) as product:
        if product.product_type != "precipitation_cs" or product.grid_id != GRID_ID or tuple(product.shape[1:]) != (46, 46):
            raise ValueError(f"{filename}: expected precipitation_cs on {GRID_ID}")
        required = ("wic_corrected", "dwic_corrected", "method_quality_weight", "wic_coverage", "Kp")
        missing = [name for name in required if not hasattr(product, name)]
        if missing:
            raise ValueError(f"{filename}: missing WIC-CVAE fields: {', '.join(missing)}")
        quality_method = str(product.attrs.get("method_quality_weight_method", "")).lower()
        if "auroral-quality" not in quality_method or "1 minus" not in quality_method:
            raise ValueError(f"{filename}: ambiguous Product-2 auroral-quality orientation")

        time = np.asarray(product.time, dtype="datetime64[us]")
        values = np.asarray(product.read("wic_corrected"), dtype=np.float32)
        errors = np.asarray(product.read("dwic_corrected"), dtype=np.float32)
        quality = np.asarray(product.read("method_quality_weight"), dtype=np.float32)
        coverage = np.asarray(product.read("wic_coverage"), dtype=np.float32)
        kp = np.asarray(product.read("Kp"), dtype=np.float32)
        kp_interval_start = np.asarray(product.Kp_interval_start, dtype="datetime64[us]")
        wic_source_index = np.asarray(product.source_indices["wic"], dtype=np.int32)
        grid = {name: np.asarray(getattr(product, name), dtype=np.float64).copy() for name in ("mlat", "mlt")}
        metadata = {
            "orbit": int(Path(filename).stem[3:]), "product_type": str(product.product_type),
            "schema_version": int(product.schema_version), "software_version": str(product.software_version),
            "grid_id": str(product.grid_id), "count_source": str(product.attrs.get("count_source", "unrecorded")),
            "source_preprocessing_label": str(product.attrs.get("source_preprocessing_label", "unrecorded")),
            "method_quality_weight_method": str(product.attrs["method_quality_weight_method"]),
            "proton_energy_model": str(product.attrs.get("proton_energy_model", "unrecorded")),
            "proton_flux_source": str(product.attrs.get("proton_flux_source", "unrecorded"))
        }

    if values.shape[1:] != (46, 46) or any(array.shape != values.shape for array in (errors, quality, coverage)):
        raise ValueError(f"{filename}: WIC fields do not share the expected grid")
    if kp.shape != (len(values),) or kp_interval_start.shape != kp.shape or wic_source_index.shape != kp.shape:
        raise ValueError(f"{filename}: Kp/source metadata do not align with WIC frames")
    if np.any(~np.isfinite(kp)) or np.any((kp < 0) | (kp > 9)):
        raise ValueError(f"{filename}: WIC-CVAE frames require finite Kp within 0--9")
    finite_quality = quality[np.isfinite(quality)]
    if np.any((finite_quality < 0) | (finite_quality > 1)):
        raise ValueError(f"{filename}: method_quality_weight must lie within 0--1")

    return {
        "values": values, "standard_deviation": errors, "quality": quality,
        "coverage": coverage, "time": time, "kp": kp,
        "kp_interval_start": kp_interval_start,
        "wic_source_index": wic_source_index, "grid": grid,
        "metadata": metadata
    }


def _apply_mask_selection(product, selection_mask, mask_evidence, annotation_classes=None, annotation_source_frames=None):
    """Intersect one explicit mask source with intrinsic Product-2 support."""

    values = product["values"]
    errors = product["standard_deviation"]
    quality = product["quality"]
    coverage = product["coverage"]
    if selection_mask.dtype != bool or selection_mask.shape != values.shape:
        raise ValueError("WIC selection mask must be Boolean with shape (frame, 46, 46)")

    negative = np.isfinite(values) & (values < 0)
    negative_count = int(negative.sum())
    values = np.maximum(values, 0.0)
    intrinsic = np.isfinite(values) & np.isfinite(quality) & (quality > 0) & np.isfinite(coverage) & (coverage > 0)
    valid = intrinsic & selection_mask
    common_quality = quality * np.clip(coverage, 0.0, 1.0)
    positive = np.any(valid, axis=(1, 2))
    frame_missing_fraction = np.mean(~intrinsic, axis=(1, 2))
    product["metadata"]["negative_wic_clipped_cells"] = negative_count

    selected = {
        "values": values[:, None], "standard_deviation": errors[:, None],
        "quality": np.where(valid, common_quality, 0.0), "valid": valid,
        "positive": positive, "frame_missing_fraction": frame_missing_fraction,
        "time": product["time"], "kp": product["kp"],
        "kp_interval_start": product["kp_interval_start"],
        "wic_source_index": product["wic_source_index"], "grid": product["grid"],
        "metadata": product["metadata"], "mask_evidence": mask_evidence
    }
    if annotation_classes is not None:
        selected["annotation_class"] = annotation_classes
        selected["annotation_source_frame"] = annotation_source_frames
    return selected


def load_masked_wic(filename, mask_directory, mask_record, product_loader=load):
    """Load one Product-2 orbit and intersect WIC with a frozen automatic mask."""

    _source_evidence(filename, mask_record)
    mask_file = Path(mask_directory) / (Path(filename).stem + ".npz")
    with np.load(mask_file, allow_pickle=False) as stored:
        mask_time = stored["time"].astype("datetime64[us]")
        automatic_mask = stored["mask"]
    if automatic_mask.dtype != bool or automatic_mask.shape != (mask_record["frames"], 46, 46):
        raise ValueError(f"{mask_file}: expected Boolean (frame, 46, 46) masks")
    product = _load_wic_fields(filename, product_loader)
    if not np.array_equal(product["time"], mask_time):
        raise ValueError(f"{filename}: automatic masks and Product 2 have different timestamps")
    evidence = {"kind": "automatic", "filename": str(mask_file.resolve()), "sha256": file_sha256(mask_file)}
    return _apply_mask_selection(product, automatic_mask, evidence)


def load_annotated_wic(filename, annotations, product_loader=load):
    """Load one Product-2 orbit and apply completed manual frame/pixel reviews."""

    product = _load_wic_fields(filename, product_loader)
    count = len(product["time"])
    annotation_product = {
        "values": product["values"][:, None],
        "time": product["time"],
        "orbit": np.full(count, product["metadata"]["orbit"], dtype=int),
        "source_indices": {"wic": product["wic_source_index"]},
        "grid": product["grid"]
    }
    masks, classes, source_frames, unmatched = annotation_mask_arrays(annotation_product, annotations)
    evidence = {
        "kind": "manual", "orbit": product["metadata"]["orbit"],
        "selected_full": int(np.sum(classes == "full")),
        "selected_partial": int(np.sum(classes == "partial")),
        "excluded_frames": int(np.sum(classes == "excluded")),
        "unmatched_timestamps": unmatched
    }
    return _apply_mask_selection(product, masks, evidence, classes, source_frames)


#%% Prepare one orbit at a time

def prepare(arguments, product_loader=load):
    source = arguments.source.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("choose a new WIC cache directory; existing caches are never overwritten")
    if not 0 <= arguments.maximum_missing_fraction <= 1:
        raise ValueError("maximum missing fraction must lie within 0--1")

    files = sorted(source.glob("or_*.nc"))
    if arguments.orbits is not None:
        requested = set(arguments.orbits)
        files = [path for path in files if int(path.stem[3:]) in requested]
        if {int(path.stem[3:]) for path in files} != requested:
            raise FileNotFoundError("some requested Product-2 orbits are missing")
    if not files:
        raise FileNotFoundError(f"no precipitation_cs files in {source}")

    masks = None
    records = None
    annotations = None
    if arguments.masks is not None:
        masks = arguments.masks.expanduser().resolve()
        mask_manifest = masks / "manifest.json"
        mask_summary = json.loads(mask_manifest.read_text())
        records = {row["orbit"]: row for row in mask_summary["orbits"]}
        missing = [path.stem for path in files if path.stem not in records or not (masks / (path.stem + ".npz")).is_file()]
        if missing:
            raise FileNotFoundError(f"missing automatic masks for {missing[:8]}")
        mask_source = {
            "kind": "automatic", "directory": str(masks),
            "manifest_sha256": file_sha256(mask_manifest),
            "model_sha256": mask_summary["model_sha256"],
            "threshold": mask_summary["threshold"],
            "frame_gate": mask_summary.get("frame_gate"),
            "mask_meaning": mask_summary.get("mask_meaning")
        }
        estimated_frames = sum(records[path.stem]["frames"] for path in files)
        alignment = "same Product-2 file and exact timestamp; frozen automatic mask AND native WIC/quality/coverage support"
    else:
        annotations = load_annotations(arguments.annotations, data_root=arguments.annotation_data_root)
        completed_orbits = set(annotations["records"])
        files = [path for path in files if int(path.stem[3:]) in completed_orbits]
        if not files:
            raise FileNotFoundError("no Product-2 files match completed manual annotation orbits")
        if arguments.orbits is not None:
            unreviewed = sorted(set(arguments.orbits) - {int(path.stem[3:]) for path in files})
            if unreviewed:
                raise ValueError(f"requested orbits lack completed manual annotations: {unreviewed}")
        missing_orbits = sorted(completed_orbits - {int(path.stem[3:]) for path in files})
        mask_source = {
            "kind": "manual", "annotations": annotations["provenance"],
            "missing_product_orbits": missing_orbits
        }
        estimated_frames = sum(annotations["records"][int(path.stem[3:])]["n_frames"] for path in files)
        alignment = "nearest reviewed source frame within 1 second; full/partial manual mask AND native WIC/quality/coverage support"
    split = split_orbit_files(files, arguments.validation_fraction, arguments.test_fraction, arguments.seed)

    upper_bytes = estimated_frames * 46 * 46 * 13
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    free_bytes = shutil.disk_usage(ancestor).free
    print(f"WIC cache upper estimate: {upper_bytes / 1024**3:.2f} GiB; free disk: {free_bytes / 1024**3:.2f} GiB.", flush=True)
    if free_bytes < upper_bytes * 1.1:
        raise OSError("not enough free disk for the WIC cache estimate plus 10% margin")
    output.mkdir(parents=True)

    rows = []
    count = 0
    total = 0.0
    squared_total = 0.0
    training_support = np.zeros((46, 46), dtype=bool)
    reference_grid = None
    reference_metadata = None
    negative_wic_clipped_cells = 0
    used_counts = {name: 0 for name in split}
    progress = tqdm(total=len(files), desc="Prepare WIC CVAE cache", unit="orbit", dynamic_ncols=True)
    for split_name, paths in split.items():
        for filename in paths:
            if masks is not None:
                product = load_masked_wic(filename, masks, records[filename.stem], product_loader)
            else:
                product = load_annotated_wic(filename, annotations, product_loader)
            if reference_grid is None:
                reference_grid = product["grid"]
                reference_metadata = product["metadata"]
            for name in ("mlat", "mlt"):
                difference = product["grid"][name] - reference_grid[name]
                if name == "mlt":
                    difference = (difference + 12) % 24 - 12
                if not np.allclose(difference, 0, atol=1e-6):
                    raise ValueError(f"{filename}: WIC grids differ between orbits")
            for name in ("grid_id", "count_source", "source_preprocessing_label", "method_quality_weight_method"):
                if product["metadata"][name] != reference_metadata[name]:
                    raise ValueError(f"{filename}: incompatible Product-2 {name}")
            negative_wic_clipped_cells += product["metadata"]["negative_wic_clipped_cells"]

            indices = np.flatnonzero(product["positive"])
            common = product["valid"][indices]
            eligible = product["frame_missing_fraction"][indices] <= arguments.maximum_missing_fraction
            used = int(eligible.sum())
            row = {
                "orbit": int(filename.stem[3:]), "split": split_name, "source": str(filename),
                "product": product["metadata"], "mask_selection": product["mask_evidence"],
                "input_frames": len(product["time"]), "prepared_frames": len(indices),
                "used_frames": used, "intrinsic_missing_dropped_frames": int((~eligible).sum()), "shard": None
            }
            if len(indices):
                shard = output / filename.stem
                shard.mkdir()
                values = product["values"][indices]
                errors = product["standard_deviation"][indices]
                quality = product["quality"][indices, None]
                physical = np.concatenate((values, errors, quality), axis=1).astype(np.float32)
                np.save(shard / "physical.npy", physical, allow_pickle=False)
                np.save(shard / "valid.npy", common, allow_pickle=False)
                selection_fields = {}
                if "annotation_class" in product:
                    selection_fields = {
                        "annotation_class": product["annotation_class"][indices],
                        "annotation_source_frame": product["annotation_source_frame"][indices]
                    }
                np.savez(
                    shard / "frames.npz", time=product["time"][indices], source_frame=indices,
                    wic_source_index=product["wic_source_index"][indices], kp=product["kp"][indices],
                    kp_interval_start=product["kp_interval_start"][indices], eligible=eligible,
                    **selection_fields
                )
                row["shard"] = filename.stem
                if split_name == "train":
                    logarithms = np.log10(values[:, 0][common] + 1.0).astype(np.float64)
                    count += logarithms.size
                    total += logarithms.sum()
                    squared_total += np.sum(logarithms**2)
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
            raise ValueError(f"{name} split has no eligible WIC images")
    if count == 0:
        raise ValueError("no valid training WIC pixels for normalization")
    if negative_wic_clipped_cells:
        warnings.warn(
            f"Product 2 contained {negative_wic_clipped_cells} finite negative corrected-WIC cells; clipped them to zero before log transformation",
            RuntimeWarning, stacklevel=2
        )
    center = total / count
    spread = np.sqrt(max(squared_total / count - center**2, 0.0))
    if not np.isfinite(spread) or spread <= 0:
        raise ValueError("training WIC log spread must be finite and positive")
    transform = ChannelTransform([1.0], [center], [spread], kind=ChannelTransform.LOG10_KIND)
    np.save(output / "training_support.npy", training_support, allow_pickle=False)
    metadata = {
        "cache_format": WIC_CACHE_FORMAT, "input_product_type": "precipitation_cs",
        "field_names": ["wic_corrected"], "field_units": ["counts"],
        "physical_layout": {"values": [0], "standard_deviation": [1], "quality": 2},
        "source_directory": str(source),
        "mask_source": mask_source,
        "alignment": alignment,
        "quality_weight_source": "method_quality_weight_times_wic_coverage",
        "maximum_missing_fraction": arguments.maximum_missing_fraction,
        "minimum_valid_fraction": 1.0 - arguments.maximum_missing_fraction,
        "seed": arguments.seed, "validation_fraction": arguments.validation_fraction,
        "test_fraction": arguments.test_fraction,
        "normalization_population": "positive selected-mask training frames before intrinsic missing-fraction cut; float64 log moments",
        "input_negative_clipping": {"wic_corrected_cells": negative_wic_clipped_cells, "policy": "finite values below zero clipped to physical zero"},
        "transform": transform.to_dict(), "grid": {name: values.tolist() for name, values in reference_grid.items()},
        "grid_id": reference_metadata["grid_id"],
        "split_files": {name: [str(path) for path in paths] for name, paths in split.items()},
        "used_frames": used_counts, "orbits": rows
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Prepared WIC frames {used_counts}; cache: {output}", flush=True)
    return metadata


#%% CLI

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--masks", type=Path, help="exported automatic-mask directory")
    selection.add_argument("--annotations", type=Path, help="completed schema-2 manual annotation JSON")
    parser.add_argument("--annotation-data-root", type=Path, help="machine-local IMAGE_FUV root used to relocate paths stored in the annotation JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orbits", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.05)
    arguments = parser.parse_args(argv)
    if arguments.annotation_data_root is not None and arguments.annotations is None:
        parser.error("--annotation-data-root requires --annotations")
    if not np.isfinite(arguments.minimum_valid_fraction) or not 0 <= arguments.minimum_valid_fraction <= 1:
        parser.error("--minimum-valid-fraction must lie within 0--1")
    arguments.maximum_missing_fraction = 1.0 - arguments.minimum_valid_fraction
    return arguments


if __name__ == "__main__":
    prepare(parse_arguments())
