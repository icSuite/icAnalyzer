"""Prepare a paired corrected-WIC/SI12 CVAE cache.

The scientific selection mask is always the existing manual or automatic WIC
mask.  Each sensor then retains its own finite-value, uncertainty and coverage
support.  The cache stores both formal standard deviations for later loss
experiments, but cache preparation never trains a model.
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

from icanalyzer.cvae.cache import (
    WIC_SI12_CACHE_FORMAT,
    file_sha256,
    read_cache,
)
from icanalyzer.cvae.data import ChannelTransform, split_orbit_files
from scripts.CVAE.prepare_wic_cvae_cache import (
    DEFAULT_SOURCE,
    GRID_ID,
    _source_evidence,
)
from scripts.manual_masking.annotated_data import (
    annotation_mask_arrays,
    load_annotations,
)


FIELD_NAMES = ("wic_corrected", "si12")


def read_wic_si12_cache(cache_directory):
    return read_cache(
        Path(cache_directory), expected_format=WIC_SI12_CACHE_FORMAT
    )


def _load_fields(filename, product_loader=load):
    """Load paired fields without applying the WIC selection mask."""

    with product_loader(filename) as product:
        if (
            product.product_type != "precipitation_cs"
            or product.grid_id != GRID_ID
            or tuple(product.shape[1:]) != (46, 46)
        ):
            raise ValueError(
                f"{filename}: expected precipitation_cs on {GRID_ID}"
            )
        required = (
            "wic_corrected", "dwic_corrected", "si12", "dsi12",
            "method_valid", "method_quality_weight",
            "wic_coverage", "si12_coverage", "Kp"
        )
        missing = [name for name in required if not hasattr(product, name)]
        if missing:
            raise ValueError(
                f"{filename}: missing paired-CVAE fields: {', '.join(missing)}"
            )
        quality_method = str(
            product.attrs.get("method_quality_weight_method", "")
        ).lower()
        if "auroral-quality" not in quality_method or "1 minus" not in quality_method:
            raise ValueError(
                f"{filename}: ambiguous Product-2 auroral-quality orientation"
            )
        if str(product.attrs.get("proton_flux_source", "")) != "SI12":
            raise ValueError(f"{filename}: expected proton_flux_source='SI12'")

        time = np.asarray(product.time, dtype="datetime64[us]")
        arrays = {
            name: np.asarray(product.read(name)) for name in required
            if name != "Kp"
        }
        kp = np.asarray(product.read("Kp"), dtype=np.float32)
        kp_interval_start = np.asarray(
            product.Kp_interval_start, dtype="datetime64[us]"
        )
        wic_source_index = np.asarray(
            product.source_indices["wic"], dtype=np.int32
        )
        grid = {
            name: np.asarray(getattr(product, name), dtype=np.float64).copy()
            for name in ("mlat", "mlt")
        }
        metadata = {
            "orbit": int(Path(filename).stem[3:]),
            "product_type": str(product.product_type),
            "schema_version": int(product.schema_version),
            "software_version": str(product.software_version),
            "grid_id": str(product.grid_id),
            "count_source": str(
                product.attrs.get("count_source", "unrecorded")
            ),
            "source_preprocessing_label": str(
                product.attrs.get("source_preprocessing_label", "unrecorded")
            ),
            "method_quality_weight_method": str(
                product.attrs["method_quality_weight_method"]
            ),
            "proton_energy_model": str(
                product.attrs.get("proton_energy_model", "unrecorded")
            ),
            "proton_flux_source": str(product.attrs["proton_flux_source"]),
        }

    shape = arrays["wic_corrected"].shape
    if shape[1:] != (46, 46):
        raise ValueError(f"{filename}: paired fields use an unexpected grid")
    if any(array.shape != shape for array in arrays.values()):
        raise ValueError(f"{filename}: paired fields do not share one shape")
    if kp.shape != (len(time),) or kp_interval_start.shape != kp.shape:
        raise ValueError(f"{filename}: Kp metadata does not align with frames")
    if wic_source_index.shape != kp.shape:
        raise ValueError(f"{filename}: WIC source indices do not align")
    if np.any(~np.isfinite(kp)) or np.any((kp < 0) | (kp > 9)):
        raise ValueError(f"{filename}: retained frames require Kp within 0--9")
    finite_quality = arrays["method_quality_weight"][
        np.isfinite(arrays["method_quality_weight"])
    ]
    if np.any((finite_quality < 0) | (finite_quality > 1)):
        raise ValueError(
            f"{filename}: method_quality_weight must lie within 0--1"
        )

    return {
        "arrays": arrays, "time": time, "kp": kp,
        "kp_interval_start": kp_interval_start,
        "wic_source_index": wic_source_index,
        "grid": grid, "metadata": metadata,
    }


def _apply_selection(product, selection_mask, mask_evidence,
                     annotation_classes=None, annotation_source_frames=None):
    """Apply one WIC-derived selection while retaining sensor validity."""

    arrays = product["arrays"]
    shape = arrays["wic_corrected"].shape
    selection_mask = np.asarray(selection_mask, dtype=bool)
    if selection_mask.shape != shape:
        raise ValueError("WIC selection mask must have shape (frame, 46, 46)")

    wic = np.asarray(arrays["wic_corrected"], dtype=np.float32)
    si12 = np.asarray(arrays["si12"], dtype=np.float32)
    dwic = np.asarray(arrays["dwic_corrected"], dtype=np.float32)
    dsi12 = np.asarray(arrays["dsi12"], dtype=np.float32)
    method_valid = np.asarray(arrays["method_valid"], dtype=bool)
    method_quality = np.asarray(
        arrays["method_quality_weight"], dtype=np.float32
    )
    wic_coverage = np.asarray(arrays["wic_coverage"], dtype=np.float32)
    si12_coverage = np.asarray(arrays["si12_coverage"], dtype=np.float32)

    negative_wic = np.isfinite(wic) & (wic < 0)
    negative_si12 = np.isfinite(si12) & (si12 < 0)
    wic = np.maximum(wic, 0.0)
    si12 = np.maximum(si12, 0.0)

    common = (
        selection_mask & method_valid
        & np.isfinite(method_quality) & (method_quality > 0)
    )
    valid_wic = (
        common & np.isfinite(wic) & np.isfinite(dwic) & (dwic >= 0)
        & np.isfinite(wic_coverage) & (wic_coverage > 0)
    )
    valid_si12 = (
        common & np.isfinite(si12) & np.isfinite(dsi12) & (dsi12 >= 0)
        & np.isfinite(si12_coverage) & (si12_coverage > 0)
    )
    valid = np.stack((valid_wic, valid_si12), axis=1)
    coverage = np.stack((wic_coverage, si12_coverage), axis=1)
    quality = np.clip(method_quality[:, None], 0.0, 1.0) * np.clip(
        coverage, 0.0, 1.0
    )
    quality = np.where(valid, quality, 0.0).astype(np.float32)
    values = np.stack((wic, si12), axis=1)
    errors = np.stack((dwic, dsi12), axis=1)
    valid_fraction = np.mean(valid, axis=(2, 3))
    positive = np.all(np.any(valid, axis=(2, 3)), axis=1)

    product["metadata"].update({
        "negative_wic_corrected_clipped_cells": int(negative_wic.sum()),
        "negative_si12_clipped_cells": int(negative_si12.sum()),
    })
    selected = {
        "values": values, "standard_deviation": errors,
        "quality": quality, "valid": valid,
        "valid_fraction": valid_fraction, "positive": positive,
        "selection_mask": selection_mask,
        "time": product["time"], "kp": product["kp"],
        "kp_interval_start": product["kp_interval_start"],
        "wic_source_index": product["wic_source_index"],
        "grid": product["grid"], "metadata": product["metadata"],
        "mask_evidence": mask_evidence,
    }
    if annotation_classes is not None:
        selected["annotation_class"] = annotation_classes
        selected["annotation_source_frame"] = annotation_source_frames
    return selected


def load_masked_wic_si12(filename, mask_directory, mask_record,
                         product_loader=load):
    _source_evidence(filename, mask_record)
    mask_file = Path(mask_directory) / (Path(filename).stem + ".npz")
    with np.load(mask_file, allow_pickle=False) as stored:
        mask_time = stored["time"].astype("datetime64[us]")
        selection_mask = stored["mask"]
    if selection_mask.dtype != bool or selection_mask.shape != (
        mask_record["frames"], 46, 46
    ):
        raise ValueError(f"{mask_file}: expected Boolean frame masks")
    product = _load_fields(filename, product_loader)
    if not np.array_equal(product["time"], mask_time):
        raise ValueError(f"{filename}: automatic masks use different timestamps")
    evidence = {
        "kind": "automatic", "filename": str(mask_file.resolve()),
        "sha256": file_sha256(mask_file),
    }
    return _apply_selection(product, selection_mask, evidence)


def load_annotated_wic_si12(filename, annotations, product_loader=load):
    product = _load_fields(filename, product_loader)
    count = len(product["time"])
    annotation_product = {
        "values": product["arrays"]["wic_corrected"][:, None],
        "time": product["time"],
        "orbit": np.full(count, product["metadata"]["orbit"], dtype=int),
        "source_indices": {"wic": product["wic_source_index"]},
        "grid": product["grid"],
    }
    masks, classes, source_frames, unmatched = annotation_mask_arrays(
        annotation_product, annotations
    )
    evidence = {
        "kind": "manual", "orbit": product["metadata"]["orbit"],
        "selected_full": int(np.sum(classes == "full")),
        "selected_partial": int(np.sum(classes == "partial")),
        "excluded_frames": int(np.sum(classes == "excluded")),
        "unmatched_timestamps": unmatched,
    }
    return _apply_selection(
        product, masks, evidence, classes, source_frames
    )


#%% Disk-backed preparation

def prepare(arguments, product_loader=load):
    source = arguments.source.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            "choose a new WIC/SI12 cache directory; caches are never overwritten"
        )
    minimum_valid = float(arguments.minimum_valid_fraction)

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
        missing = [
            path.stem for path in files
            if path.stem not in records
            or not (masks / (path.stem + ".npz")).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"missing automatic masks for {missing[:8]}")
        mask_source = {
            "kind": "automatic", "directory": str(masks),
            "manifest_sha256": file_sha256(mask_manifest),
            "model_sha256": mask_summary["model_sha256"],
            "threshold": mask_summary["threshold"],
            "frame_gate": mask_summary.get("frame_gate"),
            "mask_meaning": mask_summary.get("mask_meaning"),
        }
        estimated_frames = sum(records[path.stem]["frames"] for path in files)
    else:
        annotations = load_annotations(arguments.annotations)
        completed = set(annotations["records"])
        files = [path for path in files if int(path.stem[3:]) in completed]
        if not files:
            raise FileNotFoundError(
                "no Product-2 files match completed manual annotations"
            )
        if arguments.orbits is not None:
            unreviewed = sorted(
                set(arguments.orbits) - {int(path.stem[3:]) for path in files}
            )
            if unreviewed:
                raise ValueError(
                    f"requested orbits lack completed annotations: {unreviewed}"
                )
        mask_source = {
            "kind": "manual", "annotations": annotations["provenance"],
            "missing_product_orbits": sorted(
                completed - {int(path.stem[3:]) for path in files}
            ),
        }
        estimated_frames = sum(
            annotations["records"][int(path.stem[3:])]["n_frames"]
            for path in files
        )

    split = split_orbit_files(
        files, arguments.validation_fraction, arguments.test_fraction,
        arguments.seed
    )
    upper_bytes = estimated_frames * 46 * 46 * 26
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    free_bytes = shutil.disk_usage(ancestor).free
    print(
        f"WIC/SI12 cache upper estimate: {upper_bytes / 1024**3:.2f} GiB; "
        f"free disk: {free_bytes / 1024**3:.2f} GiB.", flush=True
    )
    if free_bytes < upper_bytes * 1.1:
        raise OSError("not enough free disk for cache estimate plus 10%")
    output.mkdir(parents=True)

    rows = []
    counts = np.zeros(2, dtype=np.int64)
    totals = np.zeros(2, dtype=np.float64)
    squared_totals = np.zeros(2, dtype=np.float64)
    training_support = np.zeros((2, 46, 46), dtype=bool)
    used_counts = {name: 0 for name in split}
    overlap_counts = {
        "selected_cells": 0, "wic_only": 0, "si12_only": 0,
        "both": 0, "neither": 0,
    }
    negative_counts = np.zeros(2, dtype=np.int64)
    reference_grid = None
    reference_metadata = None

    progress = tqdm(
        total=len(files), desc="Prepare WIC/SI12 CVAE cache",
        unit="orbit", dynamic_ncols=True
    )
    for split_name, paths in split.items():
        for filename in paths:
            product = (
                load_masked_wic_si12(
                    filename, masks, records[filename.stem], product_loader
                )
                if masks is not None else
                load_annotated_wic_si12(
                    filename, annotations, product_loader
                )
            )
            if reference_grid is None:
                reference_grid = product["grid"]
                reference_metadata = product["metadata"]
            for name in ("mlat", "mlt"):
                difference = product["grid"][name] - reference_grid[name]
                if name == "mlt":
                    difference = (difference + 12) % 24 - 12
                if not np.allclose(difference, 0, atol=1e-6):
                    raise ValueError(f"{filename}: grids differ between orbits")
            for name in (
                "grid_id", "count_source", "source_preprocessing_label",
                "method_quality_weight_method", "proton_energy_model",
                "proton_flux_source"
            ):
                if product["metadata"][name] != reference_metadata[name]:
                    raise ValueError(f"{filename}: incompatible Product-2 {name}")

            negative_counts += [
                product["metadata"]["negative_wic_corrected_clipped_cells"],
                product["metadata"]["negative_si12_clipped_cells"],
            ]
            selected = product["selection_mask"]
            wic_valid, si12_valid = product["valid"][:, 0], product["valid"][:, 1]
            overlap_counts["selected_cells"] += int(selected.sum())
            overlap_counts["wic_only"] += int(
                np.sum(selected & wic_valid & ~si12_valid)
            )
            overlap_counts["si12_only"] += int(
                np.sum(selected & ~wic_valid & si12_valid)
            )
            overlap_counts["both"] += int(
                np.sum(selected & wic_valid & si12_valid)
            )
            overlap_counts["neither"] += int(
                np.sum(selected & ~wic_valid & ~si12_valid)
            )

            indices = np.flatnonzero(product["positive"])
            valid = product["valid"][indices]
            valid_fraction = product["valid_fraction"][indices]
            eligible = np.all(valid_fraction >= minimum_valid, axis=1)
            used = int(eligible.sum())
            row = {
                "orbit": int(filename.stem[3:]), "split": split_name,
                "source": str(filename), "product": product["metadata"],
                "mask_selection": product["mask_evidence"],
                "input_frames": len(product["time"]),
                "prepared_frames": len(indices), "used_frames": used,
                "minimum_valid_fraction_each_channel": minimum_valid,
                "intrinsic_missing_dropped_frames": int((~eligible).sum()),
                "shard": None,
            }
            if len(indices):
                shard = output / filename.stem
                shard.mkdir()
                values = product["values"][indices]
                errors = product["standard_deviation"][indices]
                quality = product["quality"][indices]
                physical = np.concatenate(
                    (values, errors, quality), axis=1
                ).astype(np.float32)
                np.save(shard / "physical.npy", physical, allow_pickle=False)
                np.save(shard / "valid.npy", valid, allow_pickle=False)
                selection_fields = {}
                if "annotation_class" in product:
                    selection_fields = {
                        "annotation_class": product["annotation_class"][indices],
                        "annotation_source_frame": product[
                            "annotation_source_frame"
                        ][indices],
                    }
                np.savez(
                    shard / "frames.npz",
                    time=product["time"][indices], source_frame=indices,
                    wic_source_index=product["wic_source_index"][indices],
                    kp=product["kp"][indices],
                    kp_interval_start=product["kp_interval_start"][indices],
                    eligible=eligible, **selection_fields
                )
                row["shard"] = filename.stem
                if split_name == "train":
                    for channel in range(2):
                        logarithms = np.log10(
                            values[:, channel][valid[:, channel]] + 1.0
                        ).astype(np.float64)
                        counts[channel] += logarithms.size
                        totals[channel] += logarithms.sum()
                        squared_totals[channel] += np.sum(logarithms**2)
                    if used:
                        training_support |= np.any(valid[eligible], axis=0)
                del values, errors, quality, physical
            used_counts[split_name] += used
            rows.append(row)
            progress.update(1)
            progress.set_postfix(used=sum(used_counts.values()), refresh=False)
            del product, valid
    progress.close()

    for name in split:
        if split[name] and used_counts[name] == 0:
            raise ValueError(f"{name} split has no eligible paired images")
    if np.any(counts == 0):
        raise ValueError("both channels need valid training pixels")
    if np.any(negative_counts):
        warnings.warn(
            "Product 2 contained finite negative paired counts; clipped "
            f"WIC={negative_counts[0]}, SI12={negative_counts[1]} to zero",
            RuntimeWarning, stacklevel=2
        )
    centers = totals / counts
    spreads = np.sqrt(
        np.maximum(squared_totals / counts - centers**2, 0.0)
    )
    if np.any(~np.isfinite(spreads)) or np.any(spreads <= 0):
        raise ValueError("paired training log spreads must be finite and positive")
    transform = ChannelTransform(
        [1.0, 1.0], centers, spreads,
        kind=ChannelTransform.LOG10_KIND
    )
    np.save(output / "training_support.npy", training_support, allow_pickle=False)

    selected_count = max(overlap_counts["selected_cells"], 1)
    overlap = {
        **overlap_counts,
        **{
            f"{name}_fraction_of_selected": overlap_counts[name] / selected_count
            for name in ("wic_only", "si12_only", "both", "neither")
        },
    }
    metadata = {
        "cache_format": WIC_SI12_CACHE_FORMAT,
        "input_product_type": "precipitation_cs",
        "field_names": list(FIELD_NAMES), "field_units": ["counts", "counts"],
        "quality_channel_count": 2, "valid_channel_count": 2,
        "physical_layout": {
            "values": [0, 1], "standard_deviation": [2, 3],
            "quality": [4, 5]
        },
        "uncertainty_arrays_cached": True,
        "source_directory": str(source), "mask_source": mask_source,
        "mask_policy": (
            "same WIC-derived selection for both sensors; separate intrinsic "
            "finite, uncertainty and positive-coverage validity"
        ),
        "quality_weight_source": [
            "method_quality_weight_times_wic_coverage",
            "method_quality_weight_times_si12_coverage",
        ],
        "encoder_mask_policy": "union_with_channel_quality_inputs",
        "minimum_valid_fraction": minimum_valid,
        "minimum_valid_fraction_policy": "required independently in each channel",
        "maximum_missing_fraction": 1.0 - minimum_valid,
        "seed": arguments.seed,
        "validation_fraction": arguments.validation_fraction,
        "test_fraction": arguments.test_fraction,
        "normalization_population": (
            "positive selected-mask training frames before minimum-valid cut; "
            "separate float64 log10(value+1) moments on each channel's validity"
        ),
        "input_negative_clipping": {
            "wic_corrected_cells": int(negative_counts[0]),
            "si12_cells": int(negative_counts[1]),
            "policy": "finite counts below zero clipped to physical zero",
        },
        "sensor_support_overlap": overlap,
        "transform": transform.to_dict(),
        "grid": {
            name: values.tolist() for name, values in reference_grid.items()
        },
        "grid_id": reference_metadata["grid_id"],
        "split_files": {
            name: [str(path) for path in paths] for name, paths in split.items()
        },
        "used_frames": used_counts, "orbits": rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(f"Prepared paired frames {used_counts}; cache: {output}", flush=True)
    return metadata


#%% CLI

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--masks", type=Path)
    selection.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orbits", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.05)
    arguments = parser.parse_args(argv)
    if (
        not np.isfinite(arguments.minimum_valid_fraction)
        or not 0 <= arguments.minimum_valid_fraction <= 1
    ):
        parser.error("--minimum-valid-fraction must lie within 0--1")
    return arguments


if __name__ == "__main__":
    prepare(parse_arguments())
