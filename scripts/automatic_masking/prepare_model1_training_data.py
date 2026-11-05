"""Inventory IMAGE maps and freeze the manual-mask input-selection experiment.

Run from the repository root; this prepares data and never trains a model.
"""

#%% Fixed reference paths and field provenance groups

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.automatic_masking.model2_unet import prepare_inputs

SOURCE = Path.home() / "Dropbox/work/data/IMAGE_FUV/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2"
MANUAL = ROOT / "data/training_data/manual_dataset_c272e0b57c36"
REFERENCE = ROOT / "figures/validation/wic_mask_unet_fullgrid_20261012"
CORE = ["wic_corrected", "method_quality_weight", "sza", "dza"]
GROUPS = {
    "wic_support": ["method_quality_weight", "wic_coverage", "dwic_corrected"],
    "geometry": ["sza", "dza", "grid/xi", "grid/eta", "grid/mlat", "grid/mlt"],
    "other_fuv": ["si12", "si13_corrected", "dsi12", "dsi13_corrected", "si12_coverage", "si13_coverage"],
    "derived": ["Ep_model", "Ep", "Fp", "R", "E0", "Fe", "dEp", "dFp", "dR", "dE0", "dFe", "varE0Fe", "source_count", "uncertainty_source_count", "coverage", "uncertainty_coverage", "method_valid", "method_uncertainty_valid", "Ep_clipping_fraction", "Ep_clipping_any"]
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(source):
    return [dict(file=p.name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sorted(source.glob("or_*.nc"))]


def physical_transform(values, field):
    """Fixed unit scales precede training-only robust standardization."""
    finite = np.isfinite(values)
    result = values.copy()
    result[~finite] = np.nan
    description = "identity"
    if field in ("wic_corrected", "si12", "si13_corrected"):
        result = np.arcsinh(result / 100)
        description = "asinh(counts/100)"
    elif field in ("dwic_corrected", "dsi12", "dsi13_corrected"):
        if np.any(result[finite] < 0):
            raise ValueError(f"Negative measurement standard deviation: {field}")
        result = np.log1p(result / 10)
        description = "log1p(std_counts/10)"
    elif field == "sza":
        result /= 180
        description = "sza/180"
    elif field == "dza":
        result /= 90
        description = "dza/90"
    elif field == "grid/mlat":
        result /= 90
        description = "mlat/90"
    elif field == "grid/mlt_sin":
        result = np.sin(2 * np.pi * result / 24)
        description = "sin(2*pi*MLT_hours/24)"
    elif field == "grid/mlt_cos":
        result = np.cos(2 * np.pi * result / 24)
        description = "cos(2*pi*MLT_hours/24)"
    elif field in ("source_count", "uncertainty_source_count"):
        if np.any(result[finite] < 0):
            raise ValueError(f"Negative support count: {field}")
        result = np.log1p(result)
        description = "log1p(count)"
    elif field in ("Ep_model", "Ep", "Fp", "R", "E0", "Fe", "dEp", "dFp", "dR", "dE0", "dFe"):
        # Preserve signed background-subtracted/retrieved values without clipping.
        result = np.arcsinh(result)
        description = "asinh(value/unit_scale): 1 keV, 1 mW m-2, or 1 dimensionless"
    elif field == "varE0Fe":
        result = np.arcsinh(result)
        description = "asinh(covariance/(1 keV*mW m-2))"
    result[~finite] = np.nan
    return result, finite, description


#%% Validate the manual targets, exact frame identities and held-out boundaries

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    annotations = ROOT / "data/partial_image_annotations_46x46.json"
    provenance = json.loads((MANUAL / "dataset_provenance.json").read_text())
    if sha(annotations) != provenance["annotation_sha256"]:
        raise ValueError("Manual annotation snapshot changed; make a new protocol, do not reuse this plan's dataset")
    for entry in provenance["sources"]:
        stat = Path(entry["file"]).stat()
        if stat.st_size != entry["bytes"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError(f"Reference source changed: {entry['file']}")
    with (MANUAL / "frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (REFERENCE / "frames.csv").open() as stream:
        reference_rows = list(csv.DictReader(stream))
    identities = [tuple(r[k] for k in ("orbit", "frame", "time", "label", "segment")) for r in rows]
    if identities != [tuple(r[k] for k in ("orbit", "frame", "time", "label", "segment")) for r in reference_rows]:
        raise ValueError("Ordered reference frame identities do not match")
    summary = json.loads((REFERENCE / "model/summary.json").read_text())
    days = np.array([r["day"] for r in rows])
    orbits = np.array([int(r["orbit"]) for r in rows])
    test_start = min(r["day"] for r in rows if int(r["orbit"]) in summary["test_orbits"])
    validation_days = {r["day"] for r in rows if int(r["orbit"]) in summary["validation_orbits"]}
    split = dict(train=np.flatnonzero((days < test_start) & ~np.isin(days, list(validation_days))), validation=np.flatnonzero((days < test_start) & np.isin(days, list(validation_days))), test=np.flatnonzero(days >= test_start))
    for name, indices in split.items():
        if len(indices) != summary[f"{name}_frames"] or set(orbits[indices]) != set(summary[f"{name}_orbits"]):
            raise ValueError(f"Frozen {name} split does not match")
    with np.load(MANUAL / "dataset.npz") as archive:
        reference_images = archive["images"]
        targets, domains = archive["targets"], archive["domains"]
    if not np.array_equal(domains, np.isfinite(reference_images[:, 0])):
        raise ValueError("Labels have an old-grid or unexplained restricted loss domain")
    for label, expected in (("full", True), ("bad", False)):
        selected = np.array([r["label"] == label for r in rows])
        if not np.all(targets[selected] == expected):
            raise ValueError(f"Unexpected {label} targets")
    output.mkdir(parents=True, exist_ok=False)
    (output / "prepare_command.txt").write_text("python " + " ".join(sys.argv) + "\n")
    shutil.copyfile(MANUAL / "frames.csv", output / "frames.csv")
    shutil.copyfile(annotations, output / "annotations_snapshot.json")
    np.save(output / "targets.npy", targets)
    np.save(output / "domains.npy", domains)
    write_json(output / "split_ids.json", {k: v.tolist() for k, v in split.items()})
    manifest = source_manifest(SOURCE)
    write_json(output / "source_manifest.json", manifest)

    #%% Stream each source orbit once: schema, availability, values, reviewed cache

    files = [SOURCE / r["file"] for r in manifest]
    with xr.open_dataset(files[0], decode_timedelta=False) as first, xr.open_dataset(files[0], group="grid", decode_timedelta=False) as grid:
        map_fields = [n for n, v in first.data_vars.items() if v.dims == ("time", "dim1", "dim2") and v.dtype.kind in "biuf"]
        map_fields += [f"grid/{n}" for n, v in grid.data_vars.items() if v.dims == ("dim1", "dim2") and v.dtype.kind in "biuf"]
        metadata = {n: dict(units=str(v.attrs.get("units", "")), dimensions=str(v.dims), dtype=str(v.dtype), missing_value_convention="xarray CF decoding; stored fill=" + str(v.encoding.get("_FillValue", "none"))) for n, v in first.data_vars.items()}
        metadata.update({f"grid/{n}": dict(units=str(v.attrs.get("units", "")), dimensions=str(v.dims), dtype=str(v.dtype), missing_value_convention="xarray CF decoding; stored fill=" + str(v.encoding.get("_FillValue", "none"))) for n, v in grid.data_vars.items()})
        source_attrs = {k: str(v) for k, v in first.attrs.items()}
    write_json(output / "source_provenance_example.json", source_attrs)
    raw = np.lib.format.open_memmap(output / "reviewed_maps.npy", mode="w+", dtype="float32", shape=(len(rows), len(map_fields), 46, 46))
    counters = {n: dict(finite=0, total=0, present_files=0, train_min=None, train_max=None, negative=0) for n in map_fields}
    split_finite = {n: {s: 0 for s in split} for n in map_fields}
    train_set = set(split["train"])
    by_orbit = {}
    for index, row in enumerate(rows):
        by_orbit.setdefault(int(row["orbit"]), []).append(index)
    loaded_rows = np.zeros(len(rows), dtype=bool)
    total_frames = 0
    for file_index, filename in enumerate(files):
        orbit = int(filename.stem[3:])
        selected = np.array(by_orbit.get(orbit, []), dtype=int)
        frame_indices = np.array([int(rows[i]["frame"]) for i in selected])
        with xr.open_dataset(filename, decode_timedelta=False) as product, xr.open_dataset(filename, group="grid", decode_timedelta=False) as grid:
            count = product.sizes["time"]
            total_frames += count
            current_maps = {n for n, v in product.data_vars.items() if v.dims == ("time", "dim1", "dim2") and v.dtype.kind in "biuf"}
            current_maps |= {f"grid/{n}" for n, v in grid.data_vars.items() if v.dims == ("dim1", "dim2") and v.dtype.kind in "biuf"}
            if current_maps - set(map_fields):
                raise ValueError(f"New map fields in {filename}; inventory schema must be expanded before training")
            if len(selected):
                actual_times = np.asarray(product.time.values[frame_indices], dtype="datetime64[us]")
                expected_times = np.array([rows[i]["time"] for i in selected], dtype="datetime64[us]")
                if not np.array_equal(actual_times, expected_times):
                    raise ValueError(f"Exposure alignment failed: {filename}")
            for channel, field in enumerate(map_fields):
                entry = counters[field]
                entry["total"] += count * 46 * 46
                if field not in current_maps:
                    values = np.full((count, 46, 46), np.nan, dtype=np.float32)
                else:
                    entry["present_files"] += 1
                    if field.startswith("grid/"):
                        image = np.asarray(grid[field[5:]].values, dtype=np.float32)
                        values = np.broadcast_to(image, (count, 46, 46))
                    else:
                        values = np.asarray(product[field].values, dtype=np.float32)
                finite = np.isfinite(values)
                entry["finite"] += int(finite.sum())
                entry["negative"] += int((finite & (values < 0)).sum())
                if len(selected):
                    reviewed = values[frame_indices]
                    raw[selected, channel] = reviewed
                    for name, indices in split.items():
                        local = np.isin(selected, indices)
                        split_finite[field][name] += int(np.isfinite(reviewed[local]).sum())
                    local_train = np.array([i in train_set for i in selected])
                    pixels = reviewed[local_train]
                    pixels = pixels[np.isfinite(pixels)]
                    if len(pixels):
                        low, high = float(pixels.min()), float(pixels.max())
                        entry["train_min"] = low if entry["train_min"] is None else min(entry["train_min"], low)
                        entry["train_max"] = high if entry["train_max"] is None else max(entry["train_max"], high)
                    if field in provenance["fields"] and not np.array_equal(reviewed, reference_images[selected, provenance["fields"].index(field)], equal_nan=True):
                        raise ValueError(f"Public-reader/cache map mismatch: {field} orbit{orbit}")
            loaded_rows[selected] = True
        if file_index % 100 == 0 or file_index + 1 == len(files):
            print(f"Inventory {file_index + 1}/{len(files)} orbit files; {total_frames} frames", flush=True)
    if not loaded_rows.all() or source_manifest(SOURCE) != manifest:
        raise ValueError("Missing reviewed rows or source mutation during preparation")
    raw.flush()
    np.save(output / "wic.npy", raw[:, map_fields.index("wic_corrected")])

    #%% Eligibility, physical transforms and train-only normalization

    inventory = []
    eligible = []
    group_lookup = {field: group for group, fields in GROUPS.items() for field in fields}
    evidence = {"anchor": "icBuilder/icbuilder/precipitationdetector.py:666; precipitationcs.py:201", "wic_support": "precipitationdetector.py:714; precipitationcs.py:201/224", "geometry": "precipitationcs.py:201; static source grid group", "other_fuv": "precipitationdetector.py:644/666; precipitationcs.py:201/224", "derived": "precipitationdetector.py:264/685; precipitationcs.py:201/224/233"}
    for field, attributes in metadata.items():
        record = dict(field=field, **attributes, group=group_lookup.get(field, "anchor" if field == "wic_corrected" else ""), eligible=False, reason="Not a numeric 46x46 source map")
        record["derivation_evidence"] = evidence.get(record["group"], "Non-image source metadata")
        if field in counters:
            entry = counters[field]
            record.update(entry)
            record["corpus_finite_fraction"] = entry["finite"] / entry["total"]
            record.update({f"{s}_finite_fraction": split_finite[field][s] / (len(split[s]) * 2116) for s in split})
            record["reason"] = "Pre-annotation upstream IMAGE field; available in every source orbit"
            record["eligible"] = (field in group_lookup or field == "wic_corrected") and entry["present_files"] == len(files) and entry["train_min"] is not None and entry["train_max"] > entry["train_min"]
            if not record["eligible"]:
                record["reason"] = "Absent/constant in training, incomplete source availability, or outside fixed groups"
            else:
                eligible.append(field)
        inventory.append(record)
    if not set(CORE) <= set(eligible):
        raise ValueError("Mandatory baseline fields failed eligibility")
    order = ["wic_corrected"] + sorted(set(eligible) - {"wic_corrected"}, key=lambda f: (group_lookup[f], f))
    fields = []
    parents = []
    for field in order:
        expanded = ["grid/mlt_sin", "grid/mlt_cos"] if field == "grid/mlt" else [field]
        fields.extend(expanded)
        parents.extend([field] * len(expanded))
    inputs = np.lib.format.open_memmap(output / "inputs.npy", mode="w+", dtype="float32", shape=(len(rows), 2 * len(fields), 46, 46))
    sample = np.random.default_rng(12026).choice(split["train"], min(512, len(split["train"])), replace=False)
    normalization = dict(fields=fields, parent_fields=parents, centre=[], scale=[], transforms=[], fit_sample_indices=sample.tolist(), finite_flags="all value channels followed by corresponding finite flags", clip=[-6, 6])
    for channel, (field, parent) in enumerate(zip(fields, parents)):
        values, observed, description = physical_transform(raw[:, map_fields.index(parent)], field)
        pixels = values[sample]
        pixels = pixels[np.isfinite(pixels)]
        low, median, high = np.percentile(pixels, [10, 50, 90])
        scale = float(max(high - low, 0.05))
        normalized = np.nan_to_num(np.clip((values - median) / scale, -6, 6), nan=0)
        inputs[:, channel] = normalized
        inputs[:, channel + len(fields)] = observed
        normalization["centre"].append(float(median))
        normalization["scale"].append(scale)
        normalization["transforms"].append(description)
    inputs.flush()
    expected, _ = prepare_inputs(reference_images[:, [provenance["fields"].index(f) for f in CORE]], split["train"], CORE)
    core_channels = [fields.index(f) for f in CORE]
    core_channels += [i + len(fields) for i in core_channels]
    for start in range(0, len(rows), 256):
        if not np.array_equal(inputs[start:start + 256, core_channels], expected[start:start + 256]):
            raise ValueError("CORE normalization differs from existing prepare_inputs")
    write_json(output / "normalization.json", normalization)
    keys = list(dict.fromkeys(k for record in inventory for k in record))
    with (output / "field_inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(inventory)

    #%% Predeclare candidate sets and fixed examples before seeing predictions

    candidates = {"CORE": CORE, "ALL": order}
    for group in GROUPS:
        if any(group_lookup.get(f) == group for f in order):
            candidates[f"ALL_without_{group}"] = [f for f in order if group_lookup.get(f) != group]
    for field in ("dza", "sza"):
        candidates[f"ALL_without_{field}"] = [f for f in order if f != field]
    unique, skipped = {}, {}
    for name, parents_ in candidates.items():
        duplicate = next((n for n, p in unique.items() if set(p) == set(parents_)), None)
        if duplicate:
            skipped[name] = duplicate
        else:
            unique[name] = parents_
    examples = []
    for label in ("full", "bad", "partial"):
        choices = np.array([i for i in split["test"] if rows[i]["label"] == label])
        missing = 1 - domains[choices].mean(axis=(1, 2))
        for quantile in (0.1, 0.5, 0.9):
            ordered = np.lexsort((choices, np.abs(missing - np.quantile(missing, quantile))))
            examples.append(int(next(i for i in choices[ordered] if i not in examples)))
    protocol = dict(annotation_sha256=sha(annotations), baseline_checkpoint_sha256=sha(ROOT / "data/training_data/U_net_weight_0p1.pt"), frames_sha256=sha(output / "frames.csv"), source=str(SOURCE), source_manifest_sha256=sha(output / "source_manifest.json"), frame_counts={k: len(v) for k, v in split.items()}, source_files=len(files), source_frames=total_frames, candidates=unique, skipped_duplicates=skipped, groups=GROUPS, seeds=[12026, 12027, 12028], metric="J: equal mean full_lost/rejected_kept/partial_lost/partial_kept; frame then equal orbit mean", thresholds=np.linspace(0.1, 0.9, 33).tolist(), model="SmallUNet ordinary convolution, single frame, no class head", training=dict(max_epochs=40, patience=8, minimum_improvement=0.0001, batch=64, learning_rate=0.001, weight_decay=0.0001, topology_weight=0.1, checkpoint="minimum class/orbit/segment-weighted validation BCE+Dice+topology", quality="input only"), fixed_test_examples=examples, mlt="static source hours represented by sin/cos, each with a finite flag", uncertainty_transform="signed asinh for derived products preserves unexpected negative source values; std/count negativity raises", software=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, xarray=xr.__version__), caveat="Previously inspected holdout; exploratory, not independent generalization confirmation")
    protocol["training"]["shuffle"] = "independent torch.Generator seeded identically per seed across field sets"
    protocol["training"]["topology_evaluation"] = "same GUDHI persistence_pairs and squared loss, batched CPU transfer/index_add; loss+gradient equivalence checked"
    write_json(output / "protocol.json", protocol)
    del raw
    (output / "reviewed_maps.npy").unlink()
    print(f"Prepared {len(fields)} value maps + finite flags; {len(unique)} screening candidates at {output}", flush=True)


if __name__ == "__main__":
    main()
