"""Prepare the adopted Model1 whole-frame classifier from Model2 training data."""

#%% Imports and shared provenance helpers

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from scripts.automatic_masking.model2_unet import prepare_inputs


CORE = ["wic_corrected", "method_quality_weight", "sza", "dza"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def physical_transform(values, field):
    """Apply the adopted Model2 physical transform to one input field."""

    values = np.asarray(values, dtype=np.float32).copy()
    observed = np.isfinite(values)
    description = field
    if field == "wic_corrected":
        values = np.arcsinh(values / 100)
        description = "asinh(WIC/100)"
    elif field == "dwic_corrected":
        values = np.log1p(np.maximum(values, 0) / 10)
        description = "log1p(std/10)"
    elif field in ("sza", "dza"):
        divisor = 180 if field == "sza" else 90
        values /= divisor
        description = f"{field}/{divisor}"
    values[~observed] = np.nan
    return values, observed, description


def run(arguments):
    prepared_model2 = arguments.prepared_model2.expanduser().resolve()
    model2_checkpoint = arguments.model2.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("choose a fresh Model1 preparation directory")

    provenance = json.loads((prepared_model2 / "dataset_provenance.json").read_text())
    annotation_snapshot = prepared_model2 / "annotations_snapshot.json"
    if sha(annotation_snapshot) != provenance["annotation_sha256"]:
        raise ValueError("Model2 annotation snapshot differs from its provenance")
    model2 = torch.load(model2_checkpoint, map_location="cpu", weights_only=False)
    if model2["input_fields"] != CORE:
        raise ValueError("Model2 checkpoint does not use the adopted four input fields")
    if model2["manual_annotation_sha256"] != provenance["annotation_sha256"]:
        raise ValueError("Model2 checkpoint and prepared annotations differ")

    rows = list(csv.DictReader((prepared_model2 / "frames.csv").open()))
    split = {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in json.loads((prepared_model2 / "split_ids.json").read_text()).items()
    }
    with np.load(prepared_model2 / "dataset.npz") as archive:
        images = archive["images"]
        targets = archive["targets"]
        domains = archive["domains"]
    images = images[:, [provenance["fields"].index(name) for name in CORE]]
    inputs, normalization = prepare_inputs(images, split["train"], CORE)
    normalization["parent_fields"] = list(CORE)
    checkpoint_normalization = model2["normalization"]
    for name in ("fields", "centre", "scale"):
        if normalization[name] != checkpoint_normalization[name]:
            raise ValueError(f"Model1 preparation and Model2 {name} normalization differ")

    orbit_sets = {
        name: {int(rows[index]["orbit"]) for index in indices}
        for name, indices in split.items()
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    if any(orbit_sets[left] & orbit_sets[right] for left, right in pairs):
        raise ValueError("Model1 orbit splits overlap")

    output.mkdir(parents=True)
    (output / "prepare_command.txt").write_text("python " + " ".join(sys.argv) + "\n")
    shutil.copyfile(prepared_model2 / "frames.csv", output / "frames.csv")
    shutil.copyfile(prepared_model2 / "split_ids.json", output / "split_ids.json")
    shutil.copyfile(annotation_snapshot, output / "annotations_snapshot.json")
    np.save(output / "inputs.npy", inputs)
    np.save(output / "targets.npy", targets)
    np.save(output / "domains.npy", domains)
    write_json(output / "normalization.json", normalization)
    protocol = {
        "annotation_sha256": provenance["annotation_sha256"],
        "prepared_model2": str(prepared_model2),
        "prepared_model2_provenance_sha256": sha(prepared_model2 / "dataset_provenance.json"),
        "baseline_checkpoint": str(model2_checkpoint),
        "baseline_checkpoint_sha256": sha(model2_checkpoint),
        "frames_sha256": sha(output / "frames.csv"),
        "split_sha256": sha(output / "split_ids.json"),
        "frame_counts": {name: len(indices) for name, indices in split.items()},
        "orbit_counts": {name: len(orbits) for name, orbits in orbit_sets.items()},
        "fields": CORE,
        "model": "Small single-frame whole-frame acceptance CNN"
    }
    write_json(output / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2), flush=True)
    return protocol


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-model2", type=Path, required=True)
    parser.add_argument("--model2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
