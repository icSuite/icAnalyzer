"""Train the single-frame Model1 whole-frame acceptance CNN."""

#%% Imports and frozen reference paths

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.automatic_masking.model1_cnn import (
    acceptance_labels,
    channels_for,
    classifier_scores,
    train_acceptance
)
from scripts.automatic_masking.prepare_model1_training_data import CORE, sha, write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREPARED = ROOT / "figures/validation/wic_input_selection_20261014"
DEFAULT_MODEL2 = ROOT / "data/training_data/U_net_weight_0p1.pt"
DEFAULT_OUTPUT = ROOT / "figures/validation/wic_model1_acceptance_20261105"
ACCEPT_SEED = 12027


def release_metrics(passed, effective, original, orbits):
    accepted = effective == "full"
    return {
        "passed": int(passed.sum()),
        "accepted": int((passed & accepted).sum()),
        "partial": int((passed & (effective == "partial")).sum()),
        "bad": int((passed & (effective == "bad")).sum()),
        "accepted_total": int(accepted.sum()),
        "accepted_recall": float(passed[accepted].mean()),
        "reclassified_partial_released": int((passed & accepted & (original == "partial")).sum()),
        "passed_orbits": int(len(np.unique(orbits[passed])))
    }


#%% One reproducible Model1 acceptance workflow

def run(arguments):
    prepared = arguments.prepared.expanduser().resolve()
    model2_checkpoint = arguments.model2.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh Model1 output directory")

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reference = json.loads((prepared / "protocol.json").read_text())
    annotation = ROOT / "data/partial_image_annotations_46x46.json"
    if sha(model2_checkpoint) != reference["baseline_checkpoint_sha256"]:
        raise ValueError("Model2 checkpoint differs from the prepared Model1 population")
    if sha(prepared / "frames.csv") != reference["frames_sha256"]:
        raise ValueError("Model1 frame table differs from the prepared population")
    if sha(annotation) != reference["annotation_sha256"]:
        raise ValueError("Manual annotations differ from the prepared Model1 population")

    rows = list(csv.DictReader((prepared / "frames.csv").open()))
    split = {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in json.loads((prepared / "split_ids.json").read_text()).items()
    }
    original_labels = np.asarray([row["label"] for row in rows])
    targets = np.load(prepared / "targets.npy", mmap_mode="r")
    domains = np.load(prepared / "domains.npy", mmap_mode="r")
    effective_labels = acceptance_labels(original_labels, targets, domains)
    corrected_rows = [
        dict(row, label=str(label), segment=row["segment"] + ":" + str(label))
        for row, label in zip(rows, effective_labels)
    ]
    orbit_sets = {
        name: {int(rows[index]["orbit"]) for index in indices}
        for name, indices in split.items()
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    if any(orbit_sets[left] & orbit_sets[right] for left, right in pairs):
        raise ValueError("Model1 orbit splits overlap")

    model2 = torch.load(model2_checkpoint, map_location="cpu", weights_only=False)
    normalization = model2["normalization"]
    prepared_inputs = np.load(prepared / "inputs.npy", mmap_mode="r")
    prepared_normalization = json.loads((prepared / "normalization.json").read_text())
    channels = channels_for(CORE, prepared_normalization)

    output.mkdir(parents=True)
    protocol = {
        "target_rule": "Manual Full or manual Partial retaining every observed finite WIC pixel is Accepted",
        "annotation_sha256": reference["annotation_sha256"],
        "frames_sha256": reference["frames_sha256"],
        "split_sha256": sha(prepared / "split_ids.json"),
        "model2_sha256": sha(model2_checkpoint),
        "seed": ACCEPT_SEED,
        "architecture": "8-channel single-frame whole-frame acceptance CNN",
        "selection": "Validation Accepted recall >=80%; no temporal veto or second Model1 network"
    }
    write_json(output / "protocol.json", protocol)
    np.savez_compressed(
        output / "frame_targets.npz",
        original_labels=original_labels,
        effective_labels=effective_labels
    )

    accept_model, accept_record = train_acceptance(
        ACCEPT_SEED, output, prepared_inputs, channels, effective_labels,
        corrected_rows, split, normalization, "cpu"
    )

    validation = split["validation"]
    validation_probability = classifier_scores(accept_model, prepared_inputs, channels, validation)
    sorted_accepted = np.sort(validation_probability[effective_labels[validation] == "full"])[::-1]
    cutoff = float(sorted_accepted[int(np.ceil(.8 * len(sorted_accepted))) - 1])
    validation_passed = validation_probability >= cutoff
    validation_orbits = np.asarray([int(rows[index]["orbit"]) for index in validation])
    selection = {
        "full_cutoff": cutoff,
        "rule": "Lowest validation score retaining at least 80% of effective Accepted frames",
        "validation": release_metrics(
            validation_passed, effective_labels[validation],
            original_labels[validation], validation_orbits
        )
    }
    write_json(output / "selection_before_test.json", selection)
    np.savez_compressed(
        output / "validation_scores.npz",
        full_score=validation_probability,
        labels=original_labels[validation],
        effective_labels=effective_labels[validation],
        orbits=validation_orbits,
        dataset_indices=validation
    )

    test = split["test"]
    test_probability = classifier_scores(accept_model, prepared_inputs, channels, test)
    test_passed = test_probability >= cutoff
    test_orbits = np.asarray([int(rows[index]["orbit"]) for index in test])
    test_metrics = release_metrics(
        test_passed, effective_labels[test], original_labels[test], test_orbits
    )
    np.savez_compressed(
        output / "test_scores.npz",
        full_score=test_probability,
        labels=original_labels[test],
        effective_labels=effective_labels[test],
        orbits=test_orbits,
        dataset_indices=test
    )

    accept_checkpoint = output / f"seed{ACCEPT_SEED}/frame_classifier.pt"
    verification = {
        "accept_checkpoint_sha256": sha(accept_checkpoint),
        "model2_sha256": sha(model2_checkpoint)
    }
    write_json(output / "verification.json", verification)
    report = {
        "training": accept_record,
        "validation": selection,
        "test": test_metrics,
        "verification": verification
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--model2", type=Path, default=DEFAULT_MODEL2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
