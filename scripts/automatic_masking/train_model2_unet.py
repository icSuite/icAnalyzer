"""Train the single-frame WIC mask U-Net with topology weight 0.1.

Run from the icAnalyzer repository root with the tConductance environment:
    python scripts/automatic_masking/train_model2_unet.py

GUDHI 3.13.0 has a Python 3.10 wheel, even though Mamba cannot resolve it
against this environment's existing Conda packages. Install that wheel without
changing the other packages:
    python -m pip install --no-deps --only-binary=:all: gudhi==3.13.0

Manual full/rejected reviews and held-forward partial keyframes are targets.
The model sees corrected WIC, method quality, SZA and DZA, plus finite flags.
"""

#%% Paths and the small set of calculations shared with the original experiment

import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gudhi
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.automatic_masking.model2_unet import (
    FrameWindowDataset, SmallUNet, class_scores, masked_loss, pixel_scores,
    predict, prepare_inputs, sample_weights, validation_correction, window_inputs
)
from scripts.automatic_masking.topology import shape_scores, topology_loss

SOURCE = Path.home() / "Dropbox/work/data/IMAGE_FUV/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2"
ANNOTATIONS = ROOT / "data/partial_image_annotations_46x46.json"
DATA = ROOT / "data/training_data"
FIGURES = ROOT / "figures/training_data"
FIELDS = ["wic_corrected", "method_quality_weight", "sza", "dza"]
WEIGHT = 0.1


def main():
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    #%% Turn completed manual reviews into one supervised row per nonempty frame

    annotation_sha = hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest()
    dataset = DATA / f"manual_dataset_{annotation_sha[:12]}"
    if not dataset.exists():
        subprocess.run([
            sys.executable, "-m", "scripts.automatic_masking.prepare_model2_training_data",
            "--annotations", str(ANNOTATIONS), "--source", str(SOURCE), "--output", str(dataset)
        ], cwd=ROOT, check=True)

    provenance = json.loads((dataset / "dataset_provenance.json").read_text())
    if provenance["annotation_sha256"] != annotation_sha:
        raise ValueError("Dataset does not match the current manual annotation JSON")
    for source in provenance["sources"]:
        stat = Path(source["file"]).stat()
        if stat.st_size != source["bytes"] or stat.st_mtime_ns != source["mtime_ns"]:
            raise ValueError(f"Source product changed since dataset creation: {source['file']}")

    with (dataset / "frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with np.load(dataset / "dataset.npz") as archive:
        images = archive["images"]
        targets = archive["targets"]
        domains = archive["domains"]
    images = images[:, [provenance["fields"].index(name) for name in FIELDS]]
    orbits = np.array([int(row["orbit"]) for row in rows])
    days = np.array([row["day"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    classes = np.array([("full", "bad", "partial").index(label) for label in labels])

    #%% Reuse the orbit-day boundaries from the evaluated weight 0.1 experiment

    reference = ROOT / "figures/validation/wic_mask_unet_fullgrid_20261012"
    reference_summary = json.loads((reference / "model/summary.json").read_text())
    with (reference / "frames.csv").open() as stream:
        reference_rows = list(csv.DictReader(stream))
    test_start = min(row["day"] for row in reference_rows if int(row["orbit"]) in reference_summary["test_orbits"])
    validation_days = {row["day"] for row in reference_rows if int(row["orbit"]) in reference_summary["validation_orbits"]}
    train = np.flatnonzero((days < test_start) & ~np.isin(days, list(validation_days)))
    validation = np.flatnonzero((days < test_start) & np.isin(days, list(validation_days)))
    test = np.flatnonzero(days >= test_start)
    assert not (set(orbits[train]) & set(orbits[validation]))
    assert not (set(orbits[train]) & set(orbits[test]))
    assert not (set(orbits[validation]) & set(orbits[test]))

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(12026)
    np.random.seed(12026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, normalization = prepare_inputs(images, train, FIELDS)
    windows = np.arange(len(rows))[:, None]
    train_weights = sample_weights(rows, train)
    validation_weights = sample_weights(rows, validation)
    loader = DataLoader(
        FrameWindowDataset(inputs, windows, train, targets, domains, classes, train_weights),
        batch_size=64, shuffle=True
    )
    model = SmallUNet(inputs.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    print(f"Training {len(train)} frames; validation {len(validation)}; test {len(test)}; device {device}", flush=True)

    #%% BCE and Dice use finite WIC pixels; topology uses the full 46x46 grid

    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(40):
        model.train()
        training_total = 0.0
        for values, target, domain, _, weight in loader:
            optimizer.zero_grad()
            logits = model(values.to(device))
            losses = masked_loss(logits, target.to(device), domain.to(device))
            losses = losses + WEIGHT * topology_loss(logits)
            loss = (losses * weight.to(device)).mean()
            loss.backward()
            optimizer.step()
            training_total += float(loss.detach()) * len(values)

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in np.array_split(validation, max(1, int(np.ceil(len(validation) / 128)))):
                values = window_inputs(inputs, windows, batch)
                logits = model(torch.from_numpy(values).to(device))
                losses = masked_loss(logits, torch.from_numpy(targets[batch].astype(np.float32)).to(device), torch.from_numpy(domains[batch].astype(np.float32)).to(device))
                validation_losses.extend((losses + WEIGHT * topology_loss(logits)).cpu().numpy())
        validation_loss = float(np.average(validation_losses, weights=validation_weights))
        history.append(dict(epoch=epoch + 1, train=training_total / len(train), validation=validation_loss))
        if validation_loss < best_loss - 0.0001:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        print(f"epoch {epoch + 1}: train {history[-1]['train']:.4f}, validation {validation_loss:.4f}, best {best_epoch}", flush=True)
        if epoch + 1 - best_epoch >= 8:
            break

    #%% Pick a threshold on validation, then inspect untouched test orbits once

    model.load_state_dict(best_state)
    validation_probability, _ = predict(model, inputs, validation, device)
    thresholds = np.linspace(0.1, 0.9, 33)
    validation_errors = [validation_correction(pixel_scores(validation_probability >= threshold, targets[validation], domains[validation]), orbits[validation], labels[validation]) for threshold in thresholds]
    threshold = float(thresholds[np.argmin(validation_errors)])
    test_probability, _ = predict(model, inputs, test, device)
    test_mask = test_probability >= threshold
    test_scores = pixel_scores(test_mask, targets[test], domains[test])
    by_class = class_scores(test_scores, orbits[test], labels[test])
    shape = shape_scores(test_mask, domains[test])
    metrics = {
        "manual_annotation_sha256": annotation_sha,
        "source": str(SOURCE),
        "frames": {"train": len(train), "validation": len(validation), "test": len(test)},
        "orbits": {"train": len(np.unique(orbits[train])), "validation": len(np.unique(orbits[validation])), "test": len(np.unique(orbits[test]))},
        "best_epoch": best_epoch, "threshold": threshold, "topology_weight": WEIGHT,
        "validation_disagreement": float(min(validation_errors)),
        "test_by_class": by_class,
        "test_observable_shape_violations": int(shape["any_violation"].sum()),
        "caveat": "Test errors cover manually reviewed orbits only; automatic masks are not manual annotations. Raw masks can violate the shape rule."
    }
    (FIGURES / "training_results.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({"state_dict": best_state, "normalization": normalization, "input_fields": FIELDS, "threshold": threshold, "topology_weight": WEIGHT, "manual_annotation_sha256": annotation_sha, "source": str(SOURCE)}, DATA / "U_net_weight_0p1.pt")
    print("Held-out pixel disagreement:", {name: round(by_class[name]["metrics"]["correction_fraction"]["orbit_mean"], 3) for name in ("full", "bad", "partial")}, flush=True)
    print(f"Raw observable shape violations: {metrics['test_observable_shape_violations']} / {len(test)}", flush=True)

    #%% Two straightforward figures: learning/results and fixed held-out examples

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([row["epoch"] for row in history], [row["train"] for row in history], label="Train")
    axes[0].plot([row["epoch"] for row in history], [row["validation"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Weighted loss", title="Training")
    axes[0].legend()
    names = ("full", "bad", "partial")
    axes[1].bar(names, [100 * by_class[name]["metrics"]["correction_fraction"]["orbit_mean"] for name in names])
    axes[1].set(ylabel="Disagreeing finite-WIC pixels (%)", title="Held-out orbit means")
    figure.tight_layout()
    figure.savefig(FIGURES / "training_summary.png", dpi=160)
    plt.close(figure)

    example_positions = []
    for name in names:
        positions = np.flatnonzero(labels[test] == name)
        example_positions.extend(positions[np.linspace(0, len(positions) - 1, 3, dtype=int)])
    figure, axes = plt.subplots(9, 3, figsize=(9, 22))
    for axis_row, position in zip(axes, example_positions):
        index = test[position]
        wic = images[index, 0]
        finite = wic[np.isfinite(wic)]
        axis_row[0].imshow(wic, origin="lower", vmin=0, vmax=max(1, np.percentile(finite, 98)))
        axis_row[1].imshow(np.where(domains[index], targets[index].astype(float), np.nan), origin="lower", vmin=0, vmax=1, cmap="RdYlGn")
        axis_row[2].imshow(np.where(domains[index], test_mask[position].astype(float), np.nan), origin="lower", vmin=0, vmax=1, cmap="RdYlGn")
        axis_row[0].set_ylabel(f"{labels[index]}\norbit {orbits[index]}, frame {rows[index]['frame']}")
        for axis in axis_row:
            axis.set(xticks=[], yticks=[])
    for axis, title in zip(axes[0], ("Corrected WIC", "Manual target", "Raw U-Net mask")):
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(FIGURES / "held_out_examples.png", dpi=140)
    plt.close(figure)
    print(f"Model: {DATA / 'U_net_weight_0p1.pt'}")
    print(f"Results: {FIGURES}")


if __name__ == "__main__":
    main()
