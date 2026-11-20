"""Train the adopted single-frame Model2 WIC-mask U-Net."""

#%% Imports

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gudhi  # noqa: F401 -- fail early when the adopted topology dependency is absent
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.automatic_masking.model2_unet import (
    FrameWindowDataset, SmallUNet, class_scores, masked_loss, pixel_scores,
    predict, prepare_inputs, sample_weights, validation_correction, window_inputs
)
from scripts.automatic_masking.topology import shape_scores, topology_loss


FIELDS = ["wic_corrected", "method_quality_weight", "sza", "dza"]
TOPOLOGY_WEIGHT = 0.1


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_torch_save(payload, filename):
    filename = Path(filename)
    temporary = filename.with_name(filename.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(filename)


def atomic_json_save(payload, filename):
    filename = Path(filename)
    temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(filename)


def run(arguments):
    prepared = arguments.prepared.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    progress_file = output / "training_state.pt"
    if output.exists() and not arguments.resume:
        raise FileExistsError("choose a fresh Model2 output directory")
    if arguments.resume and not progress_file.is_file():
        raise FileNotFoundError(
            f"cannot resume without the saved training state: {progress_file}"
        )

    provenance = json.loads((prepared / "dataset_provenance.json").read_text())
    image_shape = tuple(provenance.get("image_shape", (46, 46)))
    if len(image_shape) != 2 or any(int(value) < 1 for value in image_shape):
        raise ValueError("prepared Model2 image_shape is invalid")
    if file_sha256(prepared / "annotations_snapshot.json") != provenance["annotation_sha256"]:
        raise ValueError("prepared annotation snapshot differs from its provenance")
    for source in provenance["sources"]:
        path = Path(source["file"])
        stat = path.stat()
        if stat.st_size != source["bytes"] or stat.st_mtime_ns != source["mtime_ns"]:
            raise ValueError(f"source product changed since preparation: {path}")

    rows = list(csv.DictReader((prepared / "frames.csv").open()))
    split = {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in json.loads((prepared / "split_ids.json").read_text()).items()
    }
    with np.load(prepared / "dataset.npz") as archive:
        images = archive["images"]
        targets = archive["targets"]
        domains = archive["domains"]
    if images.shape[-2:] != image_shape or targets.shape[-2:] != image_shape or domains.shape[-2:] != image_shape:
        raise ValueError("prepared Model2 arrays do not match image_shape")
    images = images[:, [provenance["fields"].index(name) for name in FIELDS]]
    orbits = np.asarray([int(row["orbit"]) for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    classes = np.asarray([("full", "bad", "partial").index(label) for label in labels])

    torch.set_num_threads(arguments.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    device = torch.device("cuda" if arguments.device == "auto" and torch.cuda.is_available() else "cpu" if arguments.device == "auto" else arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    topology_shape = tuple(
        min(arguments.topology_resolution, size) for size in image_shape
    )
    training_configuration = {
        "prepared": str(prepared),
        "prepared_provenance_sha256": file_sha256(
            prepared / "dataset_provenance.json"
        ),
        "annotation_sha256": provenance["annotation_sha256"],
        "image_shape": list(image_shape),
        "grid_id": provenance.get("grid_id"),
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "patience": arguments.patience,
        "minimum_improvement": arguments.minimum_improvement,
        "topology_weight": TOPOLOGY_WEIGHT,
        "topology_resolution": arguments.topology_resolution,
        "topology_shape": list(topology_shape)
    }

    inputs, normalization = prepare_inputs(images, split["train"], FIELDS)
    windows = np.arange(len(rows))[:, None]
    train_weights = sample_weights(rows, split["train"])
    validation_weights = sample_weights(rows, split["validation"])
    loader = DataLoader(
        FrameWindowDataset(inputs, windows, split["train"], targets, domains, classes, train_weights),
        batch_size=arguments.batch_size, shuffle=True
    )
    model = SmallUNet(inputs.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay)
    output.mkdir(parents=True, exist_ok=arguments.resume)

    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    start_epoch = 0
    resumes = 0
    if arguments.resume:
        saved = torch.load(progress_file, map_location=device, weights_only=False)
        if saved.get("training_configuration") != training_configuration:
            raise ValueError(
                "resume arguments or prepared data differ from training_state.pt"
            )
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        history = saved["history"]
        best_loss = float(saved["best_loss"])
        best_epoch = int(saved["best_epoch"])
        best_state = saved["best_state_dict"]
        start_epoch = int(saved["completed_epoch"])
        resumes = int(saved.get("resumes", 0)) + 1
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        np.random.set_state(saved["numpy_rng_state"])
        if device.type == "cuda" and saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([
                state.cpu() for state in saved["cuda_rng_state"]
            ])
        print(
            f"Resuming after epoch {start_epoch}; best epoch {best_epoch}",
            flush=True
        )

    print(
        f"Training {len(split['train'])} frames; validation "
        f"{len(split['validation'])}; test {len(split['test'])}; device {device}; "
        f"topology grid {topology_shape}",
        flush=True
    )

    #%% Fit on training orbits and select the checkpoint on validation loss

    epoch_numbers = range(start_epoch, arguments.epochs)
    if start_epoch - best_epoch >= arguments.patience:
        epoch_numbers = ()
    for epoch in epoch_numbers:
        model.train()
        training_total = 0.0
        for values, target, domain, _, weight in loader:
            optimizer.zero_grad()
            logits = model(values.to(device))
            losses = masked_loss(logits, target.to(device), domain.to(device))
            losses = losses + TOPOLOGY_WEIGHT * topology_loss(
                logits, resolution=arguments.topology_resolution
            )
            loss = (losses * weight.to(device)).mean()
            loss.backward()
            optimizer.step()
            training_total += float(loss.detach()) * len(values)

        model.eval()
        validation_losses = []
        with torch.no_grad():
            batches = np.array_split(
                split["validation"],
                max(1, int(np.ceil(len(split["validation"]) / arguments.batch_size)))
            )
            for batch in batches:
                values = window_inputs(inputs, windows, batch)
                logits = model(torch.from_numpy(values).to(device))
                losses = masked_loss(
                    logits,
                    torch.from_numpy(targets[batch].astype(np.float32)).to(device),
                    torch.from_numpy(domains[batch].astype(np.float32)).to(device)
                )
                validation_losses.extend((
                    losses + TOPOLOGY_WEIGHT * topology_loss(
                        logits, resolution=arguments.topology_resolution
                    )
                ).cpu().numpy())
        validation_loss = float(np.average(validation_losses, weights=validation_weights))
        history.append({"epoch": epoch + 1, "train": training_total / len(split["train"]), "validation": validation_loss})
        if validation_loss < best_loss - arguments.minimum_improvement:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        print(f"epoch {epoch + 1}: train {history[-1]['train']:.4f}, validation {validation_loss:.4f}, best {best_epoch}", flush=True)

        progress = {
            "format": "model2_training_state_v1",
            "training_configuration": training_configuration,
            "completed_epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "best_state_dict": best_state,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
            "resumes": resumes
        }
        atomic_json_save(history, output / "history.json")
        atomic_torch_save(progress, progress_file)
        atomic_torch_save({
            "format": "model2_best_state_v1",
            "training_configuration": training_configuration,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "state_dict": best_state
        }, output / "best_state.pt")
        if epoch + 1 - best_epoch >= arguments.patience:
            break

    #%% Freeze the validation threshold and inspect the held-out test orbits once

    model.load_state_dict(best_state)
    validation_probability, _ = predict(
        model, inputs, split["validation"], device,
        batch_size=arguments.batch_size
    )
    thresholds = np.linspace(0.1, 0.9, 33)
    validation_errors = [
        validation_correction(
            pixel_scores(validation_probability >= threshold, targets[split["validation"]], domains[split["validation"]]),
            orbits[split["validation"]], labels[split["validation"]]
        )
        for threshold in thresholds
    ]
    threshold = float(thresholds[np.argmin(validation_errors)])
    test_probability, _ = predict(
        model, inputs, split["test"], device,
        batch_size=arguments.batch_size
    )
    test_mask = test_probability >= threshold
    by_class = class_scores(
        pixel_scores(test_mask, targets[split["test"]], domains[split["test"]]),
        orbits[split["test"]], labels[split["test"]]
    )
    shape = shape_scores(test_mask, domains[split["test"]])

    checkpoint = output / "model2_unet.pt"
    configuration = {
        **training_configuration,
        "annotation_sha256": provenance["annotation_sha256"], "fields": FIELDS,
        "image_shape": list(image_shape), "grid_id": provenance.get("grid_id"),
        "split": provenance["split"], "split_policy": provenance["split_policy"],
        "best_epoch": best_epoch, "completed_epochs": len(history),
        "training_resumes": resumes, "threshold": threshold
    }
    atomic_torch_save({
        "state_dict": best_state, "normalization": normalization,
        "input_fields": FIELDS, "threshold": threshold,
        "topology_weight": TOPOLOGY_WEIGHT,
        "manual_annotation_sha256": provenance["annotation_sha256"],
        "image_shape": list(image_shape), "grid_id": provenance.get("grid_id"),
        "configuration": configuration
    }, checkpoint)
    metrics = {
        **configuration,
        "validation_disagreement": float(min(validation_errors)),
        "test_by_class": by_class,
        "test_observable_shape_violations": int(shape["any_violation"].sum()),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint)
    }
    atomic_json_save(history, output / "history.json")
    atomic_json_save(metrics, output / "summary.json")

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["epoch"] for row in history], [row["train"] for row in history], label="Train")
    axis.plot([row["epoch"] for row in history], [row["validation"] for row in history], label="Validation")
    axis.set(xlabel="Epoch", ylabel="Weighted loss", title="Model2 training")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "training_history.png", dpi=160)
    plt.close(figure)
    print(f"Saved Model2 checkpoint: {checkpoint}", flush=True)
    return metrics


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minimum-improvement", type=float, default=0.0001)
    parser.add_argument(
        "--topology-resolution", type=int, default=46,
        help="maximum topology-loss grid dimension; BCE and Dice remain native"
    )
    parser.add_argument("--seed", type=int, default=12026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--resume", action="store_true",
        help="resume this output directory from training_state.pt"
    )
    arguments = parser.parse_args(argv)
    if min(
        arguments.epochs, arguments.batch_size, arguments.patience,
        arguments.topology_resolution, arguments.threads
    ) < 1:
        parser.error(
            "epochs, batch size, patience, topology resolution and threads "
            "must be positive"
        )
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
