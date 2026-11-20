"""Recovery of interrupted Model2 mask training."""

import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.automatic_masking import train_model2_unet as trainer


def prepared_dataset(directory):
    directory.mkdir()
    annotation = b'{"schema_version": 3}\n'
    (directory / "annotations_snapshot.json").write_bytes(annotation)
    source = directory / "source.nc"
    source.write_bytes(b"source identity")

    fields = [
        "wic_corrected", "method_quality_weight", "wic_coverage", "sza", "dza"
    ]
    rng = np.random.default_rng(4)
    images = rng.normal(size=(6, len(fields), 8, 8)).astype(np.float32)
    targets = np.ones((6, 8, 8), dtype=bool)
    targets[1::2] = False
    domains = np.ones_like(targets)
    np.savez_compressed(
        directory / "dataset.npz", images=images, targets=targets, domains=domains
    )

    rows = [
        {"orbit": orbit, "label": label, "segment": f"{orbit}:{label}:0"}
        for orbit in (1, 2, 3)
        for label in ("full", "bad")
    ]
    with (directory / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    split = {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}
    (directory / "split_ids.json").write_text(json.dumps(split))

    stat = source.stat()
    provenance = {
        "annotation_sha256": hashlib.sha256(annotation).hexdigest(),
        "fields": fields,
        "image_shape": [8, 8],
        "grid_id": "test_grid",
        "split": {
            name: {"frames": len(indices), "orbits": [position + 1]}
            for position, (name, indices) in enumerate(split.items())
        },
        "split_policy": {"kind": "test"},
        "sources": [{
            "file": str(source), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns
        }]
    }
    (directory / "dataset_provenance.json").write_text(json.dumps(provenance))


def arguments(prepared, output, resume=False):
    return SimpleNamespace(
        prepared=prepared, output=output, device="cpu", epochs=2, batch_size=2,
        learning_rate=0.001, weight_decay=0.0001, patience=8,
        minimum_improvement=0.0001, topology_resolution=46, seed=12026,
        threads=1, resume=resume
    )


def test_interrupted_training_resumes_from_last_completed_epoch(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    output = tmp_path / "run"
    prepared_dataset(prepared)
    monkeypatch.setattr(trainer.torch, "set_num_interop_threads", lambda _: None)

    calls = 0

    def interrupt_during_second_epoch(logits, resolution):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated disconnect")
        return logits.sum(dim=(1, 2)) * 0

    monkeypatch.setattr(trainer, "topology_loss", interrupt_during_second_epoch)
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        trainer.run(arguments(prepared, output))

    state = trainer.torch.load(
        output / "training_state.pt", map_location="cpu", weights_only=False
    )
    assert state["completed_epoch"] == 1
    assert len(json.loads((output / "history.json").read_text())) == 1
    assert not (output / "model2_unet.pt").exists()

    monkeypatch.setattr(
        trainer, "topology_loss",
        lambda logits, resolution: logits.sum(dim=(1, 2)) * 0
    )
    metrics = trainer.run(arguments(prepared, output, resume=True))

    assert metrics["completed_epochs"] == 2
    assert metrics["training_resumes"] == 1
    assert metrics["topology_resolution"] == 46
    assert (output / "model2_unet.pt").is_file()
    assert len(json.loads((output / "history.json").read_text())) == 2
