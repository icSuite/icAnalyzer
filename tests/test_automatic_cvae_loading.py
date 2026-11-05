"""Small cache/loading checks; no scientific model training."""

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from icanalyzer.cvae.data import ChannelTransform
from icanalyzer.cvae.training import add_training_holes
import icanalyzer.cvae.training as workflow
from scripts.CVAE.prepare_cvae_hp_cache import AutomaticConductanceDataset, CACHE_FORMAT
from scripts.CVAE.train_automatic_cvae import parse_arguments, run, training_device


def test_unavailable_cuda_fails_before_cache_loading_or_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "run"
    arguments = parse_arguments([str(tmp_path / "nonexistent_cache"), "--output", str(output), "--device", "cuda"])
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        run(arguments)
    assert not output.exists()


def test_cuda_allocation_failure_reports_driver_before_training(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    def broken_allocation(*args, **kwargs):
        raise RuntimeError("driver could not initialize")
    monkeypatch.setattr(torch, "empty", broken_allocation)
    with pytest.raises(RuntimeError, match="CUDA could not initialize"):
        training_device("cuda")


def test_device_selection_preserves_explicit_cpu_and_existing_auto_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert training_device("cpu") == "cpu"
    assert training_device("auto") == "cpu"


def make_cache(directory):
    transform = ChannelTransform([1, 1], [0, 0], [1, 1], kind=ChannelTransform.LOG10_KIND)
    rows = []
    for orbit in range(2):
        shard = directory / f"orbit_{orbit}"
        shard.mkdir()
        physical = np.ones((4, 5, 46, 46), dtype=np.float32)
        physical[:, :2] *= np.arange(1, 5, dtype=np.float32)[:, None, None, None] + orbit
        physical[:, 4] = 0.75
        valid = np.ones((4, 46, 46), dtype=bool)
        valid[:, :2] = False
        np.save(shard / "physical.npy", physical)
        np.save(shard / "valid.npy", valid)
        np.savez(shard / "frames.npz", eligible=np.ones(4, dtype=bool), kp=np.arange(4), source_frame=np.arange(4), time=np.arange(4).astype("datetime64[us]"))
        rows.append({"split": "train", "shard": shard.name, "orbit": orbit, "prepared_frames": 4})
    metadata = {"cache_format": CACHE_FORMAT, "transform": transform.to_dict(), "orbits": rows}
    (directory / "manifest.json").write_text(json.dumps(metadata))
    return directory


def test_optional_uncertainty_does_not_change_training_inputs(tmp_path):
    cache = make_cache(tmp_path)
    complete = AutomaticConductanceDataset(cache, "train")
    lean = AutomaticConductanceDataset(cache, "train", include_uncertainty=False)
    try:
        original = complete[2]
        lean.transform.variance = lambda *args: (_ for _ in ()).throw(AssertionError("unused variance was calculated"))
        minimal = lean[2]
        assert set(original) - set(minimal) == {"variance", "physical_target", "physical_standard_deviation"}
        for name in minimal:
            if torch.is_tensor(minimal[name]):
                assert torch.equal(minimal[name], original[name])
            else:
                assert minimal[name] == original[name]
    finally:
        complete.close()
        lean.close()


def test_evicted_mapping_reuses_verified_header_and_keeps_budget(tmp_path, monkeypatch):
    dataset = AutomaticConductanceDataset(make_cache(tmp_path), "train", cache_mib=1, max_open_orbits=1)
    original_load = np.load
    calls = []

    def counted_load(*args, **kwargs):
        calls.append(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)
    try:
        first = dataset[0]
        dataset[4]
        repeated = dataset[0]
        assert len(calls) == 4  # Two arrays per orbit, only on first access.
        assert torch.equal(first["encoder_input"], repeated["encoder_input"])
        assert len(dataset.open_shards) == 1
        assert dataset.mapped_bytes <= dataset.cache_limit
    finally:
        dataset.close()


def test_workers_preserve_shuffled_batches_holes_and_main_rng(tmp_path):
    cache = make_cache(tmp_path)

    def batches(workers):
        dataset = AutomaticConductanceDataset(cache, "train", cache_mib=1, max_open_orbits=1, include_uncertainty=False)
        options = {"multiprocessing_context": "spawn", "prefetch_factor": 2, "timeout": 30} if workers else {}
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=workers, **options)
        torch.manual_seed(19)
        holes = torch.Generator().manual_seed(22)
        result = []
        try:
            for batch in loader:
                damaged, mask = add_training_holes(batch["encoder_input"], batch["mask"], 0.5, 12, holes)
                result.append((batch["index"], damaged, mask))
            return result, torch.random.get_rng_state().clone()
        finally:
            dataset.close()

    serial, serial_rng = batches(0)
    parallel, parallel_rng = batches(2)
    assert torch.equal(serial_rng, parallel_rng)
    for expected, actual in zip(serial, parallel):
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))


def test_epoch_metrics_match_original_float64_accumulation(monkeypatch):
    class ZeroModel(torch.nn.Module):
        def forward(self, values, mask):
            latent = torch.zeros(len(values), 1)
            return values[:, :2], latent, latent

    values = (1e8, 1.25, 2.5)
    reported = []

    def fake_loss(*args, **kwargs):
        value = values[len(reported)]
        losses = {name: torch.tensor(value / (index + 1), dtype=torch.float32) for index, name in enumerate(("total", "reconstruction", "kl"))}
        reported.append({name: float(loss) for name, loss in losses.items()})
        return losses

    monkeypatch.setattr(workflow, "vae_loss", fake_loss)
    batch = {"encoder_input": torch.ones(2, 3, 4, 4), "target": torch.ones(2, 2, 4, 4), "mask": torch.ones(2, 1, 4, 4, dtype=torch.bool), "weight": torch.ones(2, 1, 4, 4)}
    metrics = workflow._run_epoch(ZeroModel(), [batch] * 3, None, "cpu", 1.6, False, loss_configuration=workflow.LEGACY_LOSS_CONFIGURATION)
    assert metrics == {name: sum(row[name] for row in reported) / 6 for name in metrics}
