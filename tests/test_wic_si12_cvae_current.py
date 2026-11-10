"""Paired corrected-WIC/SI12 cache, masking, loss and trainer checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from icanalyzer.cvae.artifacts import load_run_artifact
from icanalyzer.cvae.cache import AutomaticFieldDataset, WIC_SI12_CACHE_FORMAT
from icanalyzer.cvae.data import ChannelTransform
from icanalyzer.cvae.ensemble import sample_decoded_ensemble
from icanalyzer.cvae.loss import (
    CHANNEL_BALANCED_LOSS_CONFIGURATION,
    vae_loss,
)
from icanalyzer.cvae.model import SplineConditionalVAE
from icanalyzer.cvae.training import (
    add_training_holes,
    add_training_holes_vectorized,
)
from scripts.CVAE.prepare_wic_si12_cvae_cache import (
    load_masked_wic_si12,
    parse_arguments as parse_preparation_arguments,
    prepare,
)


class FakePairedProduct:
    def __init__(self, arrays):
        self.product_type = "precipitation_cs"
        self.grid_id = "image_apex_130km_46x46_v1"
        self.schema_version = 1
        self.software_version = "test"
        self.shape = (2, 46, 46)
        self.time = np.arange(2).astype("datetime64[us]")
        self.Kp_interval_start = self.time.copy()
        self.source_indices = {"wic": np.asarray([4, 5])}
        self.mlat = np.zeros((46, 46))
        self.mlt = np.zeros((46, 46))
        self.attrs = {
            "method_quality_weight_method": (
                "WIC auroral-quality field defined as 1 minus dgweight"
            ),
            "count_source": "background_subtracted",
            "source_preprocessing_label": "test",
            "proton_energy_model": "hardy",
            "proton_flux_source": "SI12",
        }
        self._arrays = arrays
        for name in arrays:
            setattr(self, name, SimpleNamespace())

    def read(self, name):
        return self._arrays[name]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_wic_mask_keeps_separate_sensor_validity_and_quality(tmp_path):
    source = tmp_path / "or_0085.nc"
    source.write_bytes(b"source")
    masks = tmp_path / "masks"
    masks.mkdir()
    selection = np.ones((2, 46, 46), dtype=bool)
    selection[0, 0, 0] = False
    np.savez(
        masks / "or_0085.npz",
        time=np.arange(2).astype("datetime64[us]"), mask=selection
    )
    record = {
        "source": str(source), "source_bytes": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns, "frames": 2,
    }
    shape = (2, 46, 46)
    arrays = {
        "wic_corrected": np.full(shape, 10.0, dtype=np.float32),
        "dwic_corrected": np.full(shape, 2.0, dtype=np.float32),
        "si12": np.full(shape, 4.0, dtype=np.float32),
        "dsi12": np.full(shape, 0.5, dtype=np.float32),
        "method_valid": np.ones(shape, dtype=bool),
        "method_quality_weight": np.full(shape, 0.5, dtype=np.float32),
        "wic_coverage": np.ones(shape, dtype=np.float32),
        "si12_coverage": np.ones(shape, dtype=np.float32),
        "Kp": np.asarray([1.0, 2.0], dtype=np.float32),
    }
    arrays["wic_coverage"][0, 1, 1] = 0
    arrays["si12_coverage"][0, 2, 2] = 0
    arrays["wic_corrected"][1, 3, 3] = -1
    arrays["si12"][1, 4, 4] = -2

    loaded = load_masked_wic_si12(
        source, masks, record, lambda _: FakePairedProduct(arrays)
    )
    assert loaded["values"].shape == (2, 2, 46, 46)
    assert loaded["standard_deviation"].shape == (2, 2, 46, 46)
    assert loaded["valid"].shape == (2, 2, 46, 46)
    assert not loaded["valid"][0, 0, 1, 1]
    assert loaded["valid"][0, 1, 1, 1]
    assert loaded["valid"][0, 0, 2, 2]
    assert not loaded["valid"][0, 1, 2, 2]
    assert not loaded["valid"][0, :, 0, 0].any()
    assert loaded["valid"][1, :, 0, 0].all()
    assert loaded["quality"][0, 0, 1, 1] == 0
    assert loaded["quality"][0, 1, 1, 1] == 0.5
    assert loaded["values"][1, 0, 3, 3] == 0
    assert loaded["values"][1, 1, 4, 4] == 0


def make_joint_cache(directory):
    rows = []
    for orbit, split in enumerate(("train", "validation", "test"), start=1):
        shard = directory / f"or_{orbit:04d}"
        shard.mkdir(parents=True)
        physical = np.ones((2, 6, 46, 46), dtype=np.float32)
        physical[:, 0] = 10
        physical[:, 1] = 4
        physical[:, 2] = 2
        physical[:, 3] = 0.5
        physical[:, 4] = 0.7
        physical[:, 5] = 0.4
        valid = np.ones((2, 2, 46, 46), dtype=bool)
        valid[:, 0, 0, 0] = False
        valid[:, 1, 1, 1] = False
        np.save(shard / "physical.npy", physical)
        np.save(shard / "valid.npy", valid)
        time = np.asarray([
            f"200{orbit}-01-01", f"200{orbit}-04-01"
        ], dtype="datetime64[us]")
        np.savez(
            shard / "frames.npz", eligible=[True, True], kp=[1.0, 3.0],
            source_frame=[0, 1], wic_source_index=[10, 11], time=time,
            kp_interval_start=time
        )
        rows.append({
            "orbit": orbit, "split": split, "source": f"or_{orbit:04d}.nc",
            "shard": shard.name, "prepared_frames": 2, "used_frames": 2,
            "product": {"software_version": "test"},
        })
    transform = ChannelTransform(
        [1, 1], [0, 0], [1, 1], kind=ChannelTransform.LOG10_KIND
    )
    mlat = np.broadcast_to(
        np.linspace(50, 90, 46)[:, None], (46, 46)
    ).copy()
    mlt = np.broadcast_to(
        np.linspace(0, 24, 46, endpoint=False)[None], (46, 46)
    ).copy()
    metadata = {
        "cache_format": WIC_SI12_CACHE_FORMAT,
        "input_product_type": "precipitation_cs",
        "field_names": ["wic_corrected", "si12"],
        "field_units": ["counts", "counts"],
        "quality_channel_count": 2, "valid_channel_count": 2,
        "transform": transform.to_dict(), "orbits": rows, "seed": 1,
        "used_frames": {name: 2 for name in ("train", "validation", "test")},
        "maximum_missing_fraction": 0.95,
        "minimum_valid_fraction": 0.05,
        "mask_source": {"kind": "manual", "annotations": {"sha256": "test"}},
        "mask_policy": "test separate validity",
        "sensor_support_overlap": {"both_fraction_of_selected": 0.9},
        "grid_id": "synthetic",
        "grid": {
            "mlat": mlat.tolist(),
            "mlt": mlt.tolist(),
        },
        "split_files": {name: [] for name in ("train", "validation", "test")},
    }
    (directory / "manifest.json").write_text(json.dumps(metadata))
    np.save(
        directory / "training_support.npy",
        np.ones((2, 46, 46), dtype=bool)
    )
    return directory


def test_full_joint_cache_preparation_retains_both_uncertainties(tmp_path):
    source_directory = tmp_path / "source"
    mask_directory = tmp_path / "masks"
    source_directory.mkdir()
    mask_directory.mkdir()
    records = []
    for orbit in (85, 86, 87):
        source = source_directory / f"or_{orbit:04d}.nc"
        source.write_bytes(f"source-{orbit}".encode())
        np.savez(
            mask_directory / f"or_{orbit:04d}.npz",
            time=np.arange(2).astype("datetime64[us]"),
            mask=np.ones((2, 46, 46), dtype=bool),
        )
        records.append({
            "orbit": source.stem, "source": str(source.resolve()),
            "source_bytes": source.stat().st_size,
            "source_mtime_ns": source.stat().st_mtime_ns, "frames": 2,
        })
    (mask_directory / "manifest.json").write_text(json.dumps({
        "orbits": records, "model_sha256": "test", "threshold": 0.5,
    }))
    shape = (2, 46, 46)
    arrays = {
        "wic_corrected": np.stack((
            np.full((46, 46), 10.0), np.full((46, 46), 20.0)
        )).astype(np.float32),
        "dwic_corrected": np.full(shape, 2.0, dtype=np.float32),
        "si12": np.stack((
            np.full((46, 46), 4.0), np.full((46, 46), 8.0)
        )).astype(np.float32),
        "dsi12": np.full(shape, 0.5, dtype=np.float32),
        "method_valid": np.ones(shape, dtype=bool),
        "method_quality_weight": np.full(shape, 0.5, dtype=np.float32),
        "wic_coverage": np.ones(shape, dtype=np.float32),
        "si12_coverage": np.ones(shape, dtype=np.float32),
        "Kp": np.asarray([1.0, 2.0], dtype=np.float32),
    }
    output = tmp_path / "prepared"
    arguments = parse_preparation_arguments([
        "--source", str(source_directory), "--masks", str(mask_directory),
        "--output", str(output), "--minimum-valid-fraction", "0.05",
    ])
    metadata = prepare(
        arguments, product_loader=lambda _: FakePairedProduct(arrays)
    )
    assert metadata["used_frames"] == {
        "train": 2, "validation": 2, "test": 2
    }
    assert metadata["valid_channel_count"] == 2
    assert metadata["quality_channel_count"] == 2
    dataset = AutomaticFieldDataset(
        output, "train", include_uncertainty=True,
        expected_cache_format=WIC_SI12_CACHE_FORMAT
    )
    try:
        np.testing.assert_allclose(
            dataset[0]["physical_standard_deviation"],
            np.stack((np.full((46, 46), 2.0), np.full((46, 46), 0.5)))
        )
    finally:
        dataset.close()


def test_joint_cache_dataset_and_union_encoder(tmp_path):
    cache = make_joint_cache(tmp_path / "cache")
    dataset = AutomaticFieldDataset(
        cache, "train", include_uncertainty=True,
        expected_cache_format=WIC_SI12_CACHE_FORMAT
    )
    try:
        batch = dataset[0]
        assert batch["target"].shape == (2, 46, 46)
        assert batch["mask"].shape == (2, 46, 46)
        assert batch["weight"].shape == (2, 46, 46)
        assert batch["encoder_input"].shape == (4, 46, 46)
        assert batch["physical_standard_deviation"].shape == (2, 46, 46)
        assert batch["target"][0, 0, 0] == 0
        assert batch["target"][1, 0, 0] != 0
    finally:
        dataset.close()

    model = SplineConditionalVAE(
        image_shape=(46, 46), latent_dim=2, spline_control_points=5,
        spline_degree=3, quality_weight_channels=2,
        encoder_mask_policy="union", field_names=("wic_corrected", "si12")
    ).eval()
    assert model.enc_pconv1.in_channels == 4
    restored = SplineConditionalVAE.from_configuration(model.configuration())
    assert restored.quality_weight_channels == 2
    assert restored.encoder_mask_policy == "union"
    output, _, _ = model(
        batch["encoder_input"][None], batch["mask"][None]
    )
    assert output.shape == (1, 2, 46, 46)
    support = np.zeros((2, 46, 46), dtype=bool)
    support[0, :23] = True
    support[1, 23:] = True
    decoded = sample_decoded_ensemble(
        model, dataset.transform, 2, batch_size=1,
        support_mask=support, device="cpu"
    )
    assert np.all(decoded[:, 0, 23:] == 0)
    assert np.all(decoded[:, 1, :23] == 0)


def test_channel_balanced_loss_normalizes_each_sensor_support():
    target = torch.zeros(1, 2, 2, 2)
    reconstructed = torch.ones_like(target)
    reconstructed[0, 0, 1, 1] = 1000
    mask = torch.tensor([[[[1, 1], [1, 0]], [[1, 0], [0, 0]]]], dtype=torch.bool)
    weight = mask.to(dtype=torch.float32)
    mu = torch.zeros(1, 1)
    logvar = torch.zeros_like(mu)
    losses = vae_loss(
        reconstructed, target, mask, weight, mu, logvar, beta=0,
        channel_balanced=True
    )
    torch.testing.assert_close(
        losses["channel_reconstruction"], torch.tensor([4.0, 4.0])
    )
    torch.testing.assert_close(losses["reconstruction"], torch.tensor(4.0))


def test_joint_holes_remove_same_region_and_keep_each_sensor():
    encoder = torch.ones(1, 4, 8, 8)
    mask = torch.ones(1, 2, 8, 8, dtype=torch.bool)
    mask[:, 0, 0, 0] = False
    damaged, remaining = add_training_holes(
        encoder, mask, probability=1.0, hole_size=3,
        generator=torch.Generator().manual_seed(3)
    )
    removed = mask & ~remaining
    torch.testing.assert_close(removed[:, 0], removed[:, 1])
    assert remaining.flatten(2).any(dim=2).all()
    assert torch.all(damaged[:, :, removed[0, 0]] == 0)

    vectorized, vectorized_mask = add_training_holes_vectorized(
        encoder, mask, probability=1.0, hole_size=3,
        generator=torch.Generator().manual_seed(3)
    )
    vectorized_removed = mask & ~vectorized_mask
    vectorized_hole = torch.all(vectorized == 0, dim=1)
    torch.testing.assert_close(
        vectorized_mask, mask & ~vectorized_hole[:, None]
    )
    assert vectorized_mask.flatten(2).any(dim=2).all()
    assert torch.all(vectorized[:, :, vectorized_removed[0, 0]] == 0)


def test_joint_trainer_uses_channel_balanced_configuration(tmp_path, monkeypatch):
    import scripts.CVAE.train_wic_cvae as trainer

    cache = make_joint_cache(tmp_path / "cache")

    def skip_training(model, *args, **kwargs):
        assert kwargs["loss_configuration"] == CHANNEL_BALANCED_LOSS_CONFIGURATION
        model.best_validation_epoch = 1
        model.best_validation_total = 0.5
        return model, [{"epoch": 1}]

    monkeypatch.setattr(trainer, "train_vae", skip_training)
    monkeypatch.setattr(
        trainer, "_run_epoch",
        lambda *args, **kwargs: {
            "total": 1.0, "reconstruction": 0.8, "kl": 0.2,
            "channel_reconstruction": [0.7, 0.9],
        }
    )
    output = tmp_path / "run"
    arguments = trainer.parse_arguments([
        str(cache), "--output", str(output), "--device", "cpu",
        "--num-workers", "0", "--epochs", "1", "--latent-dim", "2",
        "--spline-control-points", "5",
    ])
    summary = trainer.run(arguments)
    model, _, artifact = load_run_artifact(output / "wic_cvae.pt")
    assert model.field_names == ("wic_corrected", "si12")
    assert model.quality_weight_channels == 2
    assert model.encoder_mask_policy == "union"
    assert artifact["training_configuration"]["loss_configuration"] == (
        CHANNEL_BALANCED_LOSS_CONFIGURATION
    )
    assert summary["held_out_evaluation"]["test"][
        "channel_reconstruction"
    ] == [0.7, 0.9]


def test_joint_sampler_and_plotter_preserve_paired_members(tmp_path, monkeypatch):
    import scripts.CVAE.train_wic_cvae as trainer
    import scripts.CVAE.sample_wic_at_kp as sampler
    import scripts.CVAE.plot_wic_at_kp as plotter

    cache = make_joint_cache(tmp_path / "cache")

    def skip_training(model, *args, **kwargs):
        model.best_validation_epoch = 1
        model.best_validation_total = 0.5
        return model, [{"epoch": 1}]

    monkeypatch.setattr(trainer, "train_vae", skip_training)
    monkeypatch.setattr(
        trainer, "_run_epoch",
        lambda *args, **kwargs: {
            "total": 1.0, "reconstruction": 0.8, "kl": 0.2,
            "channel_reconstruction": [0.7, 0.9],
        }
    )
    run_directory = tmp_path / "run"
    trainer.run(trainer.parse_arguments([
        str(cache), "--output", str(run_directory), "--device", "cpu",
        "--num-workers", "0", "--epochs", "1", "--latent-dim", "2",
        "--spline-control-points", "5",
    ]))

    def fixed_samples(
        model, transform, n_samples, return_latent=False,
        support_mask=None, **kwargs
    ):
        assert model.field_names == ("wic_corrected", "si12")
        assert transform.channel_count == 2
        assert np.shape(support_mask) == (2, 46, 46)
        sequence = np.arange(1, n_samples + 1, dtype=np.float32)
        samples = np.empty((n_samples, 2, 46, 46), dtype=np.float32)
        samples[:, 0] = sequence[:, None, None] * 100
        samples[:, 1] = sequence[:, None, None] * 5
        latent = np.column_stack((sequence, -sequence)).astype(np.float32)
        return (samples, latent) if return_latent else samples

    def fixed_precipitation(wic, dwic, kp, mlt):
        wic, kp, mlt = np.broadcast_arrays(wic, kp, mlt)
        return {
            "E0": np.full(wic.shape, 2.0),
            "dE0": np.full(wic.shape, 0.25),
            "Fe": wic / 100,
            "dFe": np.zeros(wic.shape),
            "varE0Fe": np.zeros(wic.shape),
            "E0_response_range_flag": np.zeros(wic.shape, dtype=np.int8),
        }

    def fixed_fang(energy, flux, field, low_energy_policy="clip"):
        energy, flux, field = np.broadcast_arrays(energy, flux, field)
        valid = np.isfinite(energy) & np.isfinite(flux)
        return {
            "H": np.sqrt(flux), "P": np.sqrt(flux), "valid": valid,
            "within_model_domain": valid,
            "energy_clipped_low": np.zeros(valid.shape, dtype=bool),
            "evaluated_energy": energy,
        }

    monkeypatch.setattr(sampler, "sample_decoded_ensemble", fixed_samples)
    monkeypatch.setattr(
        sampler, "precipitation_from_collapsed_zhang_paxton",
        fixed_precipitation
    )
    monkeypatch.setattr(sampler, "fang_2025_conductance", fixed_fang)
    monkeypatch.setattr(
        sampler, "fang_magnetic_field_from_apex_grid",
        lambda mlat, mlt, support, epoch: {
            "modified_apex_longitude": np.zeros_like(mlat),
            "geographic_latitude": np.zeros_like(mlat),
            "geographic_longitude": np.zeros_like(mlat),
            "apex2geo_error_degrees": np.zeros_like(mlat),
            "magnetic_field_microtesla": np.full_like(mlat, 48.0),
            "epoch": "2001-01-01T00:00:00",
            "reference_height_km": 130.0, "field_height_km": 150.0,
            "magnetic_field_model": "test",
            "magnetic_field_package_version": "test",
            "magnetic_field_coefficients_sha256": "0" * 64,
            "coordinate_mapping": "test",
        }
    )
    output = tmp_path / "samples"
    sampled = sampler.run(sampler.parse_arguments([
        str(run_directory), "--kp", "2", "--output", str(output),
        "--draws", "4", "--batch-size", "2", "--device", "cpu",
        "--no-plots",
    ]))
    assert sampled[0]["paired_sensor_samples"]
    assert not sampled[0]["proton_conductance_conversion_applied"]
    assert sampled[0]["field_names"] == [
        "wic_corrected", "si12", "Hall", "Pedersen"
    ]
    product = output / "kp_00_2/ensemble_products.npz"
    with np.load(product) as saved:
        assert saved["wic_anomaly_factor"].shape == (46 * 46, 4)
        assert saved["si12_anomaly_factor"].shape == (46 * 46, 4)
        np.testing.assert_array_equal(
            saved["wic_mosaic_sample_indices"],
            saved["si12_mosaic_sample_indices"]
        )
        np.testing.assert_allclose(
            saved["si12_mosaic_samples"],
            saved["wic_mosaic_samples"] / 20
        )
        assert saved["si12_medoid"].shape == (46, 46)
        assert float(saved["si12_exceedance_threshold"]) == 10.0
        assert saved["conductance_mean"].shape == (2, 46, 46)

    assert plotter.run(plotter.parse_arguments([str(output)])) == ["kp_00_2"]
    for name in (
        "wic_at_kp.png", "wic_sample_mosaic.png", "wic_medoid.png",
        "wic_exceedance_probability.png", "si12_at_kp.png",
        "si12_sample_mosaic.png", "si12_medoid.png",
        "si12_exceedance_probability.png", "conductance_at_kp.png",
    ):
        assert (output / "kp_00_2" / name).is_file()

    plotted = []
    monkeypatch.setattr(
        plotter, "plot_wic", lambda *args: plotted.append("sensor")
    )
    monkeypatch.setattr(
        plotter, "plot_wic_sample_mosaic",
        lambda *args: plotted.append("mosaic")
    )
    monkeypatch.setattr(
        plotter, "plot_wic_medoid", lambda *args: plotted.append("medoid")
    )
    monkeypatch.setattr(
        plotter, "plot_wic_exceedance_probability",
        lambda *args: plotted.append("exceedance")
    )
    monkeypatch.setattr(
        plotter, "plot_conductance",
        lambda *args: plotted.append("conductance")
    )
    assert plotter.run(plotter.parse_arguments([str(output)])) == ["kp_00_2"]
    assert plotted == [
        "sensor", "mosaic", "medoid", "exceedance",
        "sensor", "mosaic", "medoid", "exceedance", "conductance",
    ]
