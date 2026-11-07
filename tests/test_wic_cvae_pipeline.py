import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from icanalyzer.cvae.artifacts import (
    CONDUCTANCE_CVAE_ARTIFACT_TYPE,
    WIC_CVAE_ARTIFACT_TYPE,
    load_run_artifact,
    save_run_artifact
)
from icanalyzer.cvae.data import ChannelTransform, split_orbit_files
from icanalyzer.cvae.ensemble import sample_decoded_ensemble
from icanalyzer.cvae.model import SplineConditionalVAE
from icanalyzer.cvae.cache import AutomaticFieldDataset
from icanalyzer.cvae.conditioning import (
    CONDITIONING_FEATURE_NAMES,
    CONDITIONING_FORMAT,
    SUPERMAG_SOURCE_COLUMNS,
    align_conditioning,
    align_conditioning_features,
    conditioning_vector_from_record,
    conditioning_coverage,
    fit_conditioning_normalization,
    read_supermag_file,
)
from scripts.CVAE.prepare_wic_cvae_cache import WIC_CACHE_FORMAT, load_annotated_wic, load_masked_wic
from scripts.CVAE.select_conditioning_cases import (
    condition_record,
    representative_indices,
)


def test_supermag_reader_alignment_and_core_coverage(tmp_path):
    filename = tmp_path / "year_2000.txt"
    headers = ["year", "month", "day", "hour", "min", "sec"] + list(SUPERMAG_SOURCE_COLUMNS)
    first_values = list(range(1, len(SUPERMAG_SOURCE_COLUMNS) + 1))
    second_values = list(range(101, 101 + len(SUPERMAG_SOURCE_COLUMNS)))
    newell = list(SUPERMAG_SOURCE_COLUMNS).index("Newell CF (Wb/s)")
    first_values[newell] = 999999
    rows = [
        [2000, 1, 1, 0, 0, 0, *first_values],
        [2000, 1, 1, 0, 1, 0, *second_values],
    ]
    filename.write_text(
        "SuperMAG test export downloaded from service on test-time.\n"
        "Data revision 6\n"
        + "\t".join(f"<{name}>" for name in headers) + "\n"
        + "\n".join("\t".join(map(str, row)) for row in rows)
        + "\n<br />\n<b>trailing download-service error</b>\n"
    )

    supermag, provenance = read_supermag_file(filename)
    assert provenance["data_revision"] == 6
    assert provenance["rows"] == 2
    assert provenance["ignored_trailing_nondata_rows"] == 2
    assert np.isnan(supermag["newell"][0])
    assert supermag["newell"][1] == second_values[newell]

    query = np.asarray([
        "2000-01-01T00:00:20", "2000-01-01T00:00:50",
        "2000-01-01T00:03:00"
    ], dtype="datetime64[s]")
    aligned = align_conditioning(supermag, query, maximum_offset_seconds=30)
    np.testing.assert_array_equal(aligned["time_matched"], [True, True, False])
    np.testing.assert_allclose(aligned["time_offset_seconds"][:2], [20, -10])
    assert aligned["sml"][0] == first_values[1]
    assert aligned["sml"][1] == second_values[1]
    assert np.isnan(aligned["sml"][2])
    coverage = conditioning_coverage(aligned)
    assert coverage["time_matched_frames"] == 2
    assert coverage["core_joint_finite_frames"] == 1


def test_conditioning_benchmark_morphology_targets_are_explicit():
    from scripts.CVAE.analyze_conditioning_importance import (
        FEATURE_NAMES,
        morphology_targets,
        time_features,
    )

    wic = np.asarray([[[1., 3.], [1., 3.]], [[0., 0.], [0., 0.]]])
    valid = np.ones_like(wic, dtype=bool)
    mlat = np.asarray([[60., 60.], [70., 70.]])
    mlt = np.asarray([[6., 18.], [6., 18.]])
    targets, valid_fraction = morphology_targets(wic, valid, mlat, mlt)

    np.testing.assert_allclose(
        targets[0], [np.log10(9), np.log10(3), 60, 65, 5, .5]
    )
    assert np.isnan(targets[1, 2:]).all()
    np.testing.assert_array_equal(valid_fraction, [1, 1])

    derived = time_features(np.asarray([
        "2000-01-01T00:00:00", "2000-07-02T12:00:00"
    ], dtype="datetime64[s]"))
    np.testing.assert_allclose(
        derived["day_of_year_sin"], [0, -0.00858348], atol=1e-8
    )
    np.testing.assert_allclose(
        derived["day_of_year_cos"], [1, -0.99996316], atol=1e-8
    )
    np.testing.assert_allclose(derived["dipole_tilt"], [-25.82656, 25.70220], atol=1e-4)
    assert set(FEATURE_NAMES) == {
        "sml", "smr", "by_gsm", "bz_gsm", "vx_gse", "dynamic_pressure",
        "newell", "kp", "day_of_year_sin", "day_of_year_cos", "dipole_tilt",
        "clock_angle_sin", "clock_angle_cos",
    }


def test_named_conditioning_exact_join_dataset_and_model(tmp_path):
    conditioning_directory = tmp_path / "conditioning"
    conditioning_directory.mkdir()
    orbit = np.asarray([1, 1, 2], dtype=np.int32)
    source_frame = np.asarray([0, 1, 0], dtype=np.int32)
    time = np.asarray([
        "2000-01-01T00:00:00", "2000-04-01T06:00:00",
        "2001-07-01T12:00:00"
    ], dtype="datetime64[us]")
    arrays = {
        "orbit": orbit, "source_frame": source_frame, "time": time,
        "sml": [-100., -200., -300.], "smr": [-10., -20., -30.],
        "by_gsm": [1., 2., 3.], "bz_gsm": [-1., -2., -3.],
        "vx_gse": [-400., -450., -500.],
        "clock_angle_gsm": [45., 90., 135.],
        "dynamic_pressure": [1., 2., 3.],
        "newell": [1000., 2000., 3000.],
    }
    np.savez(conditioning_directory / "conditioning.npz", **arrays)
    (conditioning_directory / "manifest.json").write_text(json.dumps({
        "format": CONDITIONING_FORMAT, "array_file": "conditioning.npz"
    }))
    matrix, _, _ = align_conditioning_features(
        conditioning_directory, orbit, source_frame, time, [1., 3., 5.]
    )
    assert matrix.shape == (3, len(CONDITIONING_FEATURE_NAMES))
    assert np.isfinite(matrix).all()

    cache = make_wic_cache(tmp_path / "cache")
    dataset = AutomaticFieldDataset(
        cache, "train", include_uncertainty=False,
        expected_cache_format=WIC_CACHE_FORMAT
    )
    try:
        attached = np.vstack((matrix[0], matrix[1])).astype(np.float32)
        attached[1, 0] = np.nan
        assert dataset.attach_conditioning(attached) == 1
        assert len(dataset) == 1
        torch.testing.assert_close(
            dataset[0]["conditioning"], torch.from_numpy(matrix[0])
        )
    finally:
        dataset.close()

    normalization = fit_conditioning_normalization(matrix, embedding_dim=6)
    model = SplineConditionalVAE(
        image_shape=(8, 8), latent_dim=4, spline_control_points=4,
        spline_degree=2, conditioning=normalization,
        field_names=("wic_corrected",)
    ).eval()
    values = torch.ones(2, 2, 8, 8)
    valid = torch.ones(2, 1, 8, 8, dtype=torch.bool)
    conditions = torch.from_numpy(matrix[:2])
    reconstructed, mu, logvar = model.forward_samples(
        values, valid, conditioning=conditions, latent_samples=3
    )
    assert reconstructed.shape == (2, 3, 1, 8, 8)
    assert mu.shape == logvar.shape == (2, 4)
    restored = SplineConditionalVAE.from_configuration(model.configuration())
    assert restored.conditioning["feature_names"] == list(
        CONDITIONING_FEATURE_NAMES
    )
    transform = ChannelTransform(
        [1.], [0.], [1.], kind=ChannelTransform.LOG10_KIND
    )
    samples = sample_decoded_ensemble(
        model, transform, 4, batch_size=2, conditioning=matrix[0]
    )
    assert samples.shape == (4, 1, 8, 8)
    from icanalyzer.cvae.training import _run_epoch
    from icanalyzer.cvae.loss import LEGACY_LOSS_CONFIGURATION
    metrics = _run_epoch(
        model,
        [{
            "encoder_input": values, "target": values[:, :1],
            "mask": valid, "weight": torch.ones(2, 1, 8, 8),
            "kp": torch.tensor([1., 3.]), "conditioning": conditions,
        }],
        optimizer=None, device="cpu", beta=.1, training=False,
        loss_configuration=LEGACY_LOSS_CONFIGURATION
    )
    assert set(metrics) == {"total", "reconstruction", "kl"}
    requested = conditioning_vector_from_record({
        "sml": -200, "smr": -20, "imf_by": 2, "imf_bz": -3,
        "vx": -450, "clock_angle": 90, "dynamic_pressure": 2,
        "newell": 4000, "kp": 3, "day_of_year": 100,
        "dipole_tilt": 10,
    })
    assert requested.shape == (len(CONDITIONING_FEATURE_NAMES),)
    assert requested[-2] == 1


def test_condition_case_selection_uses_distinct_observed_rows():
    random = np.random.default_rng(42)
    count = 2000
    day_phase = random.uniform(0, 2 * np.pi, count)
    clock_phase = random.uniform(0, 2 * np.pi, count)
    matrix = np.column_stack((
        -random.lognormal(5.2, .7, count),
        random.normal(-15, 25, count),
        random.normal(0, 5, count),
        random.normal(0, 5, count),
        random.normal(-450, 70, count),
        random.lognormal(.5, .5, count),
        random.lognormal(8, .7, count),
        random.choice(np.arange(0, 7.33, .33), count),
        np.sin(day_phase), np.cos(day_phase),
        random.uniform(-30, 30, count),
        np.sin(clock_phase), np.cos(clock_phase),
    )).astype(np.float32)

    selected = representative_indices(matrix)
    assert [label for label, _, _ in selected] == [
        "typical", "quiet", "active_substorm", "storm",
        "by_negative", "by_positive", "bz_south", "bz_north",
    ]
    assert len({index for _, index, _ in selected}) == len(selected)
    for label, index, candidate_count in selected:
        record = condition_record(
            label, matrix[index], 85, index,
            np.datetime64("2001-04-01T06:00:00"), candidate_count
        )
        round_trip = conditioning_vector_from_record(record)
        np.testing.assert_allclose(round_trip, matrix[index], atol=2e-6)
        assert record["source_orbit"] == 85
        assert record["regime_candidate_frames"] == candidate_count


def test_multi_sample_forward_encodes_once_and_decodes_parallel_draws():
    conditioning = dict(center=2.0, scale=1.0, training_min=0.0, training_max=9.0, embedding_dim=3)
    model = SplineConditionalVAE(
        image_shape=(8, 8), latent_dim=4, spline_control_points=4,
        spline_degree=2, kp_conditioning=conditioning,
        field_names=("wic_corrected",)
    ).eval()
    values = torch.randn(2, 2, 8, 8)
    valid = torch.ones(2, 1, 8, 8, dtype=torch.bool)
    kp = torch.tensor([1.0, 4.0])

    torch.manual_seed(17)
    expected = model(values, valid, kp=kp)
    torch.manual_seed(17)
    unchanged = model.forward_samples(values, valid, kp=kp, latent_samples=1)
    for actual, reference in zip(unchanged, expected):
        torch.testing.assert_close(actual, reference)

    reconstructed, mu, logvar = model.forward_samples(
        values, valid, kp=kp, latent_samples=3
    )
    assert reconstructed.shape == (2, 3, 1, 8, 8)
    assert mu.shape == logvar.shape == (2, 4)


def test_one_field_transform_model_artifact_and_sampling(tmp_path):
    transform = ChannelTransform([1], [0.2], [0.7], kind=ChannelTransform.LOG10_KIND)
    physical = np.arange(24, dtype=np.float32).reshape(2, 1, 3, 4)
    np.testing.assert_allclose(transform.inverse(transform.transform(physical)), physical, rtol=2e-6)

    conditioning = dict(center=2.0, scale=1.0, training_min=0.0, training_max=9.0, embedding_dim=3)
    model = SplineConditionalVAE(
        image_shape=(8, 8), latent_dim=4, spline_control_points=4,
        spline_degree=2, kp_conditioning=conditioning,
        field_names=("wic_corrected",)
    ).eval()
    assert model.enc_pconv1.in_channels == 2
    assert model.spline_conv.out_channels == 1
    restored = SplineConditionalVAE.from_configuration(model.configuration()).eval()
    assert restored.field_names == ("wic_corrected",)

    artifact = tmp_path / "wic_cvae.pt"
    save_run_artifact(
        artifact, model, ChannelTransform([1], [0], [1], kind=ChannelTransform.LOG10_KIND),
        [], {}, {"training_support": np.ones((8, 8), dtype=bool)}, {},
        artifact_type=WIC_CVAE_ARTIFACT_TYPE
    )
    loaded, loaded_transform, saved = load_run_artifact(artifact)
    assert saved["artifact_type"] == WIC_CVAE_ARTIFACT_TYPE
    samples = sample_decoded_ensemble(loaded, loaded_transform, 5, batch_size=2, kp=3)
    assert samples.shape == (5, 1, 8, 8)
    assert np.all(samples >= 0)


def test_generic_cvae_split_recognizes_native_orbit_names():
    split = split_orbit_files(
        [Path("or_0085.nc"), Path("or_0086.nc"), Path("or_0089.nc")],
        validation_fraction=0.15, test_fraction=0.15, seed=1
    )
    assert sorted(path.name for paths in split.values() for path in paths) == [
        "or_0085.nc", "or_0086.nc", "or_0089.nc"
    ]


def test_pre_rename_model_configuration_remains_loadable():
    configuration = SplineConditionalVAE(
        image_shape=(8, 8), latent_dim=4, spline_control_points=4,
        spline_degree=2, field_names=("wic_corrected", "si12")
    ).configuration()
    configuration["model"] = "SplineWICSI12VAE"
    restored = SplineConditionalVAE.from_configuration(configuration)
    assert restored.field_names == ("wic_corrected", "si12")


def test_old_conductance_artifact_field_labels_are_corrected(tmp_path):
    model = SplineConditionalVAE(
        image_shape=(8, 8), latent_dim=4, spline_control_points=4,
        spline_degree=2, field_names=("wic_corrected", "si12")
    )
    artifact = tmp_path / "conductance_vae.pt"
    save_run_artifact(
        artifact, model,
        ChannelTransform([1, 1], [0, 0], [1, 1], kind=ChannelTransform.LOG10_KIND),
        [], {}, {}, {}, artifact_type=CONDUCTANCE_CVAE_ARTIFACT_TYPE
    )
    loaded, _transform, _saved = load_run_artifact(artifact)
    assert loaded.field_names == ("Hall", "Pedersen")

    saved = torch.load(artifact, map_location="cpu", weights_only=False)
    saved["model_configuration"].pop("field_names")
    torch.save(saved, artifact)
    loaded, _transform, _saved = load_run_artifact(artifact)
    assert loaded.field_names == ("Hall", "Pedersen")


def test_wic_fang_geometry_reconstructs_130km_footprint_and_150km_igrf():
    from apexpy import Apex
    import scripts.CVAE.sample_wic_at_kp as sampler

    mlat = np.asarray([[60.0, 65.0], [70.0, 75.0]])
    mlt = np.asarray([[0.0, 6.0], [12.0, 18.0]])
    support = np.asarray([[True, True], [False, True]])
    epoch = sampler.parse_epoch("2001-07-25T15:33:37+02:00")
    result = sampler.fang_magnetic_field_from_apex_grid(mlat, mlt, support, epoch)

    apex = Apex(date=epoch, refh=130.0)
    expected_mlon = apex.mlt2mlon(mlt[support], epoch)
    expected_glat, expected_glon, expected_error = apex.apex2geo(
        mlat[support], expected_mlon, 130.0
    )
    expected_field = apex.get_babs(expected_glat, expected_glon, 150.0) * 1e6
    np.testing.assert_allclose(result["modified_apex_longitude"][support], expected_mlon)
    np.testing.assert_allclose(result["geographic_latitude"][support], expected_glat)
    np.testing.assert_allclose(result["geographic_longitude"][support], expected_glon)
    np.testing.assert_allclose(result["apex2geo_error_degrees"][support], expected_error)
    np.testing.assert_allclose(result["magnetic_field_microtesla"][support], expected_field)
    assert np.isnan(result["magnetic_field_microtesla"][~support]).all()
    assert result["epoch"] == "2001-07-25T13:33:37"
    assert len(result["magnetic_field_coefficients_sha256"]) == 64


def test_wic_polar_contour_uses_mlat_mlt_grid():
    import matplotlib.pyplot as plt
    import scripts.CVAE.sample_wic_at_kp as sampler

    mlat = np.broadcast_to(np.linspace(50, 90, 12)[:, None], (12, 12))
    mlt = np.broadcast_to(np.linspace(0, 24, 12)[None], (12, 12))
    values = np.exp(-((mlat - 68) / 8)**2) * (1 + np.cos(np.pi * mlt / 12))
    figure, axis = plt.subplots()
    try:
        contour = sampler.polar_contour(axis, mlat, mlt, values, 2.0, "test")
        assert len(contour.levels) == 17
        assert axis.get_title() == "test"
    finally:
        plt.close(figure)


def make_wic_cache(directory):
    rows = []
    for orbit, split in enumerate(("train", "validation", "test"), start=1):
        shard = directory / f"or_{orbit:04d}"
        shard.mkdir(parents=True)
        physical = np.ones((2, 3, 46, 46), dtype=np.float32)
        physical[:, 1] = 0.2
        physical[:, 2] = 0.5
        np.save(shard / "physical.npy", physical)
        np.save(shard / "valid.npy", np.ones((2, 46, 46), dtype=bool))
        frame_time = np.asarray([
            f"200{orbit}-01-01T00:00:00", f"200{orbit}-04-01T06:00:00"
        ], dtype="datetime64[us]")
        np.savez(
            shard / "frames.npz", eligible=[True, True], kp=[1.0, 3.0],
            source_frame=[0, 1], wic_source_index=[10, 11],
            time=frame_time, kp_interval_start=frame_time
        )
        rows.append({
            "orbit": orbit, "split": split, "source": f"or_{orbit:04d}.nc",
            "shard": shard.name, "prepared_frames": 2, "used_frames": 2,
            "product": {"software_version": "test"}
        })
    transform = ChannelTransform([1], [0], [1], kind=ChannelTransform.LOG10_KIND)
    metadata = {
        "cache_format": WIC_CACHE_FORMAT, "input_product_type": "precipitation_cs",
        "field_names": ["wic_corrected"], "field_units": ["counts"],
        "transform": transform.to_dict(), "orbits": rows, "seed": 1,
        "used_frames": {name: 2 for name in ("train", "validation", "test")},
        "maximum_missing_fraction": 0.95,
        "mask_source": {"kind": "manual", "annotations": {"sha256": "test"}},
        "grid_id": "synthetic", "grid": {"mlat": np.zeros((46, 46)).tolist(), "mlt": np.zeros((46, 46)).tolist()},
        "split_files": {name: [] for name in ("train", "validation", "test")}
    }
    (directory / "manifest.json").write_text(json.dumps(metadata))
    np.save(directory / "training_support.npy", np.ones((46, 46), dtype=bool))
    return directory


def make_named_conditioning(directory, cache):
    directory.mkdir()
    orbit_values, frame_values, time_values = [], [], []
    for orbit in (1, 2, 3):
        with np.load(cache / f"or_{orbit:04d}/frames.npz") as frames:
            count = len(frames["time"])
            orbit_values.extend([orbit] * count)
            frame_values.extend(frames["source_frame"])
            time_values.extend(frames["time"])
    count = len(time_values)
    sequence = np.arange(1, count + 1, dtype=np.float32)
    np.savez(
        directory / "conditioning.npz",
        orbit=np.asarray(orbit_values, dtype=np.int32),
        source_frame=np.asarray(frame_values, dtype=np.int32),
        time=np.asarray(time_values, dtype="datetime64[us]"),
        sml=-100 * sequence, smr=-10 * sequence,
        by_gsm=sequence, bz_gsm=-sequence, vx_gse=-400 - 10 * sequence,
        clock_angle_gsm=30 * sequence,
        dynamic_pressure=sequence, newell=1000 * sequence,
    )
    (directory / "manifest.json").write_text(json.dumps({
        "format": CONDITIONING_FORMAT, "array_file": "conditioning.npz"
    }))
    return directory


def test_named_conditioning_wic_trainer_artifact(tmp_path, monkeypatch):
    import scripts.CVAE.train_wic_cvae as trainer

    cache = make_wic_cache(tmp_path / "cache")
    conditioning = make_named_conditioning(tmp_path / "conditioning", cache)

    def skip_training(model, *args, **kwargs):
        model.best_validation_epoch = 1
        model.best_validation_total = 0.5
        return model, [{"epoch": 1}]

    monkeypatch.setattr(trainer, "train_vae", skip_training)
    monkeypatch.setattr(
        trainer, "_run_epoch",
        lambda *args, **kwargs: {"total": 1., "reconstruction": .8, "kl": .2}
    )
    output = tmp_path / "run"
    arguments = trainer.parse_arguments([
        str(cache), "--conditioning", str(conditioning),
        "--conditioning-embedding-dim", "6", "--output", str(output),
        "--device", "cpu", "--num-workers", "0", "--epochs", "1",
        "--latent-dim", "2", "--spline-control-points", "5"
    ])
    summary = trainer.run(arguments)
    loaded, _, artifact = load_run_artifact(output / "wic_cvae.pt")
    assert loaded.kp_conditioning is None
    assert loaded.conditioning["embedding_dim"] == 6
    assert loaded.conditioning["feature_names"] == list(
        CONDITIONING_FEATURE_NAMES
    )
    assert summary["used_frames"] == {
        "train": 2, "validation": 2, "test": 2
    }
    assert artifact["data_provenance"]["conditioning"]["format"] == (
        CONDITIONING_FORMAT
    )
    assert (output / "conditioning_manifest.json").is_file()


def test_one_field_disk_cache_and_training_artifact(tmp_path, monkeypatch):
    cache = make_wic_cache(tmp_path / "cache")
    dataset = AutomaticFieldDataset(
        cache, "train", include_uncertainty=False,
        expected_cache_format=WIC_CACHE_FORMAT
    )
    try:
        batch = dataset[0]
        assert batch["target"].shape == (1, 46, 46)
        assert batch["encoder_input"].shape == (2, 46, 46)
        assert batch["weight"].shape == (1, 46, 46)
    finally:
        dataset.close()

    import scripts.CVAE.train_wic_cvae as trainer

    def skip_training(model, *args, **kwargs):
        assert kwargs["latent_samples"] == 4
        model.best_validation_epoch = 1
        model.best_validation_total = 0.5
        return model, [{"epoch": 1}]

    monkeypatch.setattr(trainer, "train_vae", skip_training)
    monkeypatch.setattr(trainer, "_run_epoch", lambda *args, **kwargs: {"total": 1.0, "reconstruction": 0.8, "kl": 0.2})
    output = tmp_path / "run"
    arguments = trainer.parse_arguments([
        str(cache), "--output", str(output), "--device", "cpu",
        "--num-workers", "0", "--latent-dim", "2", "--latent-samples", "4",
        "--spline-control-points", "5"
    ])
    summary = trainer.run(arguments)
    loaded, transform, artifact = load_run_artifact(output / "wic_cvae.pt")
    assert summary["artifact_type"] == WIC_CVAE_ARTIFACT_TYPE
    assert artifact["artifact_type"] == WIC_CVAE_ARTIFACT_TYPE
    assert artifact["data_provenance"]["mask_source"]["kind"] == "manual"
    assert loaded.field_names == ("wic_corrected",)
    assert transform.channel_count == 1
    assert artifact["training_configuration"]["latent_samples"] == 4

    import scripts.CVAE.sample_wic_at_kp as sampler

    def fixed_precipitation(wic, dwic, kp, mlt):
        shape = np.broadcast_shapes(np.shape(wic), np.shape(mlt))
        wic = np.broadcast_to(wic, shape)
        return {
            "E0": np.full(shape, 2.0), "dE0": np.full(shape, 0.25),
            "Fe": wic / 100.0, "dFe": np.zeros(shape),
            "varE0Fe": np.zeros(shape),
            "E0_response_range_flag": np.zeros(shape, dtype=np.int8)
        }

    def fixed_fang(energy, flux, field, low_energy_policy="clip"):
        energy, flux, field = np.broadcast_arrays(energy, flux, field)
        valid = np.isfinite(energy) & np.isfinite(flux) & (flux >= 0)
        return {
            "H": np.where(valid, 2 * np.sqrt(flux), np.nan),
            "P": np.where(valid, np.sqrt(flux), np.nan), "valid": valid,
            "within_model_domain": np.isfinite(energy),
            "energy_clipped_low": np.zeros(energy.shape, dtype=bool),
            "evaluated_energy": energy
        }

    def fixed_fang_geometry(mlat, mlt, support, epoch):
        shape = np.shape(mlat)
        return {
            "modified_apex_longitude": np.full(shape, 1.0),
            "geographic_latitude": np.full(shape, 70.0),
            "geographic_longitude": np.full(shape, 20.0),
            "apex2geo_error_degrees": np.zeros(shape),
            "magnetic_field_microtesla": np.full(shape, 48.0),
            "epoch": sampler.parse_epoch(epoch).isoformat(),
            "reference_height_km": 130.0, "field_height_km": 150.0,
            "magnetic_field_model": "test-igrf",
            "magnetic_field_package_version": "test",
            "magnetic_field_coefficients_sha256": "0" * 64,
            "coordinate_mapping": "test mapping"
        }

    monkeypatch.setattr(sampler, "precipitation_from_collapsed_zhang_paxton", fixed_precipitation)
    monkeypatch.setattr(sampler, "fang_2025_conductance", fixed_fang)
    monkeypatch.setattr(sampler, "fang_magnetic_field_from_apex_grid", fixed_fang_geometry)
    sample_output = tmp_path / "samples"
    sample_arguments = sampler.parse_arguments([
        str(output), "--kp", "1", "3", "--output", str(sample_output),
        "--draws", "4", "--batch-size", "2", "--device", "cpu", "--no-plots"
    ])
    assert sample_arguments.hall_vmax == 24.0
    assert sample_arguments.pedersen_vmax == 12.0
    selections = sampler.run(sample_arguments)
    assert [row["Kp"] for row in selections] == [1.0, 3.0]
    with np.load(sample_output / "kp_00_1/ensemble_products.npz") as saved:
        assert saved["wic_mean"].shape == (46, 46)
        assert saved["wic_anomaly_factor"].shape == (46 * 46, 4)
        assert saved["wic_q50"].shape == (46, 46)
        assert saved["conductance_mean"].shape == (2, 46, 46)
        assert saved["conductance_anomaly_factor"].shape == (2 * 46 * 46, 4)
        assert saved["conductance_supported"].all()
        assert saved["electron_mean_energy"].shape == (46, 46)
        np.testing.assert_array_equal(saved["fang_magnetic_field_150km_microtesla"], 48.0)
        assert saved["fang_geographic_latitude_at_130km"].shape == (46, 46)
    sampling = json.loads((sample_output / "kp_00_1/sampling.json").read_text())
    assert sampling["fang_low_energy_policy"] == "clip"
    assert sampling["fang_magnetic_field_reference_epoch"] == "2001-01-01T00:00:00"
    assert sampling["fang_magnetic_field_coordinate_mapping"] == "test mapping"

    import scripts.CVAE.plot_wic_at_kp as plotter

    plotted = []
    monkeypatch.setattr(plotter, "plot_wic", lambda *args: plotted.append("wic"))
    monkeypatch.setattr(plotter, "plot_conductance", lambda *args: plotted.append("conductance"))
    plot_arguments = plotter.parse_arguments([str(sample_output)])
    assert plot_arguments.hall_vmax == 24.0
    assert plot_arguments.pedersen_vmax == 12.0
    assert plotter.run(plot_arguments) == ["kp_00_1", "kp_01_3"]
    assert plotted == ["wic", "conductance", "wic", "conductance"]


class FakeProduct:
    def __init__(self, arrays):
        self.product_type = "precipitation_cs"
        self.grid_id = "image_apex_130km_46x46_v1"
        self.schema_version = 1
        self.software_version = "test"
        self.shape = (2, 46, 46)
        self.time = np.arange(2).astype("datetime64[us]")
        self.Kp_interval_start = self.time.copy()
        self.source_indices = {"wic": np.array([4, 5])}
        self.mlat = np.zeros((46, 46))
        self.mlt = np.zeros((46, 46))
        self.attrs = {
            "method_quality_weight_method": "WIC auroral-quality field defined as 1 minus dgweight",
            "count_source": "background_subtracted", "source_preprocessing_label": "test",
            "proton_energy_model": "hardy", "proton_flux_source": "SI12"
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


def test_product2_wic_and_model2_mask_alignment(tmp_path):
    source = tmp_path / "or_0085.nc"
    source.write_bytes(b"source evidence")
    mask_directory = tmp_path / "masks"
    mask_directory.mkdir()
    mask = np.ones((2, 46, 46), dtype=bool)
    mask[0, 0, 0] = False
    np.savez(mask_directory / "or_0085.npz", time=np.arange(2).astype("datetime64[us]"), mask=mask)
    record = {
        "source": str(source), "source_bytes": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns, "frames": 2
    }
    shape = (2, 46, 46)
    arrays = {
        "wic_corrected": np.ones(shape, dtype=np.float32),
        "dwic_corrected": np.full(shape, 0.2, dtype=np.float32),
        "method_quality_weight": np.full(shape, 0.5, dtype=np.float32),
        "wic_coverage": np.ones(shape, dtype=np.float32),
        "Kp": np.array([1.0, 2.0], dtype=np.float32)
    }
    arrays["wic_corrected"][0, 0, 1] = -0.25
    arrays["wic_coverage"][1, 1, 1] = 0.4
    loaded = load_masked_wic(source, mask_directory, record, lambda _: FakeProduct(arrays))
    assert loaded["values"].shape == (2, 1, 46, 46)
    assert not loaded["valid"][0, 0, 0]
    assert loaded["valid"][1].all()
    assert loaded["values"][0, 0, 0, 1] == 0
    assert loaded["metadata"]["negative_wic_clipped_cells"] == 1
    assert loaded["quality"][1, 1, 1] == np.float32(0.2)
    np.testing.assert_array_equal(loaded["kp"], [1.0, 2.0])


def test_product2_wic_manual_annotation_alignment(tmp_path, monkeypatch):
    import scripts.CVAE.prepare_wic_cvae_cache as preparation

    source = tmp_path / "or_0085.nc"
    source.touch()
    shape = (2, 46, 46)
    arrays = {
        "wic_corrected": np.ones(shape, dtype=np.float32),
        "dwic_corrected": np.full(shape, 0.2, dtype=np.float32),
        "method_quality_weight": np.full(shape, 0.5, dtype=np.float32),
        "wic_coverage": np.ones(shape, dtype=np.float32),
        "Kp": np.array([1.0, 2.0], dtype=np.float32)
    }
    manual_mask = np.zeros(shape, dtype=bool)
    manual_mask[0, :20] = True
    classes = np.array(["partial", "excluded"])
    source_frames = np.array([7, -1])
    monkeypatch.setattr(
        preparation, "annotation_mask_arrays",
        lambda product, annotations: (manual_mask, classes, source_frames, 0)
    )
    loaded = load_annotated_wic(source, {"records": {}}, lambda _: FakeProduct(arrays))
    assert loaded["positive"].tolist() == [True, False]
    np.testing.assert_array_equal(loaded["valid"], manual_mask)
    np.testing.assert_array_equal(loaded["annotation_class"], classes)
    np.testing.assert_array_equal(loaded["annotation_source_frame"], source_frames)
    assert loaded["mask_evidence"]["kind"] == "manual"
    assert loaded["mask_evidence"]["selected_partial"] == 1
