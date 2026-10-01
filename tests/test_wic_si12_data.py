from pathlib import Path

import numpy as np
from types import MappingProxyType

from icanalyzer.wic_si12.data import (
    ChannelTransform,
    REQUIRED_CAMERA_FIELDS,
    fit_channel_transform,
    fit_precision_weighting,
    load_good_orbit_intervals,
    load_precipitation_cs,
    load_precipitation_cs_files,
    select_annotated_orbit_files,
    split_orbit_files,
)
from icanalyzer.wic_si12.comparison import (
    exact_time_indices,
    load_paired_orbit,
    pair_orbit_files,
)


class FakeField:
    dimensions = ("time", "dim1", "dim2")
    units = "counts"


class FakeSZAField:
    dimensions = ("time", "dim1", "dim2")
    units = "degrees"


class FakeProduct:
    product_type = "precipitation_cs"
    grid_id = "image_apex_130km_46x46_v1"
    schema_version = 1
    software_version = "test"
    proton_flux_source = "SI12"
    proton_energy_model = "hardy"
    shape = (2, 46, 46)

    def __init__(self):
        image = np.ones(self.shape, dtype=np.float32)
        self.arrays = {name: image.copy() for name in REQUIRED_CAMERA_FIELDS}
        self.arrays["method_valid"] = np.ones(self.shape, dtype=bool)
        self.arrays["method_quality_weight"] = np.zeros(self.shape, dtype=np.float32)
        self.arrays["wic_coverage"] = image.copy()
        self.arrays["si12_coverage"] = image.copy()
        self.arrays["sza"] = np.full(self.shape, 110.0, dtype=np.float32)
        self.time = np.array(["t0", "t1"], dtype=object)
        grid_coordinate = np.zeros((46, 46), dtype=float)
        self.mlat = grid_coordinate
        self.mlt = grid_coordinate
        self.source_indices = MappingProxyType({
            sensor: np.array([0, 1]) for sensor in ("wic", "si12", "si13")
        })
        self.source_times = MappingProxyType({
            sensor: self.time for sensor in ("wic", "si12", "si13")
        })
        self.frame_quality = MappingProxyType({
            sensor: np.array([2, 2]) for sensor in ("wic", "si12", "si13")
        })
        self.attrs = {
            "proton_energy_uncertainty_constant": 0.0,
            "count_source": "background_subtracted",
            "spatial_smoothing_kernel": "none",
        }
        self.read_log = []
        for name in REQUIRED_CAMERA_FIELDS:
            setattr(self, name, FakeField())
        self.sza = FakeSZAField()

    def read(self, name, index=None):
        self.read_log.append((name, index))
        values = self.arrays[name]
        return values if index is None else values[index]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeLegacyProduct:
    product_type = "precipitation"
    schema_version = 2
    precipitation_method = "image_ratio"
    proton_flux_source = "SI12"
    proton_energy_model = "hardy"
    shape = (2, 46, 46)

    def __init__(self):
        image = np.ones(self.shape, dtype=np.float32)
        for name in REQUIRED_CAMERA_FIELDS:
            setattr(self, name, image.copy())
        self.wic_corrected[0] = 21.0
        self.wic_corrected[1] = 10.0
        self.si12[0] = 4.0
        self.si12[1] = 2.0
        self.w = np.full(self.shape, 0.25, dtype=np.float32)
        self.time = np.array(["t1", "t0"], dtype=object)
        self.mlat = np.zeros((46, 46), dtype=float)
        self.mlt = np.zeros((46, 46), dtype=float)
        self.source_indices = {
            sensor: np.array([101, 100])
            for sensor in ("wic", "si12", "si13")
        }


def test_loads_strong_product2_contract_and_preserves_corrected_wic():
    product = FakeProduct()
    product.arrays["wic_corrected"][:] = 477.5
    product.arrays["si12"][:] = 12.0

    loaded = load_precipitation_cs(
        "or_0042.nc", product_loader=lambda _path: product
    )

    assert loaded["values"].shape == (2, 2, 46, 46)
    np.testing.assert_allclose(loaded["values"][:, 0], 477.5)
    np.testing.assert_allclose(loaded["values"][:, 1], 12.0)
    assert loaded["metadata"]["orbit"] == 42
    assert loaded["metadata"]["proton_energy_model"] == "hardy"
    assert loaded["valid"].all()
    assert loaded["grid"]["mlat"].shape == (46, 46)
    np.testing.assert_array_equal(loaded["source_indices"]["wic"], [0, 1])


def test_invalid_product_cells_have_zero_quality():
    product = FakeProduct()
    product.arrays["method_valid"][:, 0, 0] = False
    product.arrays["method_quality_weight"][:, 0, 0] = np.nan
    product.arrays["wic_coverage"][:, 0, 0] = np.nan
    product.arrays["si12_coverage"][:, 0, 0] = np.nan

    loaded = load_precipitation_cs(
        "or_0042.nc", product_loader=lambda _path: product
    )

    assert not loaded["valid"][:, :, 0, 0].any()
    np.testing.assert_array_equal(loaded["quality"][:, :, 0, 0], 0.0)


def test_inverts_upstream_background_fit_weight_before_applying_coverage():
    product = FakeProduct()
    product.arrays["method_quality_weight"][:, 0, 0] = 0.0
    product.arrays["method_quality_weight"][:, 0, 1] = 1.0
    product.arrays["wic_coverage"][:, 0, 0] = 0.5

    loaded = load_precipitation_cs(
        "or_0042.nc", product_loader=lambda _path: product
    )

    assert loaded["valid"][:, :, 0, :2].all()
    np.testing.assert_allclose(loaded["quality"][:, 0, 0, 0], 0.5)
    np.testing.assert_allclose(loaded["quality"][:, 1, 0, 0], 1.0)
    np.testing.assert_allclose(loaded["quality"][:, :, 0, 1], 0.0)
    assert loaded["metadata"]["quality_weight_source"] == (
        "one_minus_method_quality_weight_times_channel_coverage"
    )
    assert loaded["metadata"]["upstream_quality_weight_interpretation"] == (
        "background_fit_weight"
    )


def test_negative_camera_values_are_clipped_and_reported_as_upstream_problem():
    product = FakeProduct()
    product.arrays["wic_corrected"][0, 0, 0] = -2.0
    product.arrays["si12"][0, 0, 1] = -3.0

    loaded = load_precipitation_cs(
        "or_0042.nc", product_loader=lambda _path: product
    )

    assert loaded["values"][0, 0, 0, 0] == 0.0
    assert loaded["values"][0, 1, 0, 1] == 0.0
    assert loaded["metadata"]["negative_input_policy"] == "clip_to_zero"
    assert loaded["metadata"]["negative_wic_clipped_cells"] == 1
    assert loaded["metadata"]["negative_si12_clipped_cells"] == 1
    assert loaded["metadata"]["negative_wic_clipped_valid_cells"] == 1
    assert loaded["metadata"]["negative_si12_clipped_valid_cells"] == 1
    assert loaded["metadata"]["wic_valid_input_cells"] == int(np.prod(product.shape))
    assert loaded["metadata"]["si12_valid_input_cells"] == int(np.prod(product.shape))


def test_manual_intervals_select_wic_source_frames_before_loading():
    product = FakeProduct()
    product.source_indices = MappingProxyType({
        "wic": np.array([10, 12]),
        "si12": np.array([20, 22]),
        "si13": np.array([30, 32]),
    })

    loaded = load_precipitation_cs(
        "or_0042.nc",
        product_loader=lambda _path: product,
        wic_source_interval=(12, 15),
    )

    assert loaded["values"].shape == (1, 2, 46, 46)
    assert loaded["source_indices"]["wic"].tolist() == [12]
    assert loaded["metadata"]["source_shape"] == (2, 46, 46)
    assert loaded["metadata"]["manual_frame_selection"] == {
        "wic_source_start": 12,
        "wic_source_stop": 15,
        "selected_product_frames": 1,
    }


def test_minimum_sza_masks_pixels_and_records_filter_provenance():
    product = FakeProduct()
    product.arrays["sza"][:, 0, 0] = 104.9
    product.arrays["sza"][:, 0, 1] = 105.0

    loaded = load_precipitation_cs(
        "or_0042.nc",
        product_loader=lambda _path: product,
        minimum_sza=105.0,
    )

    assert not loaded["valid"][:, :, 0, 0].any()
    assert loaded["valid"][:, :, 0, 1].all()
    np.testing.assert_array_equal(loaded["quality"][:, :, 0, 0], 0.0)
    assert loaded["metadata"]["minimum_sza_degrees"] == 105.0
    assert loaded["metadata"]["sza_filter_input_frames"] == 2
    assert loaded["metadata"]["sza_filter_retained_frames"] == 2
    assert loaded["metadata"]["sza_filter_retained_cells"] == 2 * 46 * 46 - 2
    assert loaded["metadata"]["sza_filter_total_cells"] == 2 * 46 * 46


def test_sza_empty_frames_are_removed_before_camera_fields_are_read():
    product = FakeProduct()
    product.arrays["sza"][0] = 104.0

    loaded = load_precipitation_cs(
        "or_0042.nc",
        product_loader=lambda _path: product,
        minimum_sza=105.0,
    )

    assert loaded["values"].shape == (1, 2, 46, 46)
    np.testing.assert_array_equal(loaded["source_indices"]["wic"], [1])
    assert loaded["metadata"]["sza_filter_input_frames"] == 2
    assert loaded["metadata"]["sza_filter_retained_frames"] == 1
    wic_reads = [index for name, index in product.read_log if name == "wic_corrected"]
    np.testing.assert_array_equal(wic_reads[0], [1])


def test_sza_empty_orbit_is_skipped_before_camera_fields_are_read():
    empty = FakeProduct()
    empty.arrays["sza"][:] = 104.0
    retained = FakeProduct()
    products = {"or_0041.nc": empty, "or_0042.nc": retained}

    loaded = load_precipitation_cs_files(
        ["or_0041.nc", "or_0042.nc"],
        minimum_sza=105.0,
        product_loader=lambda path: products[Path(path).name],
    )

    assert loaded["values"].shape == (2, 2, 46, 46)
    assert loaded["excluded_files"] == ["or_0041.nc"]
    assert [name for name, _index in empty.read_log] == ["sza"]


def test_rejects_invalid_minimum_sza():
    product = FakeProduct()
    for minimum_sza in (-1, 181, np.nan):
        try:
            load_precipitation_cs(
                "or_0042.nc",
                product_loader=lambda _path: product,
                minimum_sza=minimum_sza,
            )
        except ValueError as error:
            assert "minimum_sza" in str(error)
        else:
            raise AssertionError(f"invalid SZA threshold {minimum_sza} was accepted")


def test_loads_archive_segment_semantics_and_selects_orbit_files(tmp_path):
    segments = tmp_path / "segments.csv"
    segments.write_text(
        "42,,,3,5\n43,,,,8\n44,,,,0\n45,,,,\n",
        encoding="utf-8",
    )
    intervals = load_good_orbit_intervals(segments)
    assert intervals == {42: (3, 5), 43: (0, 8)}

    selected = select_annotated_orbit_files(
        [tmp_path / "or_0042.nc", tmp_path / "or_0043.nc"], intervals
    )
    assert [path.name for path in selected] == ["or_0042.nc", "or_0043.nc"]


def test_rejects_old_precipitation_cs_without_si12_pair():
    product = FakeProduct()
    delattr(product, "si12")
    delattr(product, "dsi12")

    try:
        load_precipitation_cs("or_0042.nc", product_loader=lambda _path: product)
    except ValueError as error:
        assert "missing VAE fields" in str(error)
    else:
        raise AssertionError("old Product-2 contract was accepted")


def test_requires_hardy_upstream_proton_energy():
    product = FakeProduct()
    product.proton_energy_model = "constant"
    product.proton_energy_constant = 2.0
    try:
        load_precipitation_cs("or_0042.nc", product_loader=lambda _path: product)
    except ValueError as error:
        assert "proton_energy_model='hardy'" in str(error)
    else:
        raise AssertionError("non-Hardy proton energy was accepted")


def test_channel_transform_round_trip_and_variance():
    values = np.array(
        [
            [[[0.0, 10.0], [20.0, 30.0]], [[2.0, 0.0], [2.0, 4.0]]],
            [[[5.0, 15.0], [25.0, 35.0]], [[1.0, 1.0], [3.0, 5.0]]],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(values, dtype=bool)
    transform = fit_channel_transform(values, valid)
    assert transform.kind == ChannelTransform.LOG10_KIND
    transformed = transform.transform(values)
    restored = transform.inverse(transformed)
    np.testing.assert_allclose(restored, values, rtol=1e-5, atol=1e-5)
    variance = transform.variance(values, np.ones_like(values))
    assert np.isfinite(variance).all()
    assert (variance > 0).all()

    rebuilt = ChannelTransform.from_dict(transform.to_dict())
    np.testing.assert_allclose(rebuilt.inverse(transformed), values, atol=1e-5)


def test_log10_transform_rejects_unclipped_negative_inputs():
    values = np.zeros((1, 2, 1, 1), dtype=np.float32)
    values[0, 1, 0, 0] = -1.0
    valid = np.ones_like(values, dtype=bool)

    try:
        fit_channel_transform(values, valid)
    except ValueError as error:
        assert "clip them at the ingestion boundary" in str(error)
    else:
        raise AssertionError("negative physical input reached the log10 transform")


def test_log10_inverse_clips_decoded_values_below_physical_zero():
    transform = ChannelTransform(
        scale=[1.0, 1.0], center=[0.0, 0.0], spread=[1.0, 1.0],
        kind=ChannelTransform.LOG10_KIND
    )
    transformed = np.full((1, 2, 1, 1), -1.0, dtype=np.float32)

    restored = transform.inverse(transformed)

    np.testing.assert_array_equal(restored, np.zeros_like(restored))


def test_fits_one_training_precision_scale_per_channel():
    values = np.zeros((2, 2, 2, 2), dtype=np.float32)
    standard_deviation = np.empty_like(values)
    standard_deviation[0, 0] = 1.0
    standard_deviation[1, 0] = 2.0
    standard_deviation[0, 1] = 2.0
    standard_deviation[1, 1] = 4.0
    product = {
        "values": values,
        "standard_deviation": standard_deviation,
        "valid": np.ones_like(values, dtype=bool),
        "quality": np.ones_like(values),
    }
    transform = ChannelTransform(
        scale=[1.0, 1.0], center=[0.0, 0.0], spread=[1.0, 1.0]
    )

    weighting = fit_precision_weighting(
        product,
        transform,
        variance_floor=1e-6,
        maximum_scale_samples=100,
    )

    assert weighting["kind"] == "training_median_precision_v1"
    np.testing.assert_allclose(weighting["precision_scale"], [0.625, 0.15625])
    assert weighting["frame_stride"] == 1
    assert weighting["scale_sample_count"] == [8, 8]
    np.testing.assert_allclose(
        [item["median"] for item in weighting["relative_precision_diagnostics"]],
        [1.0, 1.0],
    )
    assert weighting["loss_reduction"] == "mean_valid_support_per_sample_channel"


def test_orbit_split_has_no_overlap():
    paths = [Path(f"or_{orbit:04d}.nc") for orbit in range(10)]
    split = split_orbit_files(paths, validation_fraction=0.2, test_fraction=0.2, seed=4)
    assert len(split["train"]) == 6
    assert len(split["validation"]) == 2
    assert len(split["test"]) == 2
    all_paths = split["train"] + split["validation"] + split["test"]
    assert len(set(all_paths)) == len(paths)


def test_exact_time_pairing_follows_current_order():
    current, legacy = exact_time_indices(
        ["t0", "t1", "t2"], ["t2", "t0", "t3"]
    )
    np.testing.assert_array_equal(current, [0, 2])
    np.testing.assert_array_equal(legacy, [1, 0])


def test_pair_orbit_files_uses_only_common_filenames(tmp_path):
    current = tmp_path / "current"
    legacy = tmp_path / "legacy"
    current.mkdir()
    legacy.mkdir()
    for directory, names in (
        (current, ("or_0001.nc", "or_0002.nc")),
        (legacy, ("or_0002.nc", "or_0003.nc")),
    ):
        for name in names:
            (directory / name).touch()

    pairs = pair_orbit_files(sorted(current.glob("*.nc")), legacy)

    assert [(left.name, right.name) for left, right in pairs] == [
        ("or_0002.nc", "or_0002.nc")
    ]


def test_paired_legacy_uses_current_time_sza_and_inverted_legacy_weight():
    current = FakeProduct()
    current.arrays["sza"][0] = 104.0
    legacy = FakeLegacyProduct()

    paired = load_paired_orbit(
        "or_0042.nc",
        "legacy/or_0042.nc",
        data_version="both",
        minimum_sza=105.0,
        current_loader=lambda _path: current,
        legacy_loader=lambda _path: legacy,
    )

    assert paired["current"]["time"].tolist() == ["t1"]
    assert paired["legacy"]["time"].tolist() == ["t1"]
    np.testing.assert_allclose(paired["legacy"]["values"][:, 0], 21.0)
    np.testing.assert_allclose(paired["legacy"]["values"][:, 1], 4.0)
    np.testing.assert_allclose(paired["legacy"]["quality"], 0.75)
    assert paired["legacy"]["metadata"]["quality_weight_source"] == (
        "one_minus_legacy_combined_w"
    )
    assert paired["legacy"]["metadata"]["upstream_quality_weight_interpretation"] == (
        "background_fit_weight"
    )
    assert paired["legacy"]["metadata"]["paired_frames"] == 1
