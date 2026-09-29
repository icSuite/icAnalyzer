from pathlib import Path

import numpy as np
from types import MappingProxyType

from icanalyzer.wic_si12.data import (
    ChannelTransform,
    REQUIRED_CAMERA_FIELDS,
    fit_channel_transform,
    load_good_orbit_intervals,
    load_precipitation_cs,
    select_annotated_orbit_files,
    split_orbit_files,
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
        self.arrays["method_quality_weight"] = image.copy()
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
            [[[0.0, 10.0], [20.0, 30.0]], [[-2.0, 0.0], [2.0, 4.0]]],
            [[[5.0, 15.0], [25.0, 35.0]], [[-1.0, 1.0], [3.0, 5.0]]],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(values, dtype=bool)
    transform = fit_channel_transform(values, valid)
    transformed = transform.transform(values)
    restored = transform.inverse(transformed)
    np.testing.assert_allclose(restored, values, rtol=1e-5, atol=1e-5)
    variance = transform.variance(values, np.ones_like(values))
    assert np.isfinite(variance).all()
    assert (variance > 0).all()

    rebuilt = ChannelTransform.from_dict(transform.to_dict())
    np.testing.assert_allclose(rebuilt.inverse(transformed), values, atol=1e-5)


def test_orbit_split_has_no_overlap():
    paths = [Path(f"or_{orbit:04d}.nc") for orbit in range(10)]
    split = split_orbit_files(paths, validation_fraction=0.2, test_fraction=0.2, seed=4)
    assert len(split["train"]) == 6
    assert len(split["validation"]) == 2
    assert len(split["test"]) == 2
    all_paths = split["train"] + split["validation"] + split["test"]
    assert len(set(all_paths)) == len(paths)
