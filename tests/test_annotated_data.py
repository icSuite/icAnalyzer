"""Focused checks of timestamp selection and paired manual masks."""

import numpy as np
import pytest

from scripts.manual_masking.annotated_data import apply_annotations, annotation_intervals, select_annotation_files
from scripts.manual_masking.annotation_store import encode_mask
from scripts.IMAGE.vae_pipeline.run_conductance_pipeline import parse_arguments
from scripts.IMAGE.vae_pipeline.run_wic_si12_pipeline import parse_arguments as parse_fuv
from scripts.manual_masking import native_annotations as native
from scripts.manual_masking import annotation_store as store


def test_full_partial_and_unmatched_frames():
    shape = (46, 46)
    footprint = np.ones(shape, bool)
    footprint[0] = False
    partial = footprint.copy()
    partial[:, 23:] = False
    times = np.array(["2000-01-01T00:00:00", "2000-01-01T00:02:00", "2000-01-01T00:04:00", "2000-01-01T00:06:00"], dtype="datetime64[us]")
    record = {
        "n_frames": 4, "review_status": "complete", "usable_start": 0, "usable_stop": 3,
        "full_good_start": 1, "full_good_stop": 1, "target_annotated_runs": encode_mask(footprint),
        "keyframes": {"0": {"valid_runs": encode_mask(partial)}, "2": {"valid_runs": encode_mask(partial)}}
    }
    # Native frame 0 is old frame 1, not old frame 0. The last native frame
    # has no original observation, so it must not borrow a nearby annotation.
    native_times = np.append(times[1:], times[-1] + np.timedelta64(120, "s"))
    product = {
        "values": np.ones((4, 2, *shape)), "standard_deviation": np.ones((4, 2, *shape)),
        "valid": np.ones((4, 2, *shape), bool), "quality": np.ones((4, 2, *shape)),
        "frame_missing_fraction": np.zeros(4), "time": native_times, "orbit": np.full(4, 85),
        "grid": {"mlat": np.zeros(shape), "mlt": np.zeros(shape)},
        "source_indices": {"wic": np.arange(4)}, "metadata": []
    }
    annotations = {"records": {85: record}, "source_times": {85: times}, "grid": product["grid"]}
    result = apply_annotations(product, annotations)
    np.testing.assert_array_equal(result["annotation_source_frame"], [1, 2, 3])
    np.testing.assert_array_equal(result["annotation_class"], ["full", "partial", "partial"])
    np.testing.assert_array_equal(result["valid"][0, 0], footprint)
    np.testing.assert_array_equal(result["valid"][1:, 0], np.stack([partial, partial]))
    np.testing.assert_array_equal(result["valid"][:, 0], result["valid"][:, 1])
    assert np.all(result["quality"][~result["valid"]] == 0)
    assert np.all(result["frame_missing_fraction"] == 0)
    assert result["annotation_selection"]["unmatched_timestamps"] == 1
    assert product["valid"].all()  # Caller input is preserved.
    del record["keyframes"]["2"]
    without_late_keyframe = apply_annotations(product, annotations)
    assert without_late_keyframe["annotation_class"].tolist() == ["full"]  # Never carry a cut across the full interval.


@pytest.mark.parametrize("parser", [parse_arguments, parse_fuv])
def test_annotation_option_is_not_default_or_combined_with_all_frames(parser):
    assert parser(["input", "--output", "output"]).annotations is None
    assert parser(["input", "--output", "output"]).hybrid_annotations is None
    args = parser(["input", "--output", "output", "--annotations", "converted.json"])
    assert args.annotations == "converted.json"
    with pytest.raises(SystemExit):
        parser(["input", "--output", "output", "--annotations", "converted.json", "--all-frames"])
    args = parser(["input", "--output", "output", "--hybrid-annotations", "converted.json"])
    assert args.hybrid_annotations == "converted.json" and args.annotations is None
    for conflicting in (["--all-frames"], ["--annotations", "another.json"]):
        with pytest.raises(SystemExit):
            parser(["input", "--output", "output", "--hybrid-annotations", "converted.json", *conflicting])


def test_hybrid_json_overrides_csv_and_csv_only_orbits_remain_unmasked():
    shape = (46, 46)
    footprint = np.ones(shape, bool)
    footprint[0] = False
    partial = footprint.copy()
    partial[:, 23:] = False
    times = np.array(["2000-01-01T00:00:00", "2000-01-01T00:02:00", "2000-01-01T00:04:00"], dtype="datetime64[us]")
    record = {
        "n_frames": 3, "review_status": "complete", "usable_start": 0, "usable_stop": 1,
        "full_good_start": 0, "full_good_stop": 0, "target_annotated_runs": encode_mask(footprint),
        "keyframes": {"1": {"valid_runs": encode_mask(partial)}}
    }
    product = {
        "values": np.ones((6, 2, *shape)), "standard_deviation": np.ones((6, 2, *shape)),
        "valid": np.ones((6, 2, *shape), bool), "quality": np.ones((6, 2, *shape)),
        "frame_missing_fraction": np.zeros(6), "time": np.concatenate([times, times]),
        "orbit": np.array([85, 85, 85, 86, 86, 87]),
        "source_indices": {"wic": np.array([100, 101, 102, 10, 11, 10])},
        "grid": {"mlat": np.zeros(shape), "mlt": np.zeros(shape)}, "metadata": []
    }
    annotations = {
        "records": {85: record, 88: {"review_status": "no_usable"}}, "source_times": {85: times},
        "csv_intervals": {85: (102, 102), 86: (10, 10), 88: (0, 10)}, "grid": product["grid"]
    }
    intervals = annotation_intervals(annotations)
    assert intervals[85] is None and intervals[86] == (10, 10)
    assert select_annotation_files(["or_0085.nc", "or_0086.nc", "or_0087.nc", "or_0088.nc"], annotations) == ["or_0085.nc", "or_0086.nc"]
    result = apply_annotations(product, annotations)
    np.testing.assert_array_equal(result["annotation_class"], ["full", "partial", "csv"])
    np.testing.assert_array_equal(result["source_indices"]["wic"], [100, 101, 10])
    np.testing.assert_array_equal(result["valid"][0, 0], footprint)
    np.testing.assert_array_equal(result["valid"][1, 0], partial)
    assert result["valid"][2].all() and np.all(result["quality"][2] == 1)
    assert result["annotation_selection"]["selected_csv"] == 1
    product["quality"][3] = 0
    result = apply_annotations(product, annotations)
    assert result["annotation_class"].tolist() == ["full", "partial", "csv"]
    assert result["valid"][2].all() and np.all(result["quality"][2] == 0)


def test_native_rebase_preserves_dropped_keyframe_mask_and_review_footprint(monkeypatch):
    times = np.datetime64("2000-01-01T00:00:00", "us") + np.arange(5) * np.timedelta64(120, "s")
    footprint = np.ones((46, 46), bool)
    footprint[0] = False
    partial = footprint.copy()
    partial[:, 23:] = False
    old = {"orbit": 85, "n_frames": 5, "review_status": "complete", "usable_start": 0, "usable_stop": 4, "full_good_start": 1, "full_good_stop": 2,
           "target_annotated_runs": encode_mask(footprint), "keyframes": {"0": {"valid_runs": encode_mask(partial)}, "3": {"valid_runs": encode_mask(partial)}}}
    document = {"orbits": {"0085": old}}
    monkeypatch.setattr(native, "annotation_times", lambda record: times)
    # The late mask's original keyframe (3) is absent in the native file.
    indices = np.array([1, 2, 4])
    record = native.native_orbit_record(document, 85, times[indices], indices)
    assert record["frame_index_space"] == "native_cs" and record["mask_shape"] == [46, 46]
    assert record["review_status"] == "complete" and record["full_good_stop"] == 1
    assert "legacy_annotation" not in record
    assert all(payload.get("human_reviewed") is False for payload in record["keyframes"].values())
    for frame, original in enumerate(indices):
        expected = store.propagated_mask(old, original, shape=(46, 46)) & footprint
        actual = store.propagated_mask(record, frame) & store.decode_mask(record["target_annotated_runs"], shape=(46, 46))
        np.testing.assert_array_equal(actual, expected)
    assert "2" in record["keyframes"]  # Initial late-partial keyframe survives.


def test_annotation_times_are_embedded():
    from scripts.manual_masking.annotated_data import annotation_times

    times = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[us]")
    record = {"orbit": 85, "n_frames": 2, "source_times_utc": np.datetime_as_string(times, unit="us").tolist()}
    np.testing.assert_array_equal(annotation_times(record), times)


def test_new_native_review_uses_source_ids_and_roundtrips_46x46_masks(tmp_path):
    document = store.empty_document()
    document.update(schema_version=3, mask_shape=[46, 46])
    times = np.datetime64("2000-01-01", "us") + np.arange(3) * np.timedelta64(120, "s")
    record = native.native_orbit_record(document, 85, times, np.array([99, 102, 108]), seed=(100, 105))
    assert record["full_good_start"] == record["full_good_stop"] == 1
    assert store.decode_mask(record["target_annotated_runs"], shape=(46, 46)).all()
    store.set_boundary(record, "usable_start", 0)
    mask = np.ones((46, 46), bool)
    mask[:, 23:] = False
    store.set_keyframe(record, 0, mask)
    store.mark_complete(record)
    native.stamp_native_record(record, times)
    path = tmp_path / "annotations.json"
    store.save_document_atomic(document, path)
    loaded = store.load_document(path)["orbits"]["0085"]
    np.testing.assert_array_equal(store.propagated_mask(loaded, 0), mask)
    assert loaded["boundary_times"]["usable_start"] == str(times[0])
    fresh = store.new_orbit_record(86, 1)
    store.set_keyframe(fresh, 0, np.ones((46, 46), bool))
    assert store.exact_keyframe_mask(fresh, 0).shape == (46, 46)


def test_native_GUI_mask_edits_save_and_feed_both_pipeline_options(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from scripts.manual_masking import annotate_partial_images as gui

    (tmp_path / "or_0085.nc").touch()
    times = np.datetime64("2000-01-01", "us") + np.arange(3) * np.timedelta64(120, "s")
    channels = tuple(np.ones((3, 46, 46)) for _ in range(5))
    sza = np.broadcast_to(np.linspace(80, 115, 46), (3, 46, 46)).copy()
    monkeypatch.setattr(gui, "load_native_images", lambda *args: (channels, ("Hall", "Pedersen", "auroral quality", "coverage", "corrected WIC"), times, np.arange(3), sza))
    path = tmp_path / "annotations.json"
    document = store.empty_document()
    document.update(schema_version=3, mask_shape=[46, 46], cursor={"orbit": 84, "page_start": 0, "frame": 0})
    store.save_document_atomic(document, path)
    viewer = gui.PartialImageAnnotator(tmp_path, None, path, tmp_path / "missing.csv")
    assert viewer.orbit == 85  # Missing converted cursor orbit does not block startup.
    store.set_boundary(viewer.record, "usable_start", 0)
    store.set_boundary(viewer.record, "usable_stop", 2)
    editor = gui.MaskEditor(viewer, 0)
    assert len(editor.axes) == 5 and len(editor.images) == 5
    for axis in editor.axes:
        assert len(axis.collections) == 2
        assert all(artist.get_paths() for artist in axis.collections)
    viewer.refresh()
    viewer.refresh()
    for axis in viewer.axes[:3]:
        assert len(axis.collections) == 2
    for _ in range(4):
        viewer.cycle_channel()
    assert viewer.channel_names[viewer.channel_index] == "corrected WIC"
    editor._lasso_selected([(-1, -1), (23, -1), (23, 47), (-1, 47), (-1, -1)])
    editor.save()
    store.mark_complete(viewer.record)
    viewer.save()
    loaded = store.load_document(path)["orbits"]["0085"]
    np.testing.assert_array_equal(store.exact_keyframe_mask(loaded, 0), editor.mask)
    product = {"values": np.ones((3, 2, 46, 46)), "standard_deviation": np.ones((3, 2, 46, 46)), "valid": np.ones((3, 2, 46, 46), bool), "quality": np.ones((3, 2, 46, 46)), "frame_missing_fraction": np.zeros(3), "time": times, "orbit": np.full(3, 85), "grid": {"mlat": np.zeros((46, 46)), "mlt": np.zeros((46, 46))}}
    annotations = {"records": {85: loaded}, "source_times": {85: times}, "grid": product["grid"]}
    for parser in (parse_arguments, parse_fuv):
        arguments = parser(["input", "--output", "output", "--annotations", str(path)])
        assert arguments.annotations == str(path)
        result = apply_annotations(product, annotations)
        np.testing.assert_array_equal(result["valid"][:, 0], np.broadcast_to(editor.mask, (3, 46, 46)))
        np.testing.assert_array_equal(result["valid"][:, 0], result["valid"][:, 1])
    plt.close(editor.figure)
    plt.close(viewer.figure)


def test_conductance_WIC_companion_keeps_pixels_outside_retrieval_mask(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    source = tmp_path / "conductance_cs/grid/method/robinson/or_0085.nc"
    companion = tmp_path / "precipitation_cs/grid/method/or_0085.nc"
    companion.parent.mkdir(parents=True)
    companion.touch()
    shape = (2, 46, 46)
    times = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[us]")
    coords = np.zeros((46, 46))
    valid = np.ones(shape, bool)
    valid[:, 0] = False
    fields = {name: np.ones(shape) for name in ("H", "P", "method_quality_weight", "coverage")}
    fields["conductance_valid"] = valid
    common = dict(grid_id="image_apex_130km_46x46_v1", time=times, mlat=coords, mlt=coords)
    conductance = SimpleNamespace(**common, product_type="conductance_cs", shape=shape, source_indices={"wic": np.arange(2)}, attrs={"companion_precipitation_cs": "/producer/missing.nc"}, read=fields.__getitem__)
    precipitation = SimpleNamespace(**common, read=lambda name: np.full(shape, 7.0))
    monkeypatch.setattr(native, "load", lambda filename: nullcontext(precipitation if filename == companion else conductance))
    channels, names, _, _, sza = native.load_native_images(source, {})
    assert np.all(sza == 7)
    assert names[-1] == "corrected WIC" and len(channels) == 5
    assert np.isnan(channels[0][:, 0]).all()
    assert np.all(channels[-1][:, 0] == 7)
    precipitation.time = times + np.timedelta64(1, "s")
    with pytest.raises(ValueError, match="frame times"):
        native.load_native_images(source, {})


def test_sza_contours_skip_missing_and_uncrossed_levels():
    import matplotlib.pyplot as plt
    from scripts.manual_masking.annotate_partial_images import draw_sza_contours

    figure, axis = plt.subplots()
    for values in (None, np.full((46, 46), np.nan), np.full((46, 46), 90.0)):
        draw_sza_contours(axis, values)
    assert len(axis.collections) == 0
    values = np.broadcast_to(np.linspace(85, 100, 46), (46, 46))
    draw_sza_contours(axis, values)
    assert len(axis.collections) == 1
    np.testing.assert_allclose(axis.collections[0].get_paths()[0].vertices[:, 0], 15)
    plt.close(figure)


def test_annotation_navigation_skips_empty_images_without_marking_rejected(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from scripts.manual_masking import annotate_partial_images as gui

    times = np.datetime64("2000-01-01", "us") + np.arange(2) * np.timedelta64(120, "s")
    for orbit in (85, 86, 87, 88):
        (tmp_path / f"or_{orbit:04d}.nc").touch()
    def images(source, document):
        empty = source.name == "or_0086.nc"
        channels = tuple(np.full((2, 46, 46), np.nan if empty and i in (0, 1, 4) else 1.0) for i in range(5))
        return channels, ("Hall", "Pedersen", "auroral quality", "coverage", "corrected WIC"), times, np.arange(2), None
    monkeypatch.setattr(gui, "load_native_images", images)
    monkeypatch.setattr(native, "annotation_times", lambda record: times)
    document = store.empty_document()
    document.update(schema_version=3, mask_shape=[46, 46], cursor={"orbit": 86})
    record = native.native_orbit_record(document, 87, times, np.arange(2), seed=(0, 1))
    store.mark_complete(record)
    path = tmp_path / "annotations.json"
    store.save_document_atomic(document, path)
    viewer = gui.PartialImageAnnotator(tmp_path, None, path, tmp_path / "missing.csv")
    assert viewer.orbit == 87
    assert "0086" not in viewer.document["orbits"]
    viewer.next_unreviewed()
    assert viewer.orbit == 88
    viewer.previous_orbit()
    assert viewer.orbit == 87  # Completed reviews remain accessible explicitly.
    viewer.previous_orbit()
    assert viewer.orbit == 85  # Quality/coverage alone must not keep an empty orbit.
    assert "0086" not in viewer.document["orbits"]
    inspect = gui.PartialImageAnnotator(tmp_path, None, path, tmp_path / "missing.csv", start_orbit=86, show_empty_orbits=True)
    assert inspect.orbit == 86
    plt.close("all")
