"""Independent checks for conservative whole-frame release and supervision."""

import numpy as np
import pytest
import torch

from scripts.automatic_masking.model1_cnn import FullFrameClassifier, acceptance_labels, apply_frame_gate, binary_weights, route_two_stage, selection_curve


def test_two_stage_exporter_accepts_explicit_model_and_mask_lineage(tmp_path):
    from scripts.automatic_masking.apply_two_stage_masks import parse_arguments

    arguments = parse_arguments([
        "--baseline", str(tmp_path / "model2"),
        "--trial", str(tmp_path / "model1"),
        "--output", str(tmp_path / "combined"),
        "--orbits", "85", "86"
    ])
    assert arguments.baseline == tmp_path / "model2"
    assert arguments.trial == tmp_path / "model1"
    assert arguments.output == tmp_path / "combined"
    assert arguments.orbits == [85, 86]


def test_accepted_partial_targets_use_observed_wic_without_promoting_empty_or_bad_frames():
    labels = np.array(["partial", "partial", "partial", "partial", "bad", "full"])
    domains = np.ones((6, 2, 2), dtype=bool)
    domains[1, 0, 0] = False
    domains[3] = False
    targets = np.ones_like(domains)
    targets[1, 0, 0] = False  # Rejection outside the observed footprint is harmless.
    targets[2, 0, 0] = False  # A rejected observed pixel still needs masking.
    np.testing.assert_array_equal(acceptance_labels(labels, targets, domains),
        ["full", "full", "partial", "partial", "bad", "full"])
    np.testing.assert_array_equal(labels[:4], ["partial"] * 4)


def test_native_grid_classifier_has_one_trainable_frame_logit():
    torch.manual_seed(1426)
    model = FullFrameClassifier(in_channels=8)
    inputs = torch.randn(3, 8, 46, 46)
    logits = model(inputs)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.]))
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_binary_weights_equalize_classes_orbits_and_segments_within_split():
    rows = []
    for label, populations in (("full", ((1, 2), (5,))), ("bad", ((6,),)), ("partial", ((3, 7), (2,)))):
        for orbit, counts in enumerate(populations):
            for segment, nframes in enumerate(counts):
                rows.extend([dict(label=label, orbit=f"{label}-{orbit}", segment=f"{label}-{orbit}-{segment}")] * nframes)
    indices = np.arange(len(rows))
    # Rows not belonging to the training split must have no influence.
    rows += [dict(label="full", orbit="full-0", segment="full-0-0")] * 50
    weights = binary_weights(rows, indices)
    labels = np.array([rows[i]["label"] for i in indices])
    total = weights.sum()
    assert weights.mean() == pytest.approx(1.)
    for label, fraction in (("full", .5), ("bad", .25), ("partial", .25)):
        selected = labels == label
        assert weights[selected].sum() / total == pytest.approx(fraction)
        orbits = sorted({rows[i]["orbit"] for i in indices[selected]})
        for orbit in orbits:
            orbit_ids = np.array([rows[i]["orbit"] == orbit for i in indices])
            expected = fraction * total / len(orbits)
            assert weights[orbit_ids].sum() == pytest.approx(expected)
            segments = sorted({rows[i]["segment"] for i in indices[orbit_ids]})
            for segment in segments:
                segment_ids = np.array([rows[i]["segment"] == segment for i in indices])
                assert weights[segment_ids].sum() == pytest.approx(expected / len(segments))


def test_frame_gate_inclusive_boundary_keeps_nonpassed_pixels_exactly():
    current = np.array([[[False, True], [False, False]], [[False, False], [True, False]],
                        [[False, False], [False, False]], [[True, False], [True, True]]])
    boundary = np.float32(.9)
    scores = np.array([np.nextafter(boundary, np.float32(0)), boundary,
                       np.nextafter(boundary, np.float32(1)), .2], dtype=np.float32)
    gated, passed = apply_frame_gate(current, scores, float(boundary))
    np.testing.assert_array_equal(passed, [False, True, True, False])
    np.testing.assert_array_equal(gated[[0, 3]], current[[0, 3]])
    assert gated[[1, 2]].all()
    assert not np.any(current & ~gated)
    # Inference must not mutate stored masks.
    assert not current[1].all()
    fallback, passed = apply_frame_gate(current, scores, None)
    np.testing.assert_array_equal(fallback, current)
    assert not passed.any()


def test_two_stage_routing_thresholds_empty_frames_and_intrinsic_missing_pixels():
    current = np.zeros((5, 2, 2), dtype=bool)
    current[:, 0, 0] = True
    saved = current.copy()
    observed = np.ones_like(current)
    observed[3] = False
    accept = np.array([.8, .8, .79, .99, .99])
    combined, released = route_two_stage(current, accept, .8, observed)
    np.testing.assert_array_equal(released, [True, True, False, False, True])
    np.testing.assert_array_equal(combined[~released], saved[~released])
    np.testing.assert_array_equal(current, saved)
    assert combined[released].all() and not np.any(current & ~combined)
    intrinsic_valid = np.ones_like(current)
    intrinsic_valid[:, 1, 1] = False
    assert not (combined & intrinsic_valid)[:, 1, 1].any()
    accept[0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        route_two_stage(current, accept, .8, observed)


def exhaustive_reference(scores, labels, orbits, gains, target, frames, support):
    entries = []
    for threshold in sorted(set(scores.tolist()), reverse=True):
        passed = scores >= threshold
        count = int(passed.sum())
        full = int((passed & (labels == "full")).sum())
        entry = dict(threshold=float(threshold), accepted_frames=count,
                     accepted_orbits=len(np.unique(orbits[passed])), true_full=full,
                     false_full=count-full, precision=full/count,
                     good_gain=float(gains[passed].sum()))
        if entry["precision"] >= target and count >= frames and entry["accepted_orbits"] >= support and entry["good_gain"] > 0:
            entries.append(entry)
    return max(entries, key=lambda r: (r["good_gain"], r["precision"], r["threshold"])) if entries else None


def test_selection_matches_exhaustive_decisions_for_ties_and_support():
    rng = np.random.default_rng(1526)
    for _ in range(40):
        scores = rng.choice(np.array([.1, .5, .8, .9], dtype=np.float32), 61)
        labels = rng.choice(np.array(["full", "bad", "partial"]), 61, p=[.8, .1, .1])
        orbits = rng.integers(0, 7, 61)
        gains = rng.integers(0, 200, 61) * (labels == "full")
        target = float(rng.choice([.75, .9, .99]))
        curve, chosen = selection_curve(scores, labels, orbits, gains, target, 10, 3)
        assert len(curve) == len(np.unique(scores))
        for entry in curve:
            passed = scores >= entry["threshold"]
            assert entry["accepted_frames"] == int(passed.sum())
            assert entry["accepted_orbits"] == len(np.unique(orbits[passed]))
            assert entry["true_full"] == int((passed & (labels == "full")).sum())
            assert entry["false_full"] == int((passed & (labels != "full")).sum())
            assert entry["precision"] == pytest.approx((labels[passed] == "full").mean())
            assert entry["good_gain"] == pytest.approx(gains[passed].sum())
        expected = exhaustive_reference(scores, labels, orbits, gains, target, 10, 3)
        assert (chosen is None) == (expected is None)
        if chosen:
            for key in expected:
                assert chosen[key] == pytest.approx(expected[key])


def test_high_scoring_error_does_not_prevent_lower_threshold_becoming_feasible():
    # All 100 duplicate-score full frames must enter together. Their precision
    # becomes 100/101 > .99 despite the erroneous top-scored frame.
    scores = np.r_[.99, np.full(100, .9), .2]
    labels = np.array(["bad"] + ["full"] * 100 + ["partial"])
    orbits = np.arange(102) % 4
    gains = np.r_[0, np.ones(100), 0]
    curve, chosen = selection_curve(scores, labels, orbits, gains)
    assert len(curve) == 3
    assert chosen["threshold"] == .9
    assert chosen["accepted_frames"] == 101
    assert chosen["precision"] == pytest.approx(100/101)


@pytest.mark.parametrize("nframes,norbits,positive_gain", [(49, 3, True), (50, 2, True), (50, 3, False)])
def test_perfect_precision_is_insufficient_without_support_and_useful_recovery(nframes, norbits, positive_gain):
    scores = np.full(nframes, .9)
    labels = np.full(nframes, "full")
    orbits = np.arange(nframes) % norbits
    gains = np.ones(nframes) if positive_gain else np.zeros(nframes)
    _, chosen = selection_curve(scores, labels, orbits, gains)
    assert chosen is None


def test_exact_support_boundary_is_eligible():
    _, chosen = selection_curve(np.full(50, .9), np.full(50, "full"), np.arange(50) % 3, np.ones(50))
    assert chosen["accepted_frames"] == 50
    assert chosen["accepted_orbits"] == 3


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -.01, 1.01])
def test_invalid_scores_are_rejected(invalid):
    with pytest.raises(ValueError):
        selection_curve(np.array([.9, invalid]), np.array(["full", "full"]), np.array([1, 2]), np.ones(2))
