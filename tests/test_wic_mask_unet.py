"""Numerical and label-boundary checks for the annotation-mask experiment."""

import numpy as np
import torch

from scripts.manual_masking.annotation_store import encode_mask, propagated_mask
from scripts.automatic_masking.prepare_model2_training_data import frame_segments
from scripts.automatic_masking.model2_unet import PartialUNet, SmallUNet, joint_loss, masked_loss, pixel_scores, prepare_inputs, sample_weights, summarize, widen_for_context, window_inputs


def test_constant_targets_and_partial_masks_stop_at_full_interval():
    left = np.array([[True, False], [True, False]])
    right = ~left
    record = dict(orbit=1, n_frames=8, usable_start=1, usable_stop=6, full_good_start=3, full_good_stop=4, keyframes={"1": {"valid_runs": encode_mask(left)}, "5": {"valid_runs": encode_mask(right)}})
    classes = ["bad", "partial", "partial", "full", "full", "partial", "partial", "bad"]
    masks = [propagated_mask(record, frame, shape=(2, 2)) for frame in range(8)]
    assert not masks[0].any() and not masks[7].any()
    assert masks[3].all() and masks[4].all()
    np.testing.assert_array_equal(masks[2], left)
    np.testing.assert_array_equal(masks[6], right)
    segments = frame_segments(record, classes)
    assert segments[1] == segments[2] and segments[5] == segments[6]
    assert segments[1] != segments[5] and segments[0] != segments[7]


def test_class_balance_does_not_depend_on_number_of_propagated_frames():
    rows = [dict(label="full", orbit="1", segment="1:full:0") for _ in range(20)]
    rows += [dict(label="bad", orbit="2", segment="2:bad:0")]
    rows += [dict(label="partial", orbit="3", segment="3:partial:0") for _ in range(10)]
    rows += [dict(label="partial", orbit="3", segment="3:partial:10")]
    weights = sample_weights(rows, np.arange(len(rows)))
    totals = [sum(w for row, w in zip(rows, weights) if row["label"] == label) for label in ("full", "bad", "partial")]
    np.testing.assert_allclose(totals, [len(rows) / 3] * 3, rtol=1e-6)
    np.testing.assert_allclose(weights[21:31].sum(), weights[31], rtol=1e-6)


def test_missing_and_unreviewed_pixels_do_not_change_training_loss():
    logits = torch.zeros((1, 2, 2))
    target = torch.tensor([[[1., 0.], [0., 1.]]])
    domain = torch.tensor([[[1., 1.], [0., 0.]]])
    expected = masked_loss(logits, target, domain)
    logits[:, 1] = torch.tensor([100., -100.])
    target[:, 1] = 1 - target[:, 1]
    torch.testing.assert_close(masked_loss(logits, target, domain), expected)


def test_normalization_fits_training_only_and_preserves_missing_indicators():
    fields = ["wic_corrected", "method_quality_weight", "sza", "dza"]
    images = np.ones((3, 4, 2, 2), dtype=np.float32)
    images[2, :, 0, 0] = np.nan
    inputs, normalization = prepare_inputs(images, np.array([0, 1]), fields)
    changed = images.copy()
    changed[2, :, 1, 1] = 1e9
    _, other = prepare_inputs(changed, np.array([0, 1]), fields)
    assert normalization == other and inputs.shape == (3, 8, 2, 2)
    assert not inputs[2, 4:, 0, 0].any()
    assert inputs[2, 4:, 1, 1].all()


def test_all_reject_agreement_and_undefined_kept_area_are_reported_separately():
    masks = np.zeros((2, 2, 2), dtype=bool)
    domain = np.ones_like(masks)
    scores = pixel_scores(masks, masks, domain)
    result = summarize(scores, np.array([1, 2]))
    assert result["correction_fraction"]["orbit_mean"] == 0
    assert result["dice"]["orbit_mean"] == 1
    assert result["useful_discarded"]["orbit_mean"] is None


def test_missing_context_is_zero_and_relative_time_does_not_include_labels():
    inputs = np.ones((3, 8, 2, 2), dtype=np.float32)
    windows = np.array([[-1, 0, 1, 2, -1]])
    lags = np.array([[np.nan, -122, 0, 122, np.nan]], dtype=np.float32)
    result = window_inputs(inputs, windows, np.array([0]), lags)
    assert result.shape == (1, 45, 2, 2)
    assert not result[:, :9].any() and not result[:, 36:].any()
    np.testing.assert_array_equal(result[:, 18:26], inputs[1:2])
    np.testing.assert_allclose(result[:, 17], -122 / 300)


def test_temporal_model_starts_with_identical_center_predictions():
    import copy
    torch.manual_seed(12026)
    single = SmallUNet(8, class_head=True).eval()
    temporal = widen_for_context(copy.deepcopy(single), 5).eval()
    inputs = torch.randn((2, 45, 12, 12))
    with torch.no_grad():
        single_mask, single_class = single(inputs[:, 18:26], return_class=True)
        temporal_mask, temporal_class = temporal(inputs, return_class=True)
    torch.testing.assert_close(single_mask, temporal_mask)
    torch.testing.assert_close(single_class, temporal_class)


def test_classification_loss_reaches_shared_encoder_and_preserves_mask_loss():
    torch.manual_seed(12026)
    model = SmallUNet(8, class_head=True)
    mask_logits, class_logits = model(torch.randn((2, 8, 12, 12)), return_class=True)
    target = torch.ones_like(mask_logits)
    domain = torch.ones_like(mask_logits)
    labels = torch.tensor([0, 1])
    mask_only = joint_loss(mask_logits, class_logits, target, domain, labels, class_weight=0)
    torch.testing.assert_close(mask_only, masked_loss(mask_logits, target, domain))
    joint_loss(mask_logits, class_logits, target, domain, labels).mean().backward()
    assert model.classifier.weight.grad.abs().sum() > 0
    assert model.first[0].weight.grad.abs().sum() > 0


def test_mask_only_ablation_preserves_joint_initial_weights_and_shuffle_rng():
    torch.manual_seed(12026)
    joint = widen_for_context(SmallUNet(8, class_head=True), 5)
    joint_rng = torch.get_rng_state().clone()
    torch.manual_seed(12026)
    mask_only = widen_for_context(SmallUNet(8, match_joint_initialization=True), 5)
    assert torch.equal(torch.get_rng_state(), joint_rng)
    assert mask_only.classifier is None
    for name, parameter in mask_only.state_dict().items():
        torch.testing.assert_close(parameter, joint.state_dict()[name])
    inputs = torch.randn((2, 45, 12, 12))
    with torch.no_grad():
        mask, classes = mask_only(inputs, return_class=True)
        torch.testing.assert_close(mask, joint(inputs))
    assert classes is None


def test_partial_encoder_matches_standard_encoder_when_wic_is_fully_valid():
    torch.manual_seed(12026)
    standard = SmallUNet(8).eval()
    torch.manual_seed(12026)
    partial = PartialUNet().eval()
    for name, parameter in standard.state_dict().items():
        torch.testing.assert_close(parameter, partial.state_dict()[name])
    values = torch.randn((2, 8, 46, 46))
    values[:, 4] = 1
    with torch.no_grad():
        torch.testing.assert_close(partial(values), standard(values), atol=2e-6, rtol=2e-6)


def test_partial_encoder_excludes_missing_wic_sites_and_has_gradients():
    torch.manual_seed(12026)
    model = PartialUNet()
    values = torch.randn((2, 8, 46, 46))
    values[:, 4] = 1
    values[:, 4, 10:20, 10:20] = 0
    changed = values.clone()
    changed[:, :4, 10:20, 10:20] = 1000
    changed[:, 5:, 10:20, 10:20] = -1000
    torch.testing.assert_close(model(values), model(changed), atol=1e-6, rtol=0)
    model(values).mean().backward()
    assert torch.isfinite(model.first[0].weight.grad).all()
    assert model.first[0].weight.grad.abs().sum() > 0
