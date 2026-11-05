"""Core preparation checks retained after removing input-selection experiments."""

import numpy as np

from scripts.automatic_masking.model1_cnn import channels_for
from scripts.automatic_masking.model2_unet import prepare_inputs
from scripts.automatic_masking.prepare_model1_training_data import physical_transform


def test_channel_selection_preserves_parent_order_and_finite_flags():
    normalization = {
        "fields": ["wic_corrected", "grid/mlt_sin", "grid/mlt_cos", "dza", "sza"],
        "parent_fields": ["wic_corrected", "grid/mlt", "grid/mlt", "dza", "sza"]
    }
    assert channels_for(["sza", "grid/mlt", "wic_corrected"], normalization) == [4, 1, 2, 0, 9, 6, 7, 5]


def test_core_physical_transforms_match_model2_preparation():
    fields = ["wic_corrected", "method_quality_weight", "sza", "dza"]
    images = np.arange(3 * 4 * 2 * 2, dtype=np.float32).reshape(3, 4, 2, 2)
    images[2, :, 0, 0] = np.nan
    train = np.array([0, 1])
    reference, normalization = prepare_inputs(images, train, fields)
    changed = images.copy()
    changed[2, :, 1, 1] = 1e7
    _, changed_normalization = prepare_inputs(changed, train, fields)
    assert changed_normalization == normalization
    for index, field in enumerate(fields):
        physical, observed, _ = physical_transform(images[:, index], field)
        normalized = np.nan_to_num(
            np.clip(
                (physical - normalization["centre"][index]) / normalization["scale"][index],
                -6, 6
            ),
            nan=0
        )
        np.testing.assert_array_equal(normalized, reference[:, index])
        np.testing.assert_array_equal(observed, reference[:, index + len(fields)])
