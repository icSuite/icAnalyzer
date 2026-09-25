import numpy as np

from icphysics.image import fWm

from icanalyzer.wic_si12.physics import (
    FROZEN_WIC_RESPONSE_AT_2_KEV,
    conductance_from_corrected_wic,
)


def test_frozen_2kev_response_and_reference_values():
    assert float(fWm(2.0)) == FROZEN_WIC_RESPONSE_AT_2_KEV
    wic = np.array([0.0, 47.75, 477.5, 1910.0])
    result = conductance_from_corrected_wic(wic)
    np.testing.assert_allclose(result["Fe"], [0.0, 0.1, 1.0, 4.0])
    np.testing.assert_allclose(
        result["P"], [0.0, 1.2649110640673518, 4.0, 8.0]
    )
    np.testing.assert_allclose(
        result["H"],
        [0.0, 1.0260015134870324, 3.244501665398989, 6.489003330797978],
    )


def test_ensemble_shape_nan_zero_and_uncertainty():
    wic = np.full((3, 46, 46), 477.5)
    wic[0, 0, 0] = np.nan
    wic[1, 0, 0] = 0.0
    result = conductance_from_corrected_wic(wic, np.full_like(wic, 47.75))
    assert result["H"].shape == wic.shape
    assert np.isnan(result["H"][0, 0, 0])
    assert result["H"][1, 0, 0] == 0.0
    np.testing.assert_allclose(result["dFe"][2], 0.1)


def test_negative_decoded_wic_is_rejected():
    try:
        conductance_from_corrected_wic(np.array([1.0, -0.01]))
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative corrected WIC was accepted")


def test_si12_is_not_reapplied_after_upstream_correction():
    wic = np.full((2, 2), 477.5)
    first = conductance_from_corrected_wic(wic)
    # SI12 is deliberately absent from the adapter signature.  Holding
    # corrected WIC fixed therefore gives one unambiguous conductance field.
    second = conductance_from_corrected_wic(wic.copy())
    np.testing.assert_array_equal(first["H"], second["H"])
    np.testing.assert_array_equal(first["P"], second["P"])
