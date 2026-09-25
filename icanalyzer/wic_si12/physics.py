"""Compatibility adapter for fixed-energy conductance propagation.

The input is already proton-corrected WIC from ``precipitation_cs``.  This
module intentionally does not call ``proton_correct_images`` and does not
modify icPhysics or icBuilder.
"""

#%% Imports and frozen reference

import numpy as np

from icphysics import hall, ped, robinson_conductance
from icphysics.image import fWm


BASELINE_ELECTRON_ENERGY_KEV = 2.0
FROZEN_WIC_RESPONSE_AT_2_KEV = 477.5
PROPAGATION_ORDER = "grid_corrected_wic_then_fixed_energy_robinson_v1"


def _wic_response(electron_energy_keV):
    response = float(np.asarray(fWm(float(electron_energy_keV))))
    if not np.isfinite(response) or response <= 0:
        raise ValueError("WIC electron response must be finite and positive")
    if np.isclose(electron_energy_keV, BASELINE_ELECTRON_ENERGY_KEV):
        if not np.isclose(response, FROZEN_WIC_RESPONSE_AT_2_KEV, rtol=0, atol=1e-12):
            raise RuntimeError(
                "icPhysics WIC response at 2 keV changed from the frozen "
                f"{FROZEN_WIC_RESPONSE_AT_2_KEV} counts/(mW m^-2) to {response}"
            )
    return response


def conductance_from_corrected_wic(
    wic_corrected,
    dwic_corrected=None,
    electron_energy_keV=BASELINE_ELECTRON_ENERGY_KEV,
):
    """Convert already proton-corrected WIC counts to Hall/Pedersen fields.

    The nonlinear conversion is applied after detector fields have been
    reduced to the fixed CS grid.  It therefore must not be compared as if it
    were identical to ``conductance_cs``, which bins detector-space
    conductance after applying the nonlinear conversion.
    """

    if not np.isclose(electron_energy_keV, BASELINE_ELECTRON_ENERGY_KEV):
        raise ValueError("the accepted baseline requires electron_energy_keV=2.0")

    wic_corrected = np.asarray(wic_corrected, dtype=float)
    finite_negative = np.isfinite(wic_corrected) & (wic_corrected < 0)
    if np.any(finite_negative):
        raise ValueError("corrected WIC must be non-negative where finite")

    response = _wic_response(electron_energy_keV)
    electron_energy_flux = wic_corrected / response
    pedersen = ped(electron_energy_keV, electron_energy_flux)
    hall_conductance = hall(electron_energy_keV, electron_energy_flux)

    result = {
        "E0": np.full(wic_corrected.shape, electron_energy_keV, dtype=float),
        "Fe": electron_energy_flux,
        "P": pedersen,
        "H": hall_conductance,
        "wic_response": response,
        "propagation_order": PROPAGATION_ORDER,
    }

    if dwic_corrected is not None:
        dwic_corrected = np.asarray(dwic_corrected, dtype=float)
        if dwic_corrected.shape != wic_corrected.shape:
            raise ValueError("dwic_corrected must match wic_corrected shape")
        if np.any(np.isfinite(dwic_corrected) & (dwic_corrected < 0)):
            raise ValueError("corrected-WIC uncertainty must be non-negative")

        dFe = dwic_corrected / response
        zeros = np.zeros(wic_corrected.shape, dtype=float)
        uncertainty = robinson_conductance(
            result["E0"], electron_energy_flux, zeros, dFe, zeros
        )
        result.update({
            "dFe": dFe,
            "dP": uncertainty["dP"],
            "dH": uncertainty["dH"],
            "varE0Fe": zeros,
        })

    return result
