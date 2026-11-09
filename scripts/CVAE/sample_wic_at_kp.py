"""Generate conditional WIC-CVAE and Zhang--Paxton/Fang ensembles."""

#%% Imports

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from icphysics import (
    FANG_2025_ENERGY_RANGE_KEV,
    FANG_2025_MODEL_DOI,
    fang_2025_conductance,
    precipitation_from_collapsed_zhang_paxton
)
from icanalyzer.cvae.artifacts import WIC_CVAE_ARTIFACT_TYPE, load_run_artifact
from icanalyzer.cvae.conditioning import (
    CONDITIONING_FEATURE_NAMES,
    conditioning_vector_from_record,
    resolve_conditioning_features,
)
from icanalyzer.cvae.ensemble import ensemble_statistics, sample_decoded_ensemble
from icanalyzer.cvae.regimes import (
    CONTROLLED_SML_BANDS,
    CONTROLLED_SML_LABELS,
    REGIME_NAMES,
    controlled_regime_conditions,
    empirical_regime_masks,
    load_training_conditions,
)
from icanalyzer.cvae.summaries import summarize_wic_samples


#%% Fang configuration

IMAGE_APEX_REFERENCE_HEIGHT_KM = 130.0
FANG_MAGNETIC_FIELD_HEIGHT_KM = 150.0
FANG_REFERENCE_EPOCH = datetime(2001, 1, 1)
FANG_REFERENCE_EPOCH_POLICY = "fixed round epoch within the 2000--2003 training period"


def parse_epoch(value):
    """Parse an ISO epoch and return a timezone-free UTC datetime."""

    if isinstance(value, datetime):
        epoch = value
    else:
        try:
            epoch = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as error:
            raise argparse.ArgumentTypeError("Fang epoch must be an ISO date or datetime") from error
    if epoch.tzinfo is not None:
        epoch = epoch.astimezone(timezone.utc).replace(tzinfo=None)
    return epoch


def polar_contour(
    axis, mlat, mlt, values, maximum, title,
    coordinate_labels=True, title_fontsize=None
):
    """Plot one saved 46x46 field using Hardy-comparison polar contours."""

    from polplot import Polarplot

    values = np.asarray(values, dtype=float)
    if values.shape != np.shape(mlat) or not np.isfinite(values).any():
        raise ValueError("polar contour values must align with the finite MLAT/MLT grid")
    polar_axis = Polarplot(axis, minlat=50)
    contour = polar_axis.contourf(
        mlat, mlt, values.copy(), levels=np.linspace(0, maximum, 17),
        vmin=0, vmax=maximum, cmap="viridis", extend="max"
    )
    if coordinate_labels:
        polar_axis.writeLTlabels(fontsize=7)
        polar_axis.writeLATlabels(fontsize=6)
    axis.set_title(title, fontsize=title_fontsize)
    return contour


def plot_wic(mean, quantiles, support, mlat, mlt, title, output, maximum):
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.4), layout="constrained")
    fields = [mean, *quantiles]
    labels = ("mean", "10th percentile", "median", "90th percentile")
    for axis, values, label in zip(axes, fields, labels):
        values = np.where(support, values, np.nan)
        contour = polar_contour(axis, mlat, mlt, values, maximum, label)
    figure.colorbar(contour, ax=axes, shrink=0.72, label="corrected WIC [counts]")
    figure.suptitle(f"Corrected-WIC model prior: {title}; common latent seed")
    figure.savefig(output / "wic_at_kp.png", dpi=160)
    figure.savefig(output / "wic_at_kp.pdf")
    plt.close(figure)


def plot_wic_sample_mosaic(
    samples, sample_indices, support, mlat, mlt, title, output, maximum
):
    """Plot up to 36 coherent WIC realizations as a polar 6x6 mosaic."""

    samples = np.asarray(samples, dtype=float)
    sample_indices = np.asarray(sample_indices, dtype=int)
    if (
        samples.ndim != 3 or samples.shape[1:] != np.shape(support)
        or len(samples) < 1 or len(samples) > 36
        or sample_indices.shape != (len(samples),)
    ):
        raise ValueError("WIC mosaic needs 1--36 indexed fields on the saved grid")
    figure, axes = plt.subplots(
        6, 6, figsize=(12, 12), layout="constrained"
    )
    flat_axes = axes.ravel()
    for axis, values, sample_index in zip(flat_axes, samples, sample_indices):
        values = np.where(support, values, np.nan)
        contour = polar_contour(
            axis, mlat, mlt, values, maximum,
            f"sample {sample_index + 1}", coordinate_labels=False,
            title_fontsize=6
        )
    for axis in flat_axes[len(samples):]:
        axis.set_axis_off()
    figure.colorbar(
        contour, ax=axes, shrink=0.72, label="corrected WIC [counts]"
    )
    figure.suptitle(f"Corrected-WIC individual realizations: {title}")
    figure.savefig(output / "wic_sample_mosaic.png", dpi=160)
    figure.savefig(output / "wic_sample_mosaic.pdf")
    plt.close(figure)


def plot_wic_medoid(
    medoid, medoid_index, support, mlat, mlt, title, output, maximum
):
    """Plot one coherent central ensemble member."""

    figure, axis = plt.subplots(figsize=(4.6, 4.2), layout="constrained")
    values = np.where(support, medoid, np.nan)
    contour = polar_contour(
        axis, mlat, mlt, values, maximum,
        f"sample {int(medoid_index) + 1}"
    )
    figure.colorbar(contour, ax=axis, shrink=0.72, label="corrected WIC [counts]")
    figure.suptitle(f"Central coherent WIC realization: {title}")
    figure.savefig(output / "wic_medoid.png", dpi=160)
    figure.savefig(output / "wic_medoid.pdf")
    plt.close(figure)


def plot_wic_exceedance_probability(
    probability, threshold, support, mlat, mlt, title, output
):
    """Plot the marginal probability of exceeding one fixed WIC threshold."""

    figure, axis = plt.subplots(figsize=(4.8, 4.2), layout="constrained")
    values = np.where(support, probability, np.nan)
    contour = polar_contour(
        axis, mlat, mlt, values, 1.0,
        f"P(WIC exceeds {threshold:g} counts)"
    )
    figure.colorbar(contour, ax=axis, shrink=0.72, label="probability")
    figure.suptitle(f"WIC exceedance probability: {title}")
    figure.savefig(output / "wic_exceedance_probability.png", dpi=160)
    figure.savefig(output / "wic_exceedance_probability.pdf")
    plt.close(figure)


def plot_conductance(mean, quantiles, support, mlat, mlt, title, output, limits):
    figure, axes = plt.subplots(2, 4, figsize=(13, 6.2), layout="constrained")
    fields = [mean, *quantiles]
    labels = ("mean", "10th percentile", "median", "90th percentile")
    for channel, name in enumerate(("Hall", "Pedersen")):
        for column, (field, label) in enumerate(zip(fields, labels)):
            values = np.where(support, field[channel], np.nan)
            contour = polar_contour(
                axes[channel, column], mlat, mlt, values, limits[channel], label
            )
        axes[channel, 0].set_ylabel(name)
        figure.colorbar(contour, ax=axes[channel], shrink=0.72, label=f"{name} [S]")
    figure.suptitle(f"WIC → collapsed Zhang–Paxton → Fang: {title}")
    figure.savefig(output / "conductance_at_kp.png", dpi=160)
    figure.savefig(output / "conductance_at_kp.pdf")
    plt.close(figure)


#%% Physics propagation

def fang_magnetic_field_from_apex_grid(
    mlat, mlt, support, epoch,
    reference_height_km=IMAGE_APEX_REFERENCE_HEIGHT_KM,
    field_height_km=FANG_MAGNETIC_FIELD_HEIGHT_KM
):
    """Reconstruct geographic footprints and ApexPy IGRF for a timeless grid.

    Product 3 normally uses geographic coordinates already stored in Product 2.
    A WIC-CVAE artifact contains only the fixed Modified-Apex latitude/MLT grid,
    so this sampler reconstructs the equivalent geographic footprint at the
    product reference height for one explicitly recorded reference epoch.
    """

    from apexpy import Apex

    mlat = np.asarray(mlat, dtype=float)
    mlt = np.asarray(mlt, dtype=float)
    support = np.asarray(support, dtype=bool)
    epoch = parse_epoch(epoch)
    if mlat.shape != mlt.shape or support.shape != mlat.shape or mlat.ndim != 2:
        raise ValueError("MLAT, MLT and support must share a two-dimensional grid")
    if not np.isfinite(reference_height_km) or reference_height_km <= 0:
        raise ValueError("Apex reference height must be positive and finite")
    if not np.isfinite(field_height_km) or field_height_km <= 0:
        raise ValueError("Fang magnetic-field height must be positive and finite")
    selected = support & np.isfinite(mlat) & np.isfinite(mlt)
    if not selected.any():
        raise ValueError("no finite Modified-Apex coordinates exist on WIC support")

    apex = Apex(date=epoch, refh=float(reference_height_km))
    mlon = np.full(mlat.shape, np.nan)
    glat = np.full(mlat.shape, np.nan)
    glon = np.full(mlat.shape, np.nan)
    mapping_error = np.full(mlat.shape, np.nan)
    magnetic_field = np.full(mlat.shape, np.nan)

    mlon[selected] = apex.mlt2mlon(mlt[selected], epoch)
    latitude, longitude, error = apex.apex2geo(
        mlat[selected], mlon[selected], float(reference_height_km)
    )
    glat[selected] = latitude
    glon[selected] = longitude
    mapping_error[selected] = error
    magnetic_field[selected] = np.asarray(
        apex.get_babs(latitude, longitude, float(field_height_km)), dtype=float
    ) * 1e6
    if np.any(~np.isfinite(magnetic_field[selected])) or np.any(magnetic_field[selected] <= 0):
        raise ValueError("ApexPy IGRF produced an invalid field on WIC support")

    coefficients = Path(apex.igrf_fn)
    return {
        "modified_apex_longitude": mlon,
        "geographic_latitude": glat,
        "geographic_longitude": glon,
        "apex2geo_error_degrees": mapping_error,
        "magnetic_field_microtesla": magnetic_field,
        "epoch": epoch.isoformat(),
        "reference_height_km": float(reference_height_km),
        "field_height_km": float(field_height_km),
        "magnetic_field_model": coefficients.stem,
        "magnetic_field_package_version": version("apexpy"),
        "magnetic_field_coefficients_sha256": hashlib.sha256(coefficients.read_bytes()).hexdigest(),
        "coordinate_mapping": (
            "fixed-grid MLT converted to Modified-Apex longitude at the reference epoch; "
            f"Modified-Apex coordinates converted to geographic coordinates at {reference_height_km:g} km; "
            f"ApexPy IGRF evaluated at {field_height_km:g} km at the same geographic latitude/longitude"
        )
    }


def fang_conductance_from_wic(
    wic_samples, kp, mlt, model_support, magnetic_field_microtesla,
    low_energy_policy, batch_size
):
    """Propagate sampled corrected WIC through collapsed ZP and Fang.

    Unit corrected-WIC is passed through the public icPhysics precipitation
    function once. The resulting flux per count makes the nominal conversion
    linear in each WIC draw and permits physical zero-WIC/zero-flux samples.
    The collapsed energy spread is retained as a structural diagnostic and is
    not sampled as predictive uncertainty.
    """

    wic_samples = np.asarray(wic_samples, dtype=np.float32)
    mlt = np.asarray(mlt, dtype=float)
    model_support = np.asarray(model_support, dtype=bool)
    if wic_samples.ndim != 3 or wic_samples.shape[1:] != mlt.shape:
        raise ValueError("WIC samples and MLT grid do not align")
    if model_support.shape != mlt.shape or not model_support.any():
        raise ValueError("WIC model support must be nonempty and match the grid")
    if np.any(~np.isfinite(wic_samples[:, model_support])) or np.any(wic_samples[:, model_support] < 0):
        raise ValueError("corrected-WIC samples must be finite and nonnegative on model support")
    magnetic_field_microtesla = np.asarray(magnetic_field_microtesla, dtype=float)
    try:
        magnetic_field_microtesla = np.broadcast_to(magnetic_field_microtesla, mlt.shape)
    except ValueError as error:
        raise ValueError("Fang magnetic field must broadcast to the model grid") from error
    if np.any(~np.isfinite(magnetic_field_microtesla[model_support])) or np.any(magnetic_field_microtesla[model_support] <= 0):
        raise ValueError("Fang magnetic field must be positive and finite on model support")
    if batch_size < 1:
        raise ValueError("physics batch size must be positive")

    kp = np.asarray(kp, dtype=np.float32)
    scalar_kp = kp.ndim == 0
    if scalar_kp:
        sample_kp = np.full(len(wic_samples), kp, dtype=np.float32)
    elif kp.shape == (len(wic_samples),):
        sample_kp = kp
    else:
        raise ValueError("Kp must be a scalar or one value per WIC sample")
    if not np.isfinite(sample_kp).all() or np.any((sample_kp < 0) | (sample_kp > 9)):
        raise ValueError("sample Kp must be finite and lie within 0--9")
    unique_kp, sample_kp_index = np.unique(sample_kp, return_inverse=True)
    unique_shape = (len(unique_kp), *mlt.shape)
    unit_precipitation = precipitation_from_collapsed_zhang_paxton(
        np.ones(unique_shape), np.zeros(unique_shape),
        unique_kp[:, None, None], mlt[None]
    )
    energy = np.asarray(unit_precipitation["E0"], dtype=np.float64)
    energy_spread = np.asarray(unit_precipitation["dE0"], dtype=np.float64)
    flux_per_count = np.asarray(unit_precipitation["Fe"], dtype=np.float64)
    response_flag = np.asarray(unit_precipitation["E0_response_range_flag"], dtype=np.int8)
    reference = fang_2025_conductance(
        energy, np.zeros(energy.shape), magnetic_field_microtesla[None],
        low_energy_policy=low_energy_policy
    )
    valid_by_kp = model_support[None] & np.asarray(reference["valid"], dtype=bool)
    conductance_support = np.all(valid_by_kp, axis=0)
    if not conductance_support.any():
        raise ValueError("collapsed Zhang--Paxton/Fang has no valid cells on WIC support")

    conductance = np.zeros((len(wic_samples), 2, *mlt.shape), dtype=np.float32)
    for start in range(0, len(wic_samples), batch_size):
        stop = min(start + batch_size, len(wic_samples))
        lookup = sample_kp_index[start:stop]
        batch_energy = energy[lookup]
        energy_flux = wic_samples[start:stop] * flux_per_count[lookup]
        propagated = fang_2025_conductance(
            batch_energy, energy_flux, magnetic_field_microtesla,
            low_energy_policy=low_energy_policy
        )
        hall = np.asarray(propagated["H"], dtype=np.float32)
        pedersen = np.asarray(propagated["P"], dtype=np.float32)
        if np.any(~np.isfinite(hall[:, conductance_support])) or np.any(~np.isfinite(pedersen[:, conductance_support])):
            raise ValueError("Fang produced nonfinite conductance on supported cells")
        conductance[start:stop, 0, conductance_support] = hall[:, conductance_support]
        conductance[start:stop, 1, conductance_support] = pedersen[:, conductance_support]

    diagnostics = {
        "electron_mean_energy": energy,
        "electron_energy_structural_spread": energy_spread,
        "electron_energy_flux_per_wic_count": flux_per_count,
        "electron_response_range_flag": response_flag,
        "fang_within_model_domain": np.asarray(reference["within_model_domain"], dtype=bool),
        "fang_energy_clipped_low": np.asarray(reference["energy_clipped_low"], dtype=bool),
        "fang_evaluated_energy": np.asarray(reference["evaluated_energy"], dtype=np.float64),
        "conductance_supported": conductance_support,
        "unique_kp": unique_kp,
        "sample_kp_index": sample_kp_index.astype(np.int16),
    }
    if scalar_kp:
        for name in (
            "electron_mean_energy", "electron_energy_structural_spread",
            "electron_energy_flux_per_wic_count", "electron_response_range_flag",
            "fang_within_model_domain", "fang_energy_clipped_low",
            "fang_evaluated_energy"
        ):
            diagnostics[name] = diagnostics[name][0]
    return conductance, diagnostics


def _named_selections(filename, model):
    contents = json.loads(Path(filename).expanduser().read_text())
    records = contents.get("conditions") if isinstance(contents, dict) else contents
    if not isinstance(records, list) or not records:
        raise ValueError("conditions JSON must contain a nonempty list")
    selections = []
    labels = set()
    for number, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("every condition selection must be an object")
        label = str(record.get("label", f"condition_{number:02d}"))
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_.-")
        if not safe_label or safe_label in labels:
            raise ValueError("condition labels must be distinct safe directory names")
        labels.add(safe_label)
        names = resolve_conditioning_features(model.conditioning["feature_names"])
        vector = conditioning_vector_from_record(record, feature_names=names)
        if "kp" not in record:
            raise ValueError("condition record needs Kp for conductance propagation")
        conceptual_names = tuple(model.conditioning["conceptual_features"])
        lower = np.asarray(model.conditioning["training_min"])
        upper = np.asarray(model.conditioning["training_max"])
        outside = [
            name for name, value, minimum, maximum in zip(
                names, vector, lower, upper
            ) if value < minimum or value > maximum
        ]
        selections.append({
            "mode": "fixed_conditions",
            "label": label, "directory": safe_label,
            "conditions": {
                name: float(record[name])
                for name in dict.fromkeys((*conceptual_names, "kp"))
            },
            "vector": vector, "outside_training_columns": outside,
        })
    return selections


def _regime_selections(paths, names, model, artifact, arguments):
    """Load actual finite training rows and form physical IMF regime pools."""

    model_names = resolve_conditioning_features(
        model.conditioning["feature_names"]
    )
    model_columns = [CONDITIONING_FEATURE_NAMES.index(name) for name in model_names]
    rows = load_training_conditions(paths[0], paths[1])
    evidence = rows["provenance"]
    expected_cache = artifact["data_provenance"].get("cache_manifest_sha256")
    expected_conditioning = (
        artifact["data_provenance"].get("conditioning") or {}
    ).get("manifest_sha256")
    if evidence["cache_manifest_sha256"] != expected_cache:
        raise ValueError("regime cache differs from the cache used to train this model")
    if evidence["conditioning_manifest_sha256"] != expected_conditioning:
        raise ValueError("regime conditioning table differs from model training")

    masks, definitions, standard_clock = empirical_regime_masks(
        rows["matrix"], field_minimum=arguments.regime_field_minimum,
        field_maximum=arguments.regime_field_maximum,
        clock_half_width=arguments.regime_clock_half_width,
        maximum_absolute_tilt=arguments.regime_tilt_limit
    )
    model_finite = np.all(np.isfinite(rows["matrix"][:, model_columns]), axis=1)
    kp_column = CONDITIONING_FEATURE_NAMES.index("kp")
    kp_finite = np.isfinite(rows["matrix"][:, kp_column])
    selections = []
    for name in names:
        selected = masks[name] & model_finite & kp_finite
        count = int(selected.sum())
        if count < 2:
            raise ValueError(f"{name} has fewer than two matching training rows")
        selections.append({
            "mode": "empirical_regime", "label": name.replace("_", " ").title(),
            "directory": name, "conditions": None,
            "vector": rows["matrix"][selected][:, model_columns],
            "pool_full_vector": rows["matrix"][selected],
            "pool_kp": rows["matrix"][selected, kp_column],
            "pool_standard_clock": standard_clock[selected].astype(np.float32),
            "pool_orbit": rows["orbit"][selected],
            "pool_source_frame": rows["source_frame"][selected],
            "pool_time": rows["time"][selected],
            "pool_size": count, "definition": definitions[name],
            "pool_provenance": evidence, "outside_training_columns": [],
        })
    return selections


def _controlled_regime_selections(regimes, sml_bands, model, arguments):
    """Construct the prescribed four-condition climatology experiment."""

    model_names = resolve_conditioning_features(
        model.conditioning["feature_names"]
    )
    expected_names = resolve_conditioning_features("hybrid4")
    if model_names != expected_names:
        raise ValueError(
            "controlled IMF/SML sampling requires the hybrid4 WIC-CVAE"
        )
    lower = np.asarray(model.conditioning["training_min"])
    upper = np.asarray(model.conditioning["training_max"])
    condition_seed = arguments.seed + 97
    selections = []
    for sml_band in sml_bands:
        for regime in regimes:
            matrix, standard_clock, definition = controlled_regime_conditions(
                regime, sml_band, arguments.draws, condition_seed,
                field_minimum=arguments.regime_field_minimum,
                field_maximum=arguments.regime_field_maximum,
                clock_half_width=arguments.regime_clock_half_width,
                dipole_tilt=arguments.controlled_tilt,
            )
            outside = [
                name for column, name in enumerate(model_names)
                if (
                    matrix[:, column].min() < lower[column]
                    or matrix[:, column].max() > upper[column]
                )
            ]
            label = (
                f"{CONTROLLED_SML_LABELS[sml_band]}; "
                f"{regime.replace('_', ' ').title()}"
            )
            selections.append({
                "mode": "controlled_regime",
                "label": label,
                "directory": f"{sml_band}_{regime}",
                "conditions": {
                    "sml_band": sml_band,
                    "dipole_tilt": float(arguments.controlled_tilt),
                    "kp": float(arguments.conversion_kp),
                },
                "vector": matrix,
                "standard_clock": standard_clock,
                "condition_seed": condition_seed,
                "definition": {
                    "imf_regime": regime,
                    **definition,
                    "conductance_conversion_kp": float(
                        arguments.conversion_kp
                    ),
                },
                "outside_training_columns": outside,
            })
    return selections


def _condition_summary(matrix, standard_clock):
    """Summarize the empirical condition rows used for one ensemble."""

    matrix = np.asarray(matrix, dtype=float)
    values = {
        name: matrix[:, CONDITIONING_FEATURE_NAMES.index(name)]
        for name in (
            "sml", "smr", "by_gsm", "bz_gsm", "vx_gse",
            "dynamic_pressure", "newell", "kp", "dipole_tilt"
        )
    }
    values["standard_clock_angle"] = np.asarray(standard_clock, dtype=float)
    day_phase = np.mod(np.arctan2(
        matrix[:, CONDITIONING_FEATURE_NAMES.index("day_of_year_sin")],
        matrix[:, CONDITIONING_FEATURE_NAMES.index("day_of_year_cos")]
    ), 2 * np.pi)
    values["day_of_year"] = 1 + day_phase * 365.25 / (2 * np.pi)
    quantiles = (0.0, 0.1, 0.5, 0.9, 1.0)
    return {
        name: dict(zip(
            ("minimum", "q10", "median", "q90", "maximum"),
            np.quantile(column, quantiles).tolist()
        ))
        for name, column in values.items()
    }


def _controlled_condition_summary(matrix, feature_names, standard_clock, kp):
    """Summarize the explicitly designed condition ensemble."""

    matrix = np.asarray(matrix, dtype=float)
    values = {
        name: matrix[:, column]
        for column, name in enumerate(feature_names)
    }
    values["standard_clock_angle"] = np.asarray(standard_clock, dtype=float)
    values["kp"] = np.full(len(matrix), kp, dtype=float)
    quantiles = (0.0, 0.1, 0.5, 0.9, 1.0)
    return {
        name: dict(zip(
            ("minimum", "q10", "median", "q90", "maximum"),
            np.quantile(column, quantiles).tolist()
        ))
        for name, column in values.items()
    }


def run(arguments):
    limits = np.asarray((arguments.hall_vmax, arguments.pedersen_vmax), dtype=float)
    if arguments.draws < 2 or arguments.threads < 1 or not np.isfinite(arguments.wic_vmax) or arguments.wic_vmax <= 0:
        raise ValueError("need at least two draws, positive threads and a positive WIC display limit")
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("conductance display limits must be finite and positive")

    run_directory = arguments.run_directory.expanduser().resolve()
    artifact_path = run_directory / "wic_cvae.pt"
    output = arguments.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("choose an empty WIC-sampling output directory")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if arguments.device == "auto" else arguments.device
    torch.set_num_threads(arguments.threads)
    model, transform, artifact = load_run_artifact(artifact_path, device=device)
    if artifact.get("artifact_type") != WIC_CVAE_ARTIFACT_TYPE or tuple(model.field_names) != ("wic_corrected",):
        raise ValueError("requires a one-field WIC-CVAE artifact")
    if transform.channel_count != 1:
        raise ValueError("requires a one-channel WIC transform")
    if model.conditioning is not None:
        if arguments.controlled_regimes:
            names = REGIME_NAMES if arguments.regime is None else arguments.regime
            sml_bands = (
                tuple(CONTROLLED_SML_BANDS)
                if arguments.sml_band is None else arguments.sml_band
            )
            requested = _controlled_regime_selections(
                names, sml_bands, model, arguments
            )
        elif arguments.regimes is not None:
            names = REGIME_NAMES if arguments.regime is None else arguments.regime
            requested = _regime_selections(
                arguments.regimes, names, model, artifact, arguments
            )
        elif arguments.conditions is not None:
            requested = _named_selections(arguments.conditions, model)
        else:
            raise ValueError("named-conditioned models require --regimes or --conditions")
    elif model.kp_conditioning is not None:
        if arguments.kp is None or arguments.conditions is not None:
            raise ValueError("Kp-conditioned models require --kp")
        kp_values = np.asarray(arguments.kp, dtype=float)
        if (
            kp_values.ndim != 1 or not len(kp_values)
            or not np.isfinite(kp_values).all()
            or np.any((kp_values < 0) | (kp_values > 9))
        ):
            raise ValueError("requested Kp values must be finite and lie in 0--9")
        if len(np.unique(kp_values)) != len(kp_values):
            raise ValueError("requested Kp values must be distinct")
        requested = [{
            "mode": "fixed_kp",
            "label": f"Kp {kp:g}", "directory": f"kp_{index:02d}_{kp:g}",
            "conditions": {"kp": float(kp)}, "vector": None,
            "outside_training_columns": (
                [] if model.kp_conditioning["training_min"] <= kp
                <= model.kp_conditioning["training_max"] else ["kp"]
            ),
        } for index, kp in enumerate(kp_values)]
    else:
        raise ValueError("requires a conditional WIC-CVAE artifact")

    provenance = artifact["data_provenance"]
    support = np.asarray(provenance["training_support"], dtype=bool)
    mlat = np.asarray(provenance["grid"]["mlat"], dtype=float)
    mlt = np.asarray(provenance["grid"]["mlt"], dtype=float)
    if support.shape != tuple(model.image_shape) or mlat.shape != support.shape or mlt.shape != support.shape or not support.any():
        raise ValueError("saved WIC support and grid do not align with the model")
    fang_geometry = fang_magnetic_field_from_apex_grid(
        mlat, mlt, support, arguments.fang_epoch
    )
    model_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)

    selections = []
    for selection in requested:
        if selection["mode"] == "empirical_regime":
            condition_seed = arguments.seed + 97
            condition_rng = np.random.default_rng(condition_seed)
            pool_indices = condition_rng.integers(
                0, selection["pool_size"], size=arguments.draws
            )
            condition_values = selection["vector"][pool_indices]
            full_condition_values = selection["pool_full_vector"][pool_indices]
            standard_clock = selection["pool_standard_clock"][pool_indices]
            sample_orbit = selection["pool_orbit"][pool_indices]
            sample_source_frame = selection["pool_source_frame"][pool_indices]
            sample_time = selection["pool_time"][pool_indices]
            sample_kp = selection["pool_kp"][pool_indices]
            kp = sample_kp
        elif selection["mode"] == "controlled_regime":
            condition_seed = selection["condition_seed"]
            pool_indices = np.asarray([], dtype=np.int64)
            condition_values = selection["vector"]
            full_condition_values = np.asarray([], dtype=np.float32)
            standard_clock = selection["standard_clock"]
            sample_orbit = np.asarray([], dtype=np.int32)
            sample_source_frame = np.asarray([], dtype=np.int32)
            sample_time = np.asarray([], dtype="datetime64[us]")
            kp = selection["conditions"]["kp"]
            sample_kp = np.full(arguments.draws, kp, dtype=np.float32)
        else:
            condition_seed = None
            pool_indices = np.asarray([], dtype=np.int64)
            condition_values = selection["vector"]
            full_condition_values = np.asarray([], dtype=np.float32)
            standard_clock = np.asarray([], dtype=np.float32)
            sample_orbit = np.asarray([], dtype=np.int32)
            sample_source_frame = np.asarray([], dtype=np.int32)
            sample_time = np.asarray([], dtype="datetime64[us]")
            kp = selection["conditions"]["kp"]
            sample_kp = np.full(arguments.draws, kp, dtype=np.float32)
        within_range = not selection["outside_training_columns"]
        if not within_range:
            warnings.warn(
                "requested conditions extend outside the WIC-CVAE training range: "
                + ", ".join(selection["outside_training_columns"]),
                RuntimeWarning, stacklevel=2
            )
        destination = output / selection["directory"]
        destination.mkdir()
        print(
            f"{selection['label']}: generating {arguments.draws} corrected-WIC fields",
            flush=True
        )
        condition_arguments = (
            {"conditioning": condition_values}
            if model.conditioning is not None else {"kp": kp}
        )
        samples, latent = sample_decoded_ensemble(
            model, transform, arguments.draws, batch_size=arguments.batch_size,
            device=device, seed=arguments.seed, return_latent=True,
            support_mask=support, **condition_arguments
        )
        wic = samples[:, 0]
        if not np.isfinite(wic[:, support]).all() or np.any(wic[:, support] < 0):
            raise ValueError("decoder produced invalid corrected WIC on learned support")
        wic_quantiles = np.quantile(wic, [0.1, 0.5, 0.9], axis=0)
        wic_quantiles[:, ~support] = np.nan
        mosaic_count = min(36, len(wic))
        mosaic_indices = np.arange(mosaic_count, dtype=np.int32)
        mosaic_samples = wic[mosaic_indices].copy()
        distribution_summary = summarize_wic_samples(
            wic, support, mlat, mlt,
            threshold=arguments.wic_exceedance_threshold
        )
        wic_statistics = ensemble_statistics(samples)
        conductance, physics = fang_conductance_from_wic(
            wic, kp, mlt, support, fang_geometry["magnetic_field_microtesla"],
            arguments.fang_low_energy_policy, arguments.batch_size
        )
        conductance_quantiles = np.quantile(conductance, [0.1, 0.5, 0.9], axis=0)
        conductance_quantiles[:, :, ~physics["conductance_supported"]] = np.nan
        conductance_statistics = ensemble_statistics(conductance)
        np.savez_compressed(
            destination / "ensemble_products.npz", latent_samples=latent,
            wic_mean=wic_statistics["mean"][0], wic_anomaly_factor=wic_statistics["anomaly_factor"],
            wic_supported=support, wic_sample_kp=sample_kp,
            wic_q10=wic_quantiles[0], wic_q50=wic_quantiles[1], wic_q90=wic_quantiles[2],
            wic_mosaic_samples=mosaic_samples,
            wic_mosaic_sample_indices=mosaic_indices,
            wic_medoid=distribution_summary["medoid"],
            wic_medoid_sample_index=distribution_summary["medoid_index"],
            wic_medoid_feature_names=np.asarray(
                distribution_summary["medoid_feature_names"], dtype="U40"
            ),
            wic_medoid_feature_values=distribution_summary[
                "medoid_feature_values"
            ],
            wic_exceedance_probability=distribution_summary[
                "exceedance_probability"
            ],
            wic_exceedance_threshold=distribution_summary[
                "exceedance_threshold"
            ],
            conductance_mean=conductance_statistics["mean"],
            conductance_anomaly_factor=conductance_statistics["anomaly_factor"],
            conductance_supported=physics["conductance_supported"],
            conductance_sample_kp=sample_kp,
            conductance_q10=conductance_quantiles[0],
            conductance_q50=conductance_quantiles[1],
            conductance_q90=conductance_quantiles[2],
            electron_mean_energy=physics["electron_mean_energy"],
            electron_energy_structural_spread=physics["electron_energy_structural_spread"],
            electron_energy_flux_per_wic_count=physics["electron_energy_flux_per_wic_count"],
            electron_response_range_flag=physics["electron_response_range_flag"],
            fang_within_model_domain=physics["fang_within_model_domain"],
            fang_energy_clipped_low=physics["fang_energy_clipped_low"],
            fang_evaluated_energy=physics["fang_evaluated_energy"],
            physics_unique_kp=physics["unique_kp"],
            physics_sample_kp_index=physics["sample_kp_index"],
            fang_modified_apex_longitude=fang_geometry["modified_apex_longitude"],
            fang_geographic_latitude_at_130km=fang_geometry["geographic_latitude"],
            fang_geographic_longitude_at_130km=fang_geometry["geographic_longitude"],
            fang_apex2geo_error_degrees=fang_geometry["apex2geo_error_degrees"],
            fang_magnetic_field_150km_microtesla=fang_geometry["magnetic_field_microtesla"],
            conditioning_feature_names=np.asarray(
                model.conditioning["feature_names"]
                if condition_values is not None and model.conditioning is not None
                else (),
                dtype="U32"
            ),
            conditioning_values=np.asarray(
                condition_values if condition_values is not None else (),
                dtype=np.float32
            ),
            conditioning_full_feature_names=np.asarray(
                CONDITIONING_FEATURE_NAMES
                if selection["mode"] == "empirical_regime" else (),
                dtype="U32"
            ),
            conditioning_full_values=full_condition_values,
            conditioning_standard_clock_angle=standard_clock,
            conditioning_pool_index=pool_indices,
            conditioning_source_orbit=sample_orbit,
            conditioning_source_frame=sample_source_frame,
            conditioning_source_time=sample_time,
            mlat=mlat, mlt=mlt
        )
        if selection["mode"] == "empirical_regime":
            interpretation = (
                "Empirical-regime WIC prior propagated member-by-member to Fang "
                "electron conductance. Unrestricted nuisance conditions are "
                "marginalized over actual finite training rows and each member's "
                "Kp is used in the physical conversion."
            )
        elif selection["mode"] == "controlled_regime":
            interpretation = (
                "Designed-condition WIC prior with SML and IMF sampled "
                "independently and uniformly inside explicit bounds, fixed "
                "dipole tilt and fixed Kp used only for conductance conversion. "
                "No empirical condition-row distribution is used."
            )
        else:
            interpretation = (
                "Exact-condition WIC prior propagated member-by-member to Fang "
                "electron conductance."
            )
        interpretation += (
            " The fixed reference epoch approximates frame-specific Product-3 "
            "IGRF because this climatological artifact has no observation time. "
            "No proton/solar-EUV contribution, WIC measurement-noise propagation, "
            "sampled Zhang--Paxton structural spread or calibrated uncertainty claim."
        )
        metadata = {
            "label": selection["label"],
            "Kp": None if selection["mode"] == "empirical_regime" else kp,
            "sample_kp_summary": {
                "minimum": float(np.min(sample_kp)),
                "median": float(np.median(sample_kp)),
                "maximum": float(np.max(sample_kp)),
            },
            "within_training_range": bool(within_range), "draws": arguments.draws,
            "sample_mosaic": {
                "selection": "first 36 paired ensemble members",
                "saved_members": int(mosaic_count),
                "sample_indices_zero_based": mosaic_indices.tolist(),
            },
            "distribution_summaries": {
                "medoid_sample_index_zero_based": distribution_summary[
                    "medoid_index"
                ],
                "medoid_method": distribution_summary["medoid_method"],
                "medoid_feature_names": list(
                    distribution_summary["medoid_feature_names"]
                ),
                "medoid_feature_values": distribution_summary[
                    "medoid_feature_values"
                ].tolist(),
                "exceedance_threshold_counts": distribution_summary[
                    "exceedance_threshold"
                ],
            },
            "seed": arguments.seed, "device": device, "model_sha256": model_sha,
            "model_file": str(artifact_path),
            "kp_conditioning": (
                None if model.kp_conditioning is None
                else dict(model.kp_conditioning)
            ),
            "conditioning": (
                None if model.conditioning is None else dict(model.conditioning)
            ),
            "requested_conditions": selection["conditions"],
            "conditioning_regime": selection.get("definition"),
            "conditioning_pool_frames": selection.get("pool_size"),
            "conditioning_pool_provenance": selection.get("pool_provenance"),
            "conditioning_sampling": (
                {
                    "method": "uniform empirical training-row resampling with replacement",
                    "seed": condition_seed,
                    "sample_specific_kp_used_for_conductance": True,
                    "condition_summary": _condition_summary(
                        full_condition_values, standard_clock
                    ),
                }
                if selection["mode"] == "empirical_regime" else
                {
                    "method": "independent uniform designed-condition sampling",
                    "seed": condition_seed,
                    "sample_specific_kp_used_for_conductance": False,
                    "fixed_conductance_conversion_kp": float(kp),
                    "condition_summary": _controlled_condition_summary(
                        condition_values,
                        model.conditioning["feature_names"],
                        standard_clock, kp
                    ),
                }
                if selection["mode"] == "controlled_regime" else None
            ),
            "outside_training_columns": selection["outside_training_columns"],
            "grid_id": provenance["grid_id"], "supported_cells": int(support.sum()),
            "conductance_supported_cells": int(physics["conductance_supported"].sum()),
            "field_names": ["wic_corrected", "Hall", "Pedersen"],
            "field_units": ["counts", "S", "S"],
            "electron_energy_model": "flux-weighted collapsed Zhang--Paxton over 50--90 degrees MLAT",
            "electron_conductance_model": "Fang 2025 electron-impact Hall/Pedersen",
            "fang_model_doi": FANG_2025_MODEL_DOI,
            "fang_published_energy_range_keV": list(FANG_2025_ENERGY_RANGE_KEV),
            "fang_low_energy_policy": arguments.fang_low_energy_policy,
            "fang_magnetic_field_model": fang_geometry["magnetic_field_model"],
            "fang_magnetic_field_package_version": fang_geometry["magnetic_field_package_version"],
            "fang_magnetic_field_coefficients_sha256": fang_geometry["magnetic_field_coefficients_sha256"],
            "fang_magnetic_field_reference_epoch": fang_geometry["epoch"],
            "fang_magnetic_field_epoch_policy": (
                FANG_REFERENCE_EPOCH_POLICY if arguments.fang_epoch == FANG_REFERENCE_EPOCH
                else "user-specified fixed reference epoch"
            ),
            "fang_product_reference_height_km": fang_geometry["reference_height_km"],
            "fang_magnetic_field_height_km": fang_geometry["field_height_km"],
            "fang_magnetic_field_coordinate_mapping": fang_geometry["coordinate_mapping"],
            "fang_magnetic_field_uncertainty_method": "not modelled; magnetic_field_std=0",
            "conductance_field_order": ["Hall", "Pedersen"],
            "directory": destination.name,
            "interpretation": interpretation
        }
        (destination / "sampling.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if not arguments.no_plots:
            plot_wic(
                wic_statistics["mean"][0], wic_quantiles, support, mlat, mlt,
                selection["label"], destination, arguments.wic_vmax
            )
            plot_wic_sample_mosaic(
                mosaic_samples, mosaic_indices, support, mlat, mlt,
                selection["label"], destination, arguments.wic_vmax
            )
            plot_wic_medoid(
                distribution_summary["medoid"],
                distribution_summary["medoid_index"],
                support, mlat, mlt, selection["label"], destination,
                arguments.wic_vmax
            )
            plot_wic_exceedance_probability(
                distribution_summary["exceedance_probability"],
                distribution_summary["exceedance_threshold"],
                support, mlat, mlt, selection["label"], destination
            )
            plot_conductance(
                conductance_statistics["mean"], conductance_quantiles,
                physics["conductance_supported"], mlat, mlt,
                selection["label"], destination, limits
            )
        selections.append(metadata)
        del samples, latent, wic, wic_quantiles, wic_statistics
        del mosaic_samples, mosaic_indices
        del distribution_summary
        del conductance, conductance_quantiles, conductance_statistics, physics

    if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != model_sha:
        raise RuntimeError("source WIC model changed during sampling")
    first_regime = selections[0].get("conditioning_regime") or {}
    controlled = first_regime.get("design") == "independent uniform conditions"
    comparison_policy = (
        "identical latent seed and identical uniform-rank condition draws for "
        "every controlled IMF/SML group; SML and IMF are sampled independently, "
        "dipole tilt and conductance-conversion Kp are fixed, and no empirical "
        "condition-row distribution is used"
        if controlled else
        "identical latent seed for every requested condition or regime; empirical "
        "regimes resample actual training condition rows and each WIC draw is "
        "propagated with its own Kp through collapsed Zhang--Paxton and Fang before "
        "conductance statistics"
    )
    summary = {
        "selections": selections,
        "comparison_policy": comparison_policy,
    }
    (output / "sampling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return selections


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--kp", type=float, nargs="+")
    selection.add_argument(
        "--conditions", type=Path,
        help="JSON list of exact named condition selections"
    )
    selection.add_argument(
        "--regimes", type=Path, nargs=2, metavar=("CACHE", "CONDITIONING"),
        help="sample empirical IMF regimes from the model's cache and conditioning table"
    )
    selection.add_argument(
        "--controlled-regimes", action="store_true",
        help="sample the designed hybrid4 IMF by SML experiment"
    )
    parser.add_argument("--regime", choices=REGIME_NAMES, nargs="+")
    parser.add_argument(
        "--sml-band", choices=tuple(CONTROLLED_SML_BANDS), nargs="+",
        help="controlled SML bands (default: quiet normal disturbed)"
    )
    parser.add_argument("--regime-field-minimum", type=float, default=2.0)
    parser.add_argument("--regime-field-maximum", type=float, default=10.0)
    parser.add_argument("--regime-clock-half-width", type=float, default=30.0)
    parser.add_argument("--regime-tilt-limit", type=float, default=5.0)
    parser.add_argument("--controlled-tilt", type=float, default=0.0)
    parser.add_argument("--conversion-kp", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12104)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--wic-vmax", type=float, default=1000.0)
    parser.add_argument("--wic-exceedance-threshold", type=float, default=100.0)
    parser.add_argument("--hall-vmax", type=float, default=24.0)
    parser.add_argument("--pedersen-vmax", type=float, default=12.0)
    parser.add_argument(
        "--fang-epoch", type=parse_epoch, default=FANG_REFERENCE_EPOCH,
        help=(
            "fixed UTC epoch used to reconstruct geographic coordinates and ApexPy IGRF "
            f"(default: {FANG_REFERENCE_EPOCH.isoformat()}, a round in-period epoch)"
        )
    )
    parser.add_argument(
        "--fang-low-energy-policy", choices=("reject", "clip"), default="clip",
        help="apply Product-3's operational 1-keV floor (default) or strictly reject sub-1-keV energy"
    )
    arguments = parser.parse_args(argv)
    if (
        arguments.regime is not None
        and arguments.regimes is None and not arguments.controlled_regimes
    ):
        parser.error("--regime requires --regimes or --controlled-regimes")
    if arguments.sml_band is not None and not arguments.controlled_regimes:
        parser.error("--sml-band requires --controlled-regimes")
    if (
        not np.isfinite(arguments.controlled_tilt)
        or not np.isfinite(arguments.conversion_kp)
        or not 0 <= arguments.conversion_kp <= 9
        or not np.isfinite(arguments.wic_exceedance_threshold)
        or arguments.wic_exceedance_threshold < 0
    ):
        parser.error(
            "controlled tilt and WIC threshold must be finite and nonnegative; "
            "conversion Kp must be 0--9"
        )
    if arguments.regime is not None and len(set(arguments.regime)) != len(arguments.regime):
        parser.error("requested IMF regimes must be distinct")
    if arguments.sml_band is not None and len(set(arguments.sml_band)) != len(arguments.sml_band):
        parser.error("requested SML bands must be distinct")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
