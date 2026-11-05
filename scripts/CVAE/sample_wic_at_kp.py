"""Generate WIC CVAE and collapsed-Zhang--Paxton/Fang ensembles at fixed Kp."""

#%% Imports

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
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
from icanalyzer.cvae.ensemble import ensemble_statistics, sample_decoded_ensemble


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


def polar_contour(axis, mlat, mlt, values, maximum, title):
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
    polar_axis.writeLTlabels(fontsize=7)
    polar_axis.writeLATlabels(fontsize=6)
    axis.set_title(title)
    return contour


def plot_wic(mean, quantiles, support, mlat, mlt, kp, output, maximum):
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.4), layout="constrained")
    fields = [mean, *quantiles]
    labels = ("mean", "10th percentile", "median", "90th percentile")
    for axis, values, label in zip(axes, fields, labels):
        values = np.where(support, values, np.nan)
        contour = polar_contour(axis, mlat, mlt, values, maximum, label)
    figure.colorbar(contour, ax=axes, shrink=0.72, label="corrected WIC [counts]")
    figure.suptitle(f"Corrected-WIC model prior at Kp = {kp:g}; common latent seed")
    figure.savefig(output / "wic_at_kp.png", dpi=160)
    figure.savefig(output / "wic_at_kp.pdf")
    plt.close(figure)


def plot_conductance(mean, quantiles, support, mlat, mlt, kp, output, limits):
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
    figure.suptitle(f"WIC → collapsed Zhang–Paxton → Fang at Kp = {kp:g}")
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

    unit_precipitation = precipitation_from_collapsed_zhang_paxton(
        np.ones(mlt.shape), np.zeros(mlt.shape), kp, mlt
    )
    energy = np.asarray(unit_precipitation["E0"], dtype=np.float64)
    energy_spread = np.asarray(unit_precipitation["dE0"], dtype=np.float64)
    flux_per_count = np.asarray(unit_precipitation["Fe"], dtype=np.float64)
    response_flag = np.asarray(unit_precipitation["E0_response_range_flag"], dtype=np.int8)
    reference = fang_2025_conductance(
        energy, np.zeros(energy.shape), magnetic_field_microtesla,
        low_energy_policy=low_energy_policy
    )
    conductance_support = model_support & np.asarray(reference["valid"], dtype=bool)
    if not conductance_support.any():
        raise ValueError("collapsed Zhang--Paxton/Fang has no valid cells on WIC support")

    conductance = np.zeros((len(wic_samples), 2, *mlt.shape), dtype=np.float32)
    for start in range(0, len(wic_samples), batch_size):
        stop = min(start + batch_size, len(wic_samples))
        energy_flux = wic_samples[start:stop] * flux_per_count[None]
        propagated = fang_2025_conductance(
            energy[None], energy_flux, magnetic_field_microtesla,
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
        "conductance_supported": conductance_support
    }
    return conductance, diagnostics


def run(arguments):
    kp_values = np.asarray(arguments.kp, dtype=float)
    if kp_values.ndim != 1 or not len(kp_values) or not np.isfinite(kp_values).all() or np.any((kp_values < 0) | (kp_values > 9)):
        raise ValueError("requested Kp values must be finite and lie in 0--9")
    if len(np.unique(kp_values)) != len(kp_values):
        raise ValueError("requested Kp values must be distinct")
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
    if model.kp_conditioning is None or transform.channel_count != 1:
        raise ValueError("requires a Kp-conditioned one-channel transform")

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
    for index, kp in enumerate(kp_values):
        kp = float(kp)
        within_range = model.kp_conditioning["training_min"] <= kp <= model.kp_conditioning["training_max"]
        if not within_range:
            warnings.warn(f"Kp {kp:g} is outside the WIC-CVAE training range", RuntimeWarning, stacklevel=2)
        destination = output / f"kp_{index:02d}_{kp:g}"
        destination.mkdir()
        print(f"Kp {kp:g}: generating {arguments.draws} corrected-WIC fields", flush=True)
        samples, latent = sample_decoded_ensemble(
            model, transform, arguments.draws, batch_size=arguments.batch_size,
            device=device, seed=arguments.seed, return_latent=True,
            support_mask=support, kp=kp
        )
        wic = samples[:, 0]
        if not np.isfinite(wic[:, support]).all() or np.any(wic[:, support] < 0):
            raise ValueError("decoder produced invalid corrected WIC on learned support")
        wic_quantiles = np.quantile(wic, [0.1, 0.5, 0.9], axis=0)
        wic_quantiles[:, ~support] = np.nan
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
            wic_supported=support, wic_sample_kp=np.full(arguments.draws, kp),
            wic_q10=wic_quantiles[0], wic_q50=wic_quantiles[1], wic_q90=wic_quantiles[2],
            conductance_mean=conductance_statistics["mean"],
            conductance_anomaly_factor=conductance_statistics["anomaly_factor"],
            conductance_supported=physics["conductance_supported"],
            conductance_sample_kp=np.full(arguments.draws, kp),
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
            fang_modified_apex_longitude=fang_geometry["modified_apex_longitude"],
            fang_geographic_latitude_at_130km=fang_geometry["geographic_latitude"],
            fang_geographic_longitude_at_130km=fang_geometry["geographic_longitude"],
            fang_apex2geo_error_degrees=fang_geometry["apex2geo_error_degrees"],
            fang_magnetic_field_150km_microtesla=fang_geometry["magnetic_field_microtesla"],
            mlat=mlat, mlt=mlt
        )
        metadata = {
            "Kp": kp, "within_training_range": bool(within_range), "draws": arguments.draws,
            "seed": arguments.seed, "device": device, "model_sha256": model_sha,
            "model_file": str(artifact_path), "kp_conditioning": dict(model.kp_conditioning),
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
            "interpretation": "Fixed-Kp WIC prior propagated member-by-member to Fang electron conductance. Its fixed reference epoch approximates frame-specific Product-3 IGRF because this climatological artifact has no observation time. No proton/solar-EUV contribution, WIC measurement-noise propagation, sampled Zhang--Paxton structural spread or calibrated uncertainty claim."
        }
        (destination / "sampling.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if not arguments.no_plots:
            plot_wic(
                wic_statistics["mean"][0], wic_quantiles, support, mlat, mlt,
                kp, destination, arguments.wic_vmax
            )
            plot_conductance(
                conductance_statistics["mean"], conductance_quantiles,
                physics["conductance_supported"], mlat, mlt,
                kp, destination, limits
            )
        selections.append(metadata)
        del samples, latent, wic, wic_quantiles, wic_statistics
        del conductance, conductance_quantiles, conductance_statistics, physics

    if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != model_sha:
        raise RuntimeError("source WIC model changed during sampling")
    summary = {
        "selections": selections,
        "comparison_policy": "identical latent seed at every Kp; each WIC draw is propagated through collapsed Zhang--Paxton and Fang before conductance statistics"
    }
    (output / "sampling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return selections


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--kp", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12104)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--wic-vmax", type=float, default=1000.0)
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
