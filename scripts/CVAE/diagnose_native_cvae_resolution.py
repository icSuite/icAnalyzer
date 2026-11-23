"""Diagnose native-grid reconstruction and hierarchical latent usage."""

#%% Imports

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from icreader import load

from icanalyzer.cvae.artifacts import load_run_artifact
from scripts.CVAE.compare_precipitation_cs_resolution import block_mean, expanded


FIELDS = (
    ("wic_corrected", "Corrected WIC"),
    ("si12", "SI12"),
)
DEFAULT_ORBITS = (96, 368, 595, 884, 1147, 1407, 1674, 1929)


#%% Frame selection and cache-equivalent input preparation

def choose_frame(coarse_path, mask_path, minimum_cells):
    """Choose a bright mask-supported frame without scanning the large file."""

    with load(coarse_path) as coarse, np.load(mask_path, allow_pickle=False) as saved:
        mask = saved["mask"]
        times = saved["time"].astype("datetime64[us]")
        coarse_time = np.asarray(coarse.time, dtype="datetime64[us]")
        if mask.shape[1:] != (184, 184) or not np.array_equal(times, coarse_time):
            raise ValueError(f"{mask_path}: mask and product frames do not align")
        wic = np.asarray(coarse.read("wic_corrected"), dtype=np.float64)
        si12 = np.asarray(coarse.read("si12"), dtype=np.float64)
        count = len(mask)
        coarse_mask = mask.reshape(count, 46, 4, 46, 4).any(axis=(2, 4))
        valid = coarse_mask & np.isfinite(wic) & np.isfinite(si12)
        score = np.nansum(np.where(valid, np.maximum(wic, 0), np.nan), axis=(1, 2))
        enough = valid.sum(axis=(1, 2)) >= minimum_cells
        if not np.any(enough):
            raise ValueError(f"{coarse_path}: no sufficiently supported frame")
        return int(np.nanargmax(np.where(enough, score, np.nan)))


def load_fine_frame(fine_path, mask_path, frame):
    """Read only one native frame and reproduce the training-cache semantics."""

    with np.load(mask_path, allow_pickle=False) as saved:
        selection = np.asarray(saved["mask"][frame], dtype=bool)
        mask_time = saved["time"].astype("datetime64[us]")[frame]

    names = (
        "wic_corrected", "si12", "dwic_corrected", "dsi12",
        "method_valid", "method_quality_weight", "wic_coverage",
        "si12_coverage",
    )
    with load(fine_path) as product:
        product_time = np.asarray(product.time, dtype="datetime64[us]")[frame]
        if product_time != mask_time or tuple(product.shape[1:]) != (184, 184):
            raise ValueError(f"{fine_path}: selected frame does not align")
        arrays = {
            name: np.asarray(product.read(name, frame)) for name in names
        }
        kp = float(np.asarray(product.read("Kp", frame)))

    values = np.stack((
        np.maximum(arrays["wic_corrected"], 0),
        np.maximum(arrays["si12"], 0),
    )).astype(np.float32)
    errors = np.stack((arrays["dwic_corrected"], arrays["dsi12"]))
    coverage = np.stack((arrays["wic_coverage"], arrays["si12_coverage"]))
    common = (
        selection & np.asarray(arrays["method_valid"], dtype=bool)
        & np.isfinite(arrays["method_quality_weight"])
        & (arrays["method_quality_weight"] > 0)
    )
    valid = (
        common[None] & np.isfinite(values) & np.isfinite(errors) & (errors >= 0)
        & np.isfinite(coverage) & (coverage > 0)
    )
    quality = (
        np.clip(arrays["method_quality_weight"], 0, 1)[None]
        * np.clip(coverage, 0, 1)
    )
    quality = np.where(valid, quality, 0).astype(np.float32)
    return {
        "values": values,
        "valid": valid,
        "quality": quality,
        "kp": kp,
        "time": str(mask_time),
    }


def prepare_encoder_input(frame, transform):
    transformed = transform.transform(frame["values"][None])[0].astype(np.float32)
    transformed = np.where(frame["valid"], transformed, 0)
    encoder_input = np.concatenate((transformed, frame["quality"]), axis=0)
    return transformed, encoder_input


#%% Test B: posterior reconstruction and sub-cell structure

def fine_residual(values):
    return values - expanded(block_mean(values))


def sensor_metrics(observed, reconstructed, valid):
    common = valid & np.isfinite(observed) & np.isfinite(reconstructed)
    observed_values = np.where(common, observed, np.nan)
    observed_detail = fine_residual(observed_values)
    reconstructed_detail = fine_residual(reconstructed)
    detail_common = common & np.isfinite(observed_detail)
    scale = np.nanstd(observed_values)
    rmse = np.sqrt(np.mean((observed[common] - reconstructed[common]) ** 2))
    observed_block = block_mean(observed_values)
    reconstructed_block = block_mean(np.where(common, reconstructed, np.nan))
    block_common = np.isfinite(observed_block) & np.isfinite(reconstructed_block)
    block_scale = np.std(observed_block[block_common])
    block_rmse = np.sqrt(np.mean(
        (observed_block[block_common] - reconstructed_block[block_common]) ** 2
    ))
    if detail_common.sum() > 1:
        detail_correlation = np.corrcoef(
            observed_detail[detail_common], reconstructed_detail[detail_common]
        )[0, 1]
        observed_detail_scale = np.std(observed_detail[detail_common])
        detail_ratio = (
            np.std(reconstructed_detail[detail_common]) / observed_detail_scale
            if observed_detail_scale > 0 else np.nan
        )
    else:
        detail_correlation = np.nan
        detail_ratio = np.nan
    return {
        "normalized_rmse": float(rmse / scale) if scale > 0 else np.nan,
        "coarse_normalized_rmse": (
            float(block_rmse / block_scale) if block_scale > 0 else np.nan
        ),
        "detail_correlation": float(detail_correlation),
        "detail_amplitude_ratio": float(detail_ratio),
        "observed_detail": observed_detail,
        "reconstructed_detail": reconstructed_detail,
    }


def symmetric_limit(arrays, percentile=99):
    selected = np.concatenate([
        np.abs(array[np.isfinite(array)]) for array in arrays
    ])
    return max(float(np.percentile(selected, percentile)), 1e-6)


def plot_reconstruction(record, output):
    figure, axes = plt.subplots(2, 5, figsize=(17, 7.2), layout="constrained")
    titles = (
        "Observed 184x184", "Posterior-mean reconstruction", "Reconstruction error",
        "Observed below 46-cell scale", "Reconstruction below 46-cell scale",
    )
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontsize=10)
    for sensor, (_name, label) in enumerate(FIELDS):
        observed = np.where(record["valid"][sensor], record["values"][sensor], np.nan)
        reconstruction = record["reconstruction"][sensor]
        metrics = record["metrics"][sensor]
        error = reconstruction - observed
        maximum = max(float(np.nanpercentile(observed, 99)), 1e-6)
        difference_maximum = symmetric_limit((
            error, metrics["observed_detail"], metrics["reconstructed_detail"]
        ))
        images = []
        for column, values in enumerate((observed, reconstruction)):
            images.append(axes[sensor, column].imshow(
                values, origin="lower", cmap="viridis", vmin=0, vmax=maximum
            ))
        for column, values in enumerate((
            error, metrics["observed_detail"], metrics["reconstructed_detail"]
        ), start=2):
            images.append(axes[sensor, column].imshow(
                values, origin="lower", cmap="coolwarm",
                vmin=-difference_maximum, vmax=difference_maximum
            ))
        axes[sensor, 0].set_ylabel(label)
        for axis in axes[sensor]:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.colorbar(images[1], ax=axes[sensor, :2], shrink=.76, label="counts")
        figure.colorbar(images[2], ax=axes[sensor, 2:], shrink=.76, label="counts")
    figure.suptitle(
        f"Held-out orbit {record['orbit']:04d}, frame {record['frame']}, "
        f"{record['time']} UTC, Kp={record['kp']:g}"
    )
    filename = f"orbit_{record['orbit']:04d}_frame_{record['frame']:03d}.png"
    figure.savefig(output / filename, dpi=170)
    plt.close(figure)
    return filename


def plot_test_b_summary(records, output):
    positions = np.arange(len(records))
    labels = [f"{record['orbit']:04d}" for record in records]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    axes = axes.ravel()
    metrics = (
        ("normalized_rmse", "Native-grid posterior error", "RMSE / observed spatial std"),
        ("coarse_normalized_rmse", "Error after averaging to 46x46", "RMSE / observed spatial std"),
        ("detail_correlation", "Fine-scale pattern retention", "correlation"),
        ("detail_amplitude_ratio", "Fine-scale amplitude retention", "reconstruction / observed"),
    )
    width = .36
    for panel, (name, title, ylabel) in enumerate(metrics):
        for sensor, (_field, label) in enumerate(FIELDS):
            values = [record["metrics"][sensor][name] for record in records]
            axes[panel].bar(
                positions + (sensor - .5) * width, values, width, label=label
            )
        axes[panel].set(
            title=title, ylabel=ylabel, xticks=positions, xticklabels=labels
        )
        axes[panel].tick_params(axis="x", rotation=45)
        axes[panel].legend()
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[3].set_yscale("log")
    axes[0].axhline(1, color="black", linewidth=.8, linestyle="--")
    axes[1].axhline(1, color="black", linewidth=.8, linestyle="--")
    axes[2].axhline(0, color="black", linewidth=.8)
    axes[3].axhline(1, color="black", linewidth=.8, linestyle="--")
    figure.suptitle("Test B: held-out posterior reconstruction at native resolution")
    figure.savefig(output / "test_B_summary.png", dpi=180)
    figure.savefig(output / "test_B_summary.pdf")
    plt.close(figure)


#%% Test C: latent KL and decoder dependence

def kl_per_dimension(mu, logvar):
    return 0.5 * (mu**2 + np.exp(logvar) - 1 - logvar)


def normalized_change(reference, changed, valid):
    common = valid & np.isfinite(reference) & np.isfinite(changed)
    scale = np.std(reference[common])
    change = np.sqrt(np.mean((reference[common] - changed[common]) ** 2))
    return float(change / scale) if scale > 0 else np.nan


def prior_group_sensitivity(model, kp, draws, device, seed, batch_size=32):
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(
        draws, model.latent_dim, generator=generator, device=device
    )
    global_only = latent.clone()
    global_only[:, model.global_latent_dim:] = 0
    spatial_only = latent.clone()
    spatial_only[:, :model.global_latent_dim] = 0
    results = {}
    with torch.no_grad():
        for name, values in (
            ("both", latent), ("global", global_only), ("spatial", spatial_only)
        ):
            batches = []
            for start in range(0, draws, batch_size):
                current = values[start:start + batch_size]
                conditions = torch.full(
                    (len(current),), float(kp), device=device
                )
                batches.append(model.decode(current, kp=conditions).cpu())
            decoded = torch.cat(batches)
            spread = torch.sqrt(torch.mean(
                torch.var(decoded, dim=0, unbiased=False), dim=(1, 2)
            ))
            results[name] = spread.cpu().numpy()
    return results


def plot_test_c(records, mu, logvar, ablations, sensitivities, model, output):
    kl = kl_per_dimension(mu, logvar)
    split = model.global_latent_dim
    global_kl = kl[:, :split]
    spatial_kl = kl[:, split:]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")

    positions = np.arange(len(records))
    labels = [f"{record['orbit']:04d}" for record in records]
    axes[0, 0].bar(positions - .18, global_kl.sum(axis=1), .36, label="global, 36 dims")
    axes[0, 0].bar(positions + .18, spatial_kl.sum(axis=1), .36, label="spatial, 288 dims")
    axes[0, 0].set(
        title="Posterior KL by held-out frame", ylabel="nats / frame",
        xticks=positions, xticklabels=labels,
    )
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].legend()

    axes[0, 1].plot(
        np.sort(global_kl.mean(axis=0))[::-1], label="global"
    )
    axes[0, 1].plot(
        np.sort(spatial_kl.mean(axis=0))[::-1], label="spatial"
    )
    axes[0, 1].axhline(.01, color="black", linestyle="--", linewidth=.8,
                       label="0.01 nat descriptive threshold")
    axes[0, 1].set(
        title="Mean KL per latent dimension", xlabel="dimension, sorted",
        ylabel="nats / frame", yscale="log",
    )
    axes[0, 1].legend()

    width = .18
    ablation_names = ("global_zero", "spatial_zero")
    for sensor, (_field, label) in enumerate(FIELDS):
        for ablation, name in enumerate(ablation_names):
            values = [row[sensor][name] for row in ablations]
            offset = ((sensor * 2 + ablation) - 1.5) * width
            axes[1, 0].bar(
                positions + offset, values, width,
                label=f"{label}, {name.replace('_', ' ')}"
            )
    axes[1, 0].set(
        title="Posterior-mean decoder ablation", ylabel="RMS change / output spatial std",
        xticks=positions, xticklabels=labels,
    )
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].legend(fontsize=8)

    x = np.arange(len(FIELDS))
    for number, name in enumerate(("both", "global", "spatial")):
        axes[1, 1].bar(x + (number - 1) * .24, sensitivities[name], .24, label=name)
    axes[1, 1].set(
        title="Prior decoder sensitivity at median selected Kp",
        ylabel="RMS transformed-output standard deviation",
        xticks=x, xticklabels=[label for _field, label in FIELDS],
    )
    axes[1, 1].legend()
    figure.suptitle("Test C: global and spatial latent usage")
    figure.savefig(output / "test_C_summary.png", dpi=180)
    figure.savefig(output / "test_C_summary.pdf")
    plt.close(figure)

    summary = {
        "mean_total_kl_global": float(global_kl.sum(axis=1).mean()),
        "mean_total_kl_spatial": float(spatial_kl.sum(axis=1).mean()),
        "mean_kl_per_dimension_global": float(global_kl.mean()),
        "mean_kl_per_dimension_spatial": float(spatial_kl.mean()),
        "active_dimensions_above_0.01_nat_global": int((global_kl.mean(axis=0) > .01).sum()),
        "active_dimensions_above_0.01_nat_spatial": int((spatial_kl.mean(axis=0) > .01).sum()),
        "prior_sensitivity_transformed": {
            name: values.tolist() for name, values in sensitivities.items()
        },
    }
    return summary


#%% End-to-end run

def run(arguments):
    device = torch.device(arguments.device)
    model, transform, artifact = load_run_artifact(arguments.artifact, device=device)
    if tuple(model.image_shape) != (184, 184) or not hasattr(model, "spatial_latent_dim"):
        raise ValueError("Tests B/C require the native184 hierarchical model")
    test_orbits = {
        int(Path(filename).stem[3:])
        for filename in artifact["split_files"]["test"]
    }
    if any(orbit not in test_orbits for orbit in arguments.orbits):
        raise ValueError("Tests B/C require orbit-held-out test orbits")

    arguments.output_b.mkdir(parents=True, exist_ok=True)
    arguments.output_c.mkdir(parents=True, exist_ok=True)
    records = []
    mus, logvars, ablations = [], [], []
    for number, orbit in enumerate(arguments.orbits, start=1):
        coarse_path = arguments.coarse / f"or_{orbit:04d}.nc"
        fine_path = arguments.fine / f"or_{orbit:04d}.nc"
        mask_path = arguments.masks / f"or_{orbit:04d}.npz"
        frame_number = choose_frame(coarse_path, mask_path, arguments.minimum_cells)
        frame = load_fine_frame(fine_path, mask_path, frame_number)
        transformed, encoder_input = prepare_encoder_input(frame, transform)
        encoder_tensor = torch.from_numpy(encoder_input[None]).to(device)
        valid_tensor = torch.from_numpy(frame["valid"][None]).to(device)
        kp_tensor = torch.tensor([frame["kp"]], dtype=torch.float32, device=device)
        with torch.no_grad():
            mu, logvar = model.encode(encoder_tensor, valid_tensor, kp=kp_tensor)
            full = model.decode(mu, kp=kp_tensor)
            global_zero = mu.clone()
            global_zero[:, :model.global_latent_dim] = 0
            spatial_zero = mu.clone()
            spatial_zero[:, model.global_latent_dim:] = 0
            decoded_global_zero = model.decode(global_zero, kp=kp_tensor)
            decoded_spatial_zero = model.decode(spatial_zero, kp=kp_tensor)
        reconstruction = transform.inverse(full.cpu().numpy())[0]
        metrics = [
            sensor_metrics(
                frame["values"][sensor], reconstruction[sensor], frame["valid"][sensor]
            ) for sensor in range(len(FIELDS))
        ]
        record = {
            "orbit": orbit, "frame": frame_number, "time": frame["time"],
            "kp": frame["kp"], "values": frame["values"], "valid": frame["valid"],
            "reconstruction": reconstruction, "metrics": metrics,
        }
        record["filename"] = plot_reconstruction(record, arguments.output_b)
        records.append(record)
        mus.append(mu.cpu().numpy()[0])
        logvars.append(logvar.cpu().numpy()[0])
        ablations.append([
            {
                "global_zero": normalized_change(
                    full.cpu().numpy()[0, sensor],
                    decoded_global_zero.cpu().numpy()[0, sensor], frame["valid"][sensor]
                ),
                "spatial_zero": normalized_change(
                    full.cpu().numpy()[0, sensor],
                    decoded_spatial_zero.cpu().numpy()[0, sensor], frame["valid"][sensor]
                ),
            } for sensor in range(len(FIELDS))
        ])
        print(f"{number}/{len(arguments.orbits)} orbit {orbit:04d} complete", flush=True)

    plot_test_b_summary(records, arguments.output_b)
    with (arguments.output_b / "test_B_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "orbit", "frame", "time", "kp", "field", "normalized_rmse",
            "coarse_normalized_rmse", "detail_correlation",
            "detail_amplitude_ratio",
        ))
        for record in records:
            for sensor, (field, _label) in enumerate(FIELDS):
                metrics = record["metrics"][sensor]
                writer.writerow((
                    record["orbit"], record["frame"], record["time"], record["kp"],
                    field, metrics["normalized_rmse"],
                    metrics["coarse_normalized_rmse"],
                    metrics["detail_correlation"], metrics["detail_amplitude_ratio"],
                ))

    mu = np.stack(mus)
    logvar = np.stack(logvars)
    sensitivities = prior_group_sensitivity(
        model, np.median([record["kp"] for record in records]),
        arguments.prior_draws, device, arguments.seed,
    )
    test_c_summary = plot_test_c(
        records, mu, logvar, ablations, sensitivities, model, arguments.output_c
    )
    test_c_summary.update({
        "orbits": list(arguments.orbits),
        "frames": [record["frame"] for record in records],
        "selection": "brightest mask-supported WIC frame in each held-out test orbit",
        "ablation_metrics": ablations,
    })
    (arguments.output_c / "test_C_metrics.json").write_text(
        json.dumps(test_c_summary, indent=2) + "\n"
    )
    return records, test_c_summary


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument("masks", type=Path)
    parser.add_argument("--output-b", type=Path, required=True)
    parser.add_argument("--output-c", type=Path, required=True)
    parser.add_argument("--orbits", type=int, nargs="+", default=DEFAULT_ORBITS)
    parser.add_argument("--minimum-cells", type=int, default=100)
    parser.add_argument("--prior-draws", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args(argv)
    if arguments.prior_draws < 2 or arguments.minimum_cells < 1:
        parser.error("prior draws must exceed one and minimum cells must be positive")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
