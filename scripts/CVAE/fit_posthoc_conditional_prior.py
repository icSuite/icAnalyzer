"""Fit and sample post-hoc Kp-conditioned priors for a trained CVAE."""

#%% Imports

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from icanalyzer.cvae.artifacts import load_run_artifact
from icanalyzer.cvae.prior import (
    ConditionalGaussianMixturePrior,
    aggregate_posterior_standardization,
    disable_optional_native_bmm_override,
)
from scripts.CVAE.analyze_empirical_latent_modes import (
    FIELD_NAMES,
    load_bank,
    orbit_heldout_split,
)
from scripts.CVAE.sample_wic_at_kp import (
    plot_wic,
    plot_wic_sample_mosaic,
)


#%% Posterior batches and density fitting

def candidate_label(components, covariance_rank):
    if covariance_rank:
        return f"K={components}, rank={covariance_rank}"
    return f"K={components}"


def candidate_directory(components, covariance_rank):
    suffix = f"_rank{covariance_rank}" if covariance_rank else ""
    return f"prior_k{components}{suffix}"


def latent_dataset(bank, rows):
    rows = np.asarray(rows, dtype=np.int64)
    return TensorDataset(
        torch.from_numpy(bank["mu"][rows].astype(np.float32)),
        torch.from_numpy(bank["logvar"][rows].astype(np.float32)),
        torch.from_numpy(bank["kp"][rows].astype(np.float32)),
    )


def sample_standardized_posterior(
    mu, logvar, center, scale, generator=None,
):
    noise = torch.randn(
        mu.shape, dtype=mu.dtype, device=mu.device, generator=generator
    )
    latent = mu + noise * torch.exp(0.5 * logvar)
    return (latent - center) / scale


def evaluate_density(
    prior, loader, center, scale, posterior_samples, seed, device,
):
    prior.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    frame_log_prob = []
    with torch.no_grad():
        for mu, logvar, kp in loader:
            mu = mu.to(device)
            logvar = logvar.to(device)
            kp = kp.to(device)
            values = []
            for _ in range(posterior_samples):
                latent = sample_standardized_posterior(
                    mu, logvar, center, scale, generator
                )
                values.append(prior.log_prob(latent, kp))
            frame_log_prob.append(torch.stack(values).mean(0).cpu().numpy())
    values = np.concatenate(frame_log_prob).astype(np.float64)
    return {
        "mean_log_density": float(np.mean(values)),
        "standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        "frames": int(len(values)),
    }


def fit_candidate(
    bank, train_rows, validation_rows, center, scale, arguments,
    components, covariance_rank, kp_center, kp_scale, device,
):
    torch.manual_seed(arguments.seed + components + 100 * covariance_rank)
    prior = ConditionalGaussianMixturePrior(
        latent_dim=bank["mu"].shape[1], components=components,
        hidden_dim=arguments.hidden_dim, kp_center=kp_center,
        kp_scale=kp_scale, minimum_logvar=arguments.minimum_logvar,
        maximum_logvar=arguments.maximum_logvar,
        covariance_rank=covariance_rank,
        maximum_factor=arguments.maximum_factor,
    ).to(device)
    train_loader = DataLoader(
        latent_dataset(bank, train_rows), batch_size=arguments.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(arguments.seed),
    )
    validation_loader = DataLoader(
        latent_dataset(bank, validation_rows), batch_size=arguments.batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        prior.parameters(), lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(2, arguments.patience // 3),
        min_lr=1e-6,
    )
    center_tensor = torch.from_numpy(center).to(device)
    scale_tensor = torch.from_numpy(scale).to(device)
    generator = torch.Generator(device=device).manual_seed(
        arguments.seed + 1000 + components + 100 * covariance_rank
    )
    best_loss = np.inf
    best_state = None
    wait = 0
    history = []
    progress = tqdm(
        range(1, arguments.epochs + 1),
        desc=f"Conditional prior {candidate_label(components, covariance_rank)}",
        unit="epoch",
    )
    for epoch in progress:
        prior.train()
        total_loss = 0.0
        total_frames = 0
        for mu, logvar, kp in train_loader:
            mu = mu.to(device)
            logvar = logvar.to(device)
            kp = kp.to(device)
            latent = sample_standardized_posterior(
                mu, logvar, center_tensor, scale_tensor, generator
            )
            loss = -prior.log_prob(latent, kp).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 10)
            optimizer.step()
            total_loss += float(loss.detach()) * len(mu)
            total_frames += len(mu)
        validation = evaluate_density(
            prior, validation_loader, center_tensor, scale_tensor,
            arguments.validation_samples, arguments.seed + 40_000, device,
        )
        train_loss = total_loss / total_frames
        validation_loss = -validation["mean_log_density"]
        scheduler.step(validation_loss)
        history.append({
            "epoch": epoch,
            "train_negative_log_density": train_loss,
            "validation_negative_log_density": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        progress.set_postfix(
            train=f"{train_loss:.2f}", validation=f"{validation_loss:.2f}"
        )
        if validation_loss < best_loss - arguments.minimum_improvement:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in prior.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= arguments.patience:
                break
    if best_state is None:
        raise RuntimeError("conditional-prior training did not produce a checkpoint")
    prior.load_state_dict(best_state)
    final_validation = evaluate_density(
        prior, validation_loader, center_tensor, scale_tensor,
        arguments.validation_samples, arguments.seed + 50_000, device,
    )
    return prior, history, final_validation


#%% Diagnostics and decoded galleries

def plot_history(history, output, label):
    epochs = [record["epoch"] for record in history]
    train = [record["train_negative_log_density"] for record in history]
    validation = [
        record["validation_negative_log_density"] for record in history
    ]
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    axis.plot(epochs, train, label="training")
    axis.plot(epochs, validation, label="orbit-held-out")
    axis.set(
        title=f"Conditional latent prior: {label}",
        xlabel="epoch", ylabel="negative log density",
    )
    axis.legend()
    figure.savefig(output / "training_history.png", dpi=180)
    figure.savefig(output / "training_history.pdf")
    plt.close(figure)


def plot_component_weights(prior, kp_minimum, kp_maximum, output, device):
    kp = torch.linspace(kp_minimum, kp_maximum, 241, device=device)
    with torch.no_grad():
        logits, _means, _logvars = prior.parameters_at(kp)
        weights = torch.softmax(logits, dim=-1).cpu().numpy()
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for component in range(prior.components):
        axis.plot(kp.cpu(), weights[:, component], label=f"component {component + 1}")
    axis.set(
        title="Learned conditional mixture weights",
        xlabel="stored Kp category", ylabel="mixture weight", ylim=(0, 1),
    )
    if prior.components <= 8:
        axis.legend(ncol=2)
    figure.savefig(output / "component_weights.png", dpi=180)
    figure.savefig(output / "component_weights.pdf")
    plt.close(figure)


def decode_prior(
    prior, model, transform, center, scale, target_kp, draws,
    batch_size, seed, device, support,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    condition = torch.full(
        (draws,), float(target_kp), dtype=torch.float32, device=device
    )
    prior.eval()
    model.eval()
    with torch.no_grad():
        standardized, components = prior.sample(condition, generator)
        center_tensor = torch.from_numpy(center).to(device)
        scale_tensor = torch.from_numpy(scale).to(device)
        latent = center_tensor + standardized * scale_tensor
        fields = np.empty(
            (draws, len(FIELD_NAMES), *model.image_shape), dtype=np.float32
        )
        for start in range(0, draws, batch_size):
            stop = min(start + batch_size, draws)
            decoded = model.decode(
                latent[start:stop], kp=condition[start:stop]
            )
            fields[start:stop] = transform.inverse(decoded.cpu().numpy())
    fields = np.where(support[None], fields, 0)
    return latent.cpu().numpy(), components.cpu().numpy(), fields


def save_prior_gallery(
    fields, latent, component, support, mlat, mlt, target_kp,
    candidate_label, output, wic_maximum, si12_maximum,
):
    output.mkdir()
    saved = {
        "latent": latent,
        "component": component.astype(np.int16),
        "target_kp": np.asarray(target_kp, dtype=np.float32),
    }
    for channel, (field, label, maximum) in enumerate((
        ("wic_corrected", "Corrected WIC", wic_maximum),
        ("si12", "SI12", si12_maximum),
    )):
        values = fields[:, channel]
        mean = np.mean(values, axis=0)
        quantiles = np.quantile(values, (0.1, 0.5, 0.9), axis=0)
        mosaic = values[:min(36, len(values))]
        title = f"{candidate_label}, sampled at Kp {target_kp:g}"
        plot_wic(
            mean, quantiles, support[channel], mlat, mlt, title, output,
            maximum, field_label=label, file_prefix=field,
            distribution_label="learned conditional prior",
        )
        plot_wic_sample_mosaic(
            mosaic, np.arange(len(mosaic)), support[channel], mlat, mlt,
            title, output, maximum, field_label=label, file_prefix=field,
        )
        saved.update({
            f"{field}_mean": mean,
            f"{field}_q10": quantiles[0],
            f"{field}_q50": quantiles[1],
            f"{field}_q90": quantiles[2],
            f"{field}_mosaic": mosaic,
        })
    np.savez_compressed(output / "prior_samples.npz", **saved)


def plot_candidate_comparison(records, output):
    positions = np.arange(len(records))
    labels = [record["label"] for record in records]
    density = [record["validation"]["mean_log_density"] for record in records]
    error = [record["validation"]["standard_error"] for record in records]
    figure, axis = plt.subplots(figsize=(max(6, 1.8 * len(records)), 4.5), layout="constrained")
    axis.errorbar(positions, density, yerr=error, marker="o", linestyle="none")
    axis.set(
        title="Orbit-held-out conditional prior density",
        xlabel="prior candidate", ylabel="mean log density",
    )
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    figure.savefig(output / "candidate_comparison.png", dpi=180)
    figure.savefig(output / "candidate_comparison.pdf")
    plt.close(figure)


#%% End-to-end run and CLI

def run(arguments):
    bank_path = arguments.bank.expanduser().resolve()
    run_directory = arguments.run_directory.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("choose an empty conditional-prior output directory")
    output.mkdir(parents=True, exist_ok=True)
    bank = load_bank(bank_path)
    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if arguments.device == "auto" else arguments.device
    device = torch.device(device_name)
    if disable_optional_native_bmm_override(
        max(arguments.covariance_ranks), device
    ):
        print(
            "Disabled PyTorch's optional Triton BMM override for the low-rank "
            "fit; using the standard ATen CUDA implementation.",
            flush=True,
        )
    artifact_path = run_directory / "wic_cvae.pt"
    model, transform, artifact = load_run_artifact(artifact_path, device=device)
    if (
        tuple(model.field_names) != FIELD_NAMES
        or model.latent_dim != bank["mu"].shape[1]
        or model.kp_conditioning is None
        or model.conditioning is not None
    ):
        raise ValueError("latent bank and paired WIC/SI12 artifact do not align")
    sampling_summary = bank_path.parent / "sampling_summary.json"
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if sampling_summary.is_file():
        expected = json.loads(sampling_summary.read_text()).get("artifact_sha256")
        if expected != artifact_sha256:
            raise ValueError("latent bank and run artifact hashes do not match")

    train_rows, validation_rows = orbit_heldout_split(
        bank["orbit"], arguments.validation_fraction, arguments.seed
    )
    center, scale = aggregate_posterior_standardization(
        bank["mu"], bank["logvar"], train_rows
    )
    kp_configuration = model.kp_conditioning
    kp_center = float(kp_configuration["center"])
    kp_scale = float(kp_configuration["scale"])
    support = np.asarray(
        artifact["data_provenance"]["training_support"], dtype=bool
    )
    grid = artifact["data_provenance"]["grid"]
    mlat = np.asarray(grid["mlat"], dtype=float)
    mlt = np.asarray(grid["mlt"], dtype=float)

    records = []
    payloads = {}
    bank_sha256 = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    for components in arguments.components:
        for covariance_rank in arguments.covariance_ranks:
            label = candidate_label(components, covariance_rank)
            candidate_output = output / candidate_directory(
                components, covariance_rank
            )
            candidate_output.mkdir()
            prior, history, validation = fit_candidate(
                bank, train_rows, validation_rows, center, scale, arguments,
                components, covariance_rank, kp_center, kp_scale, device,
            )
            plot_history(history, candidate_output, label)
            plot_component_weights(
                prior, float(np.min(bank["kp"])), float(np.max(bank["kp"])),
                candidate_output, device,
            )
            payload = {
                "format": "posthoc_conditional_latent_prior_v2",
                "prior_configuration": prior.configuration(),
                "prior_state_dict": {
                    name: value.detach().cpu()
                    for name, value in prior.state_dict().items()
                },
                "latent_center": center,
                "latent_scale": scale,
                "artifact_sha256": artifact_sha256,
                "bank_sha256": bank_sha256,
                "training_rows": train_rows,
                "validation_rows": validation_rows,
                "validation": validation,
            }
            torch.save(payload, candidate_output / "conditional_latent_prior.pt")
            key = (components, covariance_rank)
            payloads[key] = payload
            record = {
                "label": label,
                "components": components,
                "covariance_rank": covariance_rank,
                "epochs": len(history),
                "validation": validation,
                "history": history,
                "directory": candidate_output.name,
            }
            records.append(record)
            for target_kp in arguments.kp:
                latent, selected_component, fields = decode_prior(
                    prior, model, transform, center, scale, target_kp,
                    arguments.draws, arguments.decode_batch_size,
                    arguments.seed + int(round(target_kp * 1000))
                    + components + 100 * covariance_rank,
                    device, support,
                )
                save_prior_gallery(
                    fields, latent, selected_component, support, mlat, mlt,
                    target_kp, f"conditional prior {label}",
                    candidate_output / f"kp_{target_kp:g}",
                    arguments.wic_vmax, arguments.si12_vmax,
                )
                del fields

    selected = max(
        records, key=lambda record: record["validation"]["mean_log_density"]
    )
    torch.save(
        payloads[(selected["components"], selected["covariance_rank"])],
        output / "conditional_latent_prior.pt",
    )
    plot_candidate_comparison(records, output)
    summary = {
        "latent_bank": str(bank_path),
        "run_directory": str(run_directory),
        "device": str(device),
        "frames": int(len(bank["kp"])),
        "orbits": int(len(np.unique(bank["orbit"]))),
        "train_frames": int(len(train_rows)),
        "validation_frames": int(len(validation_rows)),
        "selection_policy": "highest orbit-held-out mean posterior-sample log density",
        "selected_components": selected["components"],
        "selected_candidate": {
            "label": selected["label"],
            "components": selected["components"],
            "covariance_rank": selected["covariance_rank"],
        },
        "candidates": records,
        "qualification": (
            "Mixture components are density basis functions and must not be "
            "interpreted as physical auroral modes."
        ),
    }
    (output / "posthoc_prior_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"Selected {selected['label']} by orbit-held-out density", flush=True
    )
    return summary


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, nargs="+", default=(1, 6))
    parser.add_argument("--covariance-ranks", type=int, nargs="+", default=(0,))
    parser.add_argument("--kp", type=float, nargs="+", default=(0, 2, 4, 6))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--minimum-improvement", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--minimum-logvar", type=float, default=-8)
    parser.add_argument("--maximum-logvar", type=float, default=4)
    parser.add_argument("--maximum-factor", type=float, default=2)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12104)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wic-vmax", type=float, default=1000)
    parser.add_argument("--si12-vmax", type=float, default=50)
    arguments = parser.parse_args(argv)
    if (
        any(value < 1 for value in arguments.components)
        or len(set(arguments.components)) != len(arguments.components)
        or any(value < 0 for value in arguments.covariance_ranks)
        or len(set(arguments.covariance_ranks)) != len(arguments.covariance_ranks)
        or arguments.hidden_dim < 1 or arguments.epochs < 1
        or arguments.patience < 1 or arguments.batch_size < 1
        or arguments.validation_samples < 1 or arguments.draws < 1
        or arguments.decode_batch_size < 1
        or not 0 < arguments.validation_fraction < 1
        or arguments.learning_rate <= 0 or arguments.weight_decay < 0
        or arguments.minimum_improvement < 0
        or not arguments.minimum_logvar < arguments.maximum_logvar
        or not np.isfinite(arguments.maximum_factor)
        or arguments.maximum_factor <= 0
        or not np.isfinite(arguments.kp).all()
        or any(not 0 <= value <= 9 for value in arguments.kp)
        or len(set(arguments.kp)) != len(arguments.kp)
        or arguments.wic_vmax <= 0 or arguments.si12_vmax <= 0
    ):
        parser.error("invalid conditional-prior training or sampling setting")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
