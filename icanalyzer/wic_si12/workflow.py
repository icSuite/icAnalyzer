"""Training and artifact helpers for the corrected-WIC/SI12 VAE."""

#%% Imports

from pathlib import Path
import random

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import ChannelTransform
from .count_likelihood import COUNT_UNCERTAINTY_LOSS_KIND, validate_count_configuration
from .model import SplineWICSI12VAE, WICSI12VAE, vae_loss
from .coverage_weighting import validate_coverage_weighting


LEGACY_LOSS_CONFIGURATION = {
    "kind": "legacy_weighted_mse_v1",
    "reconstruction_reduction": "global_weight_sum_times_state_size",
    "kl_reduction": "sum_batch_and_latent",
    "measurement_uncertainty_in_loss": False,
}

MEASUREMENT_UNCERTAINTY_LOSS_KIND = "training_median_precision_v1"
GAUSSIAN_UNCERTAINTY_LOSS_KIND = "quality_weighted_gaussian_v1"
WIC_CVAE_ARTIFACT_TYPE = "wic_cvae_run_v1"


def gaussian_uncertainty_configuration(model_error_std=(0.5, 0.5)):
    """Fixed model discrepancy in normalized log10 units, one value per channel."""

    return _validated_loss_configuration({
        "kind": GAUSSIAN_UNCERTAINTY_LOSS_KIND,
        "measurement_uncertainty_in_loss": True,
        "model_error_std": list(model_error_std),
        "model_error_units": "standardized_log10(value + 1)",
        "variance_model": "measurement_variance_plus_fixed_model_variance",
        "likelihood_scale": "twice_model_variance_per_channel",
        "reconstruction_reduction": "global_quality_sum_times_state_size",
        "kl_reduction": "sum_batch_and_latent"
    })


#%% Reproducibility

def set_random_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#%% Training

def add_training_holes(encoder_input, mask, probability, hole_size, generator):
    """Remove one square from selected inputs, retaining the original targets."""

    if probability == 0:
        return encoder_input, mask
    height, width = encoder_input.shape[-2:]
    if not 1 <= hole_size < min(height, width):
        raise ValueError("hole_size must be positive and smaller than the image dimensions")
    damaged_input = encoder_input.clone()
    input_mask = mask.clone()
    for index in range(encoder_input.shape[0]):
        if torch.rand((), generator=generator).item() >= probability:
            continue
        candidates = torch.nonzero(mask[index, 0], as_tuple=False)
        if candidates.shape[0] == 0:
            continue
        choice = torch.randint(candidates.shape[0], (), generator=generator).item()
        row, column = candidates[choice].tolist()
        top = min(max(row - hole_size // 2, 0), height - hole_size)
        left = min(max(column - hole_size // 2, 0), width - hole_size)
        remaining = mask[index].clone()
        remaining[:, top:top + hole_size, left:left + hole_size] = False
        if not torch.any(remaining):
            continue
        input_mask[index] = remaining
        damaged_input[index, :, top:top + hole_size, left:left + hole_size] = 0.0
    return damaged_input, input_mask


def _validated_loss_configuration(loss_configuration):
    if loss_configuration is None:
        raise ValueError(
            "loss_configuration is required so loss semantics are explicit"
        )
    kind = loss_configuration.get("kind")
    if kind not in ("legacy_weighted_mse_v1", MEASUREMENT_UNCERTAINTY_LOSS_KIND, GAUSSIAN_UNCERTAINTY_LOSS_KIND, COUNT_UNCERTAINTY_LOSS_KIND):
        raise ValueError(
            f"unsupported loss configuration {kind!r}"
        )
    if loss_configuration.get("coverage_weighting") is not None:
        validate_coverage_weighting(loss_configuration["coverage_weighting"])
    if kind == COUNT_UNCERTAINTY_LOSS_KIND:
        return validate_count_configuration(loss_configuration)
    if kind == GAUSSIAN_UNCERTAINTY_LOSS_KIND:
        error_std = np.asarray(loss_configuration.get("model_error_std"), dtype=float)
        if error_std.shape != (2,) or np.any(~np.isfinite(error_std)) or np.any(error_std <= 0):
            raise ValueError("loss model_error_std must contain two finite positive values")
    if kind == MEASUREMENT_UNCERTAINTY_LOSS_KIND:
        required = (
            "precision_scale", "variance_floor", "maximum_relative_precision"
        )
        missing = [name for name in required if name not in loss_configuration]
        if missing:
            raise ValueError("loss configuration is missing: " + ", ".join(missing))
        precision_scale = np.asarray(
            loss_configuration["precision_scale"], dtype=float
        )
        if (
            precision_scale.shape != (2,)
            or np.any(~np.isfinite(precision_scale))
            or np.any(precision_scale <= 0)
        ):
            raise ValueError(
                "loss precision_scale must contain two finite positive values"
            )
        for name in ("variance_floor", "maximum_relative_precision"):
            value = float(loss_configuration[name])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"loss {name} must be finite and positive")
    return dict(loss_configuration)


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    beta,
    training,
    loss_configuration=None,
    hole_probability=0.0,
    hole_size=12,
    hole_generator=None,
    progress_description=None,
):
    device = torch.device(device)
    loss_configuration = _validated_loss_configuration(loss_configuration)
    if loss_configuration["kind"] == COUNT_UNCERTAINTY_LOSS_KIND:
        if loss_configuration["transform"] != loader.dataset.transform.to_dict():
            raise ValueError("count likelihood transform does not match the dataset transform")
    model.train(training)
    metric_names = ("total", "reconstruction", "kl")
    # Match Python's float64 accumulation without synchronizing every GPU batch.
    totals = torch.zeros(len(metric_names), dtype=torch.float64, device=device)
    non_blocking = device.type == "cuda" and getattr(loader, "pin_memory", False)
    sample_count = 0
    coverage = loss_configuration.get("coverage_weighting")
    if coverage is not None:
        coverage_maps = torch.as_tensor(coverage["factors"], dtype=torch.float32, device=device)
        coverage_edges = torch.as_tensor(coverage["kp_bin_edges"][1:-1], dtype=torch.float64, device=device)

    batches = loader
    if progress_description is not None:
        batches = tqdm(loader, desc=progress_description, unit="batch", leave=False, position=1, dynamic_ncols=True)
    for batch in batches:
        encoder_input = batch["encoder_input"]
        input_mask = batch["mask"]
        if training and hole_probability > 0:
            encoder_input, input_mask = add_training_holes(
                encoder_input, input_mask, hole_probability, hole_size, hole_generator
            )
        encoder_input = encoder_input.to(device, non_blocking=non_blocking)
        input_mask = input_mask.to(device, non_blocking=non_blocking)
        target = batch["target"].to(device, non_blocking=non_blocking)
        mask = batch["mask"].to(device, non_blocking=non_blocking)
        weight = batch["weight"].to(device, non_blocking=non_blocking)
        if coverage is not None:
            if coverage_maps.shape[1:] != target.shape[-2:]:
                raise ValueError("coverage weights and reconstruction target use different grids")
            if "kp" not in batch:
                raise ValueError("coverage weighting requires per-frame Kp")
            coverage_kp = batch["kp"]
            if coverage_kp.shape != (len(target),):
                raise ValueError("coverage weighting requires one scalar Kp per frame")
            if not torch.all(torch.isfinite(coverage_kp) & (coverage_kp >= 0) & (coverage_kp <= 9)):
                raise ValueError("coverage weighting requires finite Kp within 0–9")
            coverage_kp = coverage_kp.to(device, dtype=torch.float64, non_blocking=non_blocking)
            bin_ids = torch.bucketize(coverage_kp, coverage_edges, right=True)
            # Only the loss weights change. The encoder retains original quality,
            # and the original target mask still excludes every missing pixel.
            weight = weight * coverage_maps[bin_ids, None]
        uncertainty_arguments = {}
        if loss_configuration["kind"] == COUNT_UNCERTAINTY_LOSS_KIND:
            uncertainty_arguments = {
                "count_observed": batch["physical_target"].to(device, non_blocking=non_blocking),
                "count_standard_deviation": batch["physical_standard_deviation"].to(device, non_blocking=non_blocking),
                "count_configuration": loss_configuration
            }
        elif loss_configuration["kind"] == GAUSSIAN_UNCERTAINTY_LOSS_KIND:
            uncertainty_arguments = {
                "transformed_variance": batch["variance"].to(device, non_blocking=non_blocking),
                "model_error_std": loss_configuration["model_error_std"]
            }
        elif loss_configuration["kind"] == MEASUREMENT_UNCERTAINTY_LOSS_KIND:
            uncertainty_arguments = {
                "transformed_variance": batch["variance"].to(device, non_blocking=non_blocking),
                "precision_scale": loss_configuration["precision_scale"],
                "variance_floor": loss_configuration["variance_floor"],
                "maximum_relative_precision": loss_configuration[
                    "maximum_relative_precision"
                ],
            }

        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            condition_arguments = {}
            if getattr(model, "kp_conditioning", None) is not None:
                if "kp" not in batch:
                    raise ValueError("Kp-conditioned training/evaluation requires per-image Kp labels")
                condition_arguments["kp"] = batch["kp"].to(device, non_blocking=non_blocking)
            reconstructed, mu, logvar = model(encoder_input, input_mask, **condition_arguments)
            losses = vae_loss(
                reconstructed,
                target,
                mask,
                weight,
                mu,
                logvar,
                beta=beta,
                **uncertainty_arguments,
            )
            if training:
                losses["total"].backward()
                optimizer.step()

        batch_size = encoder_input.shape[0]
        sample_count += batch_size
        totals += torch.stack([losses[name].detach().to(dtype=torch.float64) for name in metric_names])

    if sample_count == 0:
        raise ValueError("data loader contains no samples")
    return {name: value / sample_count for name, value in zip(metric_names, totals.cpu().tolist())}


def train_vae(
    model,
    train_loader,
    validation_loader=None,
    epochs=20,
    learning_rate=1e-3,
    beta=1e-3,
    device="cpu",
    seed=0,
    loss_configuration=None,
    scheduler_factor=0.5,
    scheduler_patience=10,
    scheduler_minimum_learning_rate=0.0,
    hole_probability=0.0,
    hole_size=12,
):
    """Train with separate, consistently reduced reconstruction and KL logs."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not np.isfinite(hole_probability) or not 0 <= hole_probability <= 1:
        raise ValueError("hole_probability must be finite and between zero and one")
    if not isinstance(hole_size, (int, np.integer)) or hole_size < 1:
        raise ValueError("hole_size must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(scheduler_factor) or not 0 < scheduler_factor < 1:
        raise ValueError("scheduler_factor must be finite and between zero and one")
    if scheduler_patience < 0:
        raise ValueError("scheduler_patience must be nonnegative")
    if (
        not np.isfinite(scheduler_minimum_learning_rate)
        or scheduler_minimum_learning_rate < 0
    ):
        raise ValueError(
            "scheduler_minimum_learning_rate must be finite and nonnegative"
        )
    loss_configuration = _validated_loss_configuration(loss_configuration)
    set_random_seed(seed)
    hole_generator = torch.Generator().manual_seed(int(seed) + 3) if hole_probability > 0 else None
    hole_arguments = {
        "hole_probability": hole_probability,
        "hole_size": hole_size,
        "hole_generator": hole_generator
    } if hole_probability > 0 else {}
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_factor),
        patience=int(scheduler_patience),
        min_lr=float(scheduler_minimum_learning_rate),
    )
    history = []
    best_validation_total = np.inf
    best_validation_epoch = None
    best_state = None

    print(f"Training VAE on {device}: {epochs} epochs, {len(train_loader)} training batches per epoch.", flush=True)
    with tqdm(total=epochs, desc="VAE epochs", unit="epoch", position=0, dynamic_ncols=True) as progress:
        for epoch in range(1, epochs + 1):
            train_metrics = _run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                beta,
                training=True,
                loss_configuration=loss_configuration,
                progress_description=f"Epoch {epoch}/{epochs} train",
                **hole_arguments,
            )
            record = {"epoch": epoch, "train": train_metrics}
            if validation_loader is not None:
                record["validation"] = _run_epoch(
                    model,
                    validation_loader,
                    optimizer=None,
                    device=device,
                    beta=beta,
                    training=False,
                    loss_configuration=loss_configuration,
                    progress_description=f"Epoch {epoch}/{epochs} validation",
                )
                validation_total = record["validation"]["total"]
                if validation_total < best_validation_total:
                    best_validation_total = validation_total
                    best_validation_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            scheduler.step(record["train"]["total"])
            record["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            history.append(record)

            # Reporting only: preserve loss, scheduler and checkpoint behavior.
            status = {"train_loss": f"{train_metrics['total']:.3f}", "lr": f"{record['learning_rate']:.2e}"}
            if validation_loader is not None:
                status["val_loss"] = f"{record['validation']['total']:.3f}"
            progress.set_postfix(status, refresh=False)
            progress.update(1)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_validation_epoch = best_validation_epoch
    model.best_validation_total = (
        None if best_validation_epoch is None else float(best_validation_total)
    )
    model.loss_configuration = loss_configuration
    return model, history


def evaluate_vae(
    model,
    loader,
    beta=1e-3,
    device="cpu",
    loss_configuration=None,
):
    """Evaluate the same reduced loss without changing model parameters."""

    if loss_configuration is None:
        loss_configuration = getattr(model, "loss_configuration", None)

    return _run_epoch(
        model,
        loader,
        optimizer=None,
        device=torch.device(device),
        beta=beta,
        training=False,
        loss_configuration=loss_configuration,
    )


#%% Saved run artifact

def save_run_artifact(
    filename,
    model,
    transform,
    history,
    training_configuration,
    data_provenance,
    split_files,
    evaluation=None,
    artifact_type="wic_si12_vae_run_v1",
):
    """Save everything required to reload the model and identify its inputs."""

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_type": artifact_type,
        "model_configuration": model.configuration(),
        "model_state": model.state_dict(),
        "transform": transform.to_dict(),
        "history": history,
        "training_configuration": dict(training_configuration),
        "data_provenance": data_provenance,
        "split_files": {
            name: [str(path) for path in paths]
            for name, paths in split_files.items()
        },
        "evaluation": {} if evaluation is None else dict(evaluation),
    }
    torch.save(artifact, filename)


def load_run_artifact(filename, device="cpu"):
    artifact = torch.load(filename, map_location=device, weights_only=False)
    if artifact.get("artifact_type") not in ("wic_si12_vae_run_v1", "conductance_vae_run_v1", WIC_CVAE_ARTIFACT_TYPE):
        raise ValueError("unsupported run artifact")
    model_configuration = artifact["model_configuration"]
    if model_configuration.get("kp_conditioning") is not None and artifact["artifact_type"] not in ("conductance_vae_run_v1", WIC_CVAE_ARTIFACT_TYPE):
        raise ValueError("Kp conditioning requires a native conductance or WIC CVAE artifact")
    if model_configuration.get("model") == "SplineWICSI12VAE":
        model = SplineWICSI12VAE.from_configuration(model_configuration)
    else:
        model = WICSI12VAE.from_configuration(model_configuration)
    model.load_state_dict(artifact["model_state"])
    training_configuration = artifact.get("training_configuration", {})
    model.loss_configuration = training_configuration.get(
        "loss_configuration", training_configuration.get("uncertainty_weighting")
    )
    model.to(device)
    model.eval()
    transform = ChannelTransform.from_dict(artifact["transform"])
    return model, transform, artifact
