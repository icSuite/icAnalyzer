"""Training and runtime helpers for the supported automatic CVAEs."""

#%% Imports

import random

import numpy as np
import torch
from tqdm.auto import tqdm

from .loss import (
    CHANNEL_BALANCED_LOSS_KIND,
    GAUSSIAN_UNCERTAINTY_LOSS_KIND,
    LEGACY_LOSS_CONFIGURATION,
    MEASUREMENT_UNCERTAINTY_LOSS_KIND,
    vae_loss,
    validate_loss_configuration
)


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
    """Legacy CPU hole generation retained for exact historical reproduction."""

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
        candidates = torch.nonzero(
            torch.any(mask[index].to(dtype=torch.bool), dim=0),
            as_tuple=False
        )
        if candidates.shape[0] == 0:
            continue
        choice = torch.randint(candidates.shape[0], (), generator=generator).item()
        row, column = candidates[choice].tolist()
        top = min(max(row - hole_size // 2, 0), height - hole_size)
        left = min(max(column - hole_size // 2, 0), width - hole_size)
        remaining = mask[index].clone()
        remaining[:, top:top + hole_size, left:left + hole_size] = False
        if not torch.all(remaining.flatten(1).any(dim=1)):
            continue
        input_mask[index] = remaining
        damaged_input[index, :, top:top + hole_size, left:left + hole_size] = 0.0
    return damaged_input, input_mask


def add_training_holes_vectorized(encoder_input, mask, probability, hole_size, generator):
    """Remove one square per selected image without per-image device synchronization."""

    if probability == 0:
        return encoder_input, mask
    height, width = encoder_input.shape[-2:]
    if not 1 <= hole_size < min(height, width):
        raise ValueError("hole_size must be positive and smaller than the image dimensions")
    if encoder_input.device != mask.device:
        raise ValueError("encoder input and mask must be on the same device")

    batch_size = encoder_input.shape[0]
    selected = torch.rand(
        batch_size, device=mask.device, generator=generator
    ) < probability
    support = torch.any(mask.to(dtype=torch.bool), dim=1).reshape(
        batch_size, -1
    )
    selected &= support.any(dim=1)
    indices = torch.nonzero(selected, as_tuple=False).flatten()

    damaged_input = encoder_input.clone()
    input_mask = mask.clone()
    if indices.numel() == 0:
        return damaged_input, input_mask

    choices = torch.multinomial(
        support[indices].to(dtype=torch.float32), 1,
        replacement=True, generator=generator
    ).flatten()
    center_row = torch.div(choices, width, rounding_mode="floor")
    center_column = choices % width
    top = (center_row - hole_size // 2).clamp(0, height - hole_size)
    left = (center_column - hole_size // 2).clamp(0, width - hole_size)

    rows = torch.arange(height, device=mask.device)[None, :, None]
    columns = torch.arange(width, device=mask.device)[None, None, :]
    holes = (
        (rows >= top[:, None, None])
        & (rows < top[:, None, None] + hole_size)
        & (columns >= left[:, None, None])
        & (columns < left[:, None, None] + hole_size)
    )
    remaining = mask[indices] & ~holes[:, None]
    keep = remaining.flatten(2).any(dim=2).all(dim=1)
    kept_indices = indices[keep]
    kept_holes = holes[keep]

    input_mask[kept_indices] = remaining[keep]
    full_holes = torch.zeros(
        (batch_size, height, width), dtype=torch.bool, device=mask.device
    )
    full_holes[kept_indices] = kept_holes
    damaged_input.masked_fill_(full_holes[:, None], 0.0)
    return damaged_input, input_mask


def resolve_hole_device(requested, training_device):
    """Resolve automatic hole placement while retaining an explicit CPU option."""

    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("hole device must be auto, cpu or cuda")
    training_device = torch.device(training_device)
    resolved = training_device.type if requested == "auto" else requested
    if resolved == "cuda" and training_device.type != "cuda":
        raise ValueError("CUDA hole generation requires CUDA training")
    return resolved


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
    hole_device="cpu",
    latent_samples=1,
    progress_description=None,
):
    device = torch.device(device)
    loss_configuration = validate_loss_configuration(loss_configuration)
    model.train(training)
    metric_names = ("total", "reconstruction", "kl")
    # Match Python's float64 accumulation without synchronizing every GPU batch.
    totals = torch.zeros(len(metric_names), dtype=torch.float64, device=device)
    channel_totals = None
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
            if hole_device == "cuda":
                encoder_input = encoder_input.to(device, non_blocking=non_blocking)
                input_mask = input_mask.to(device, non_blocking=non_blocking)
                encoder_input, input_mask = add_training_holes_vectorized(
                    encoder_input, input_mask, hole_probability, hole_size,
                    hole_generator
                )
            else:
                encoder_input, input_mask = add_training_holes(
                    encoder_input, input_mask, hole_probability, hole_size,
                    hole_generator
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
        if loss_configuration["kind"] == GAUSSIAN_UNCERTAINTY_LOSS_KIND:
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
            if getattr(model, "conditioning", None) is not None:
                if "conditioning" not in batch:
                    raise ValueError(
                        "named-conditioned training requires per-image conditions"
                    )
                condition_arguments["conditioning"] = batch["conditioning"].to(
                    device, non_blocking=non_blocking
                )
            elif getattr(model, "kp_conditioning", None) is not None:
                if "kp" not in batch:
                    raise ValueError("Kp-conditioned training/evaluation requires per-image Kp labels")
                condition_arguments["kp"] = batch["kp"].to(device, non_blocking=non_blocking)
            if latent_samples == 1:
                reconstructed, mu, logvar = model(
                    encoder_input, input_mask, **condition_arguments
                )
            else:
                reconstructed, mu, logvar = model.forward_samples(
                    encoder_input, input_mask,
                    latent_samples=latent_samples, **condition_arguments
                )
            losses = vae_loss(
                reconstructed,
                target,
                mask,
                weight,
                mu,
                logvar,
                beta=beta,
                channel_balanced=(
                    loss_configuration["kind"] == CHANNEL_BALANCED_LOSS_KIND
                ),
                **uncertainty_arguments,
            )
            if training:
                losses["total"].backward()
                optimizer.step()

        batch_size = encoder_input.shape[0]
        sample_count += batch_size
        totals += torch.stack([losses[name].detach().to(dtype=torch.float64) for name in metric_names])
        if "channel_reconstruction" in losses:
            current = losses["channel_reconstruction"].detach().to(
                dtype=torch.float64
            )
            if channel_totals is None:
                channel_totals = torch.zeros_like(current)
            channel_totals += current

    if sample_count == 0:
        raise ValueError("data loader contains no samples")
    metrics = {
        name: value / sample_count
        for name, value in zip(metric_names, totals.cpu().tolist())
    }
    if channel_totals is not None:
        metrics["channel_reconstruction"] = [
            value / sample_count for value in channel_totals.cpu().tolist()
        ]
    return metrics


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
    hole_device="auto",
    latent_samples=1,
):
    """Train with separate, consistently reduced reconstruction and KL logs."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not np.isfinite(hole_probability) or not 0 <= hole_probability <= 1:
        raise ValueError("hole_probability must be finite and between zero and one")
    if not isinstance(hole_size, (int, np.integer)) or hole_size < 1:
        raise ValueError("hole_size must be a positive integer")
    if not isinstance(latent_samples, (int, np.integer)) or latent_samples < 1:
        raise ValueError("latent_samples must be a positive integer")
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
    loss_configuration = validate_loss_configuration(loss_configuration)
    device = torch.device(device)
    resolved_hole_device = resolve_hole_device(hole_device, device)
    set_random_seed(seed)
    hole_generator = (
        torch.Generator(device=resolved_hole_device).manual_seed(int(seed) + 3)
        if hole_probability > 0 else None
    )
    hole_arguments = {
        "hole_probability": hole_probability,
        "hole_size": hole_size,
        "hole_generator": hole_generator,
        "hole_device": resolved_hole_device
    } if hole_probability > 0 else {}
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

    print(f"Training VAE on {device}: {epochs} epochs, {len(train_loader)} training batches per epoch; {latent_samples} latent sample(s) per image.", flush=True)
    if hole_probability > 0:
        print(f"Training-hole generation: {resolved_hole_device}.", flush=True)
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
                latent_samples=latent_samples,
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
                    latent_samples=latent_samples,
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


#%% Device selection

def training_device(requested):
    """Resolve CPU/CUDA selection and fail before creating run output."""

    if requested == "cpu":
        return "cpu"
    available = torch.cuda.is_available()
    if requested == "auto" and not available:
        return "cpu"
    if not available:
        raise RuntimeError(
            "CUDA is unavailable. Check that nvidia-smi works, or explicitly "
            "choose --device cpu. No run directory was created."
        )
    try:
        torch.empty(1, device="cuda")
    except RuntimeError as error:
        raise RuntimeError(
            "CUDA could not initialize. Check the NVIDIA driver with "
            "nvidia-smi, or explicitly choose --device cpu. No run directory "
            "was created."
        ) from error
    return "cuda"
