"""Training and artifact helpers for the corrected-WIC/SI12 VAE."""

#%% Imports

from pathlib import Path
import random

import numpy as np
import torch

from .data import ChannelTransform
from .model import SplineWICSI12VAE, WICSI12VAE, vae_loss


#%% Reproducibility

def set_random_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#%% Training

def _validated_loss_configuration(loss_configuration):
    if loss_configuration is None:
        raise ValueError(
            "loss_configuration is required so uncertainty weighting is explicit"
        )
    if loss_configuration.get("kind") not in (
        "training_median_precision_v1", "fixed_precision_scale_v1"
    ):
        raise ValueError(
            f"unsupported loss configuration {loss_configuration.get('kind')!r}"
        )
    required = (
        "precision_scale", "variance_floor", "maximum_relative_precision"
    )
    missing = [name for name in required if name not in loss_configuration]
    if missing:
        raise ValueError(
            "loss configuration is missing: " + ", ".join(missing)
        )
    precision_scale = np.asarray(
        loss_configuration["precision_scale"], dtype=float
    )
    if precision_scale.shape != (2,) or np.any(~np.isfinite(precision_scale)):
        raise ValueError("loss precision_scale must contain two finite values")
    if np.any(precision_scale <= 0):
        raise ValueError("loss precision_scale values must be positive")
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
):
    loss_configuration = _validated_loss_configuration(loss_configuration)
    model.train(training)
    totals = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0}
    sample_count = 0

    for batch in loader:
        encoder_input = batch["encoder_input"].to(device)
        target = batch["target"].to(device)
        valid = batch["valid"].to(device)
        quality = batch["quality"].to(device)
        variance = batch["variance"].to(device)

        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            reconstructed, mu, logvar = model(encoder_input, valid)
            losses = vae_loss(
                reconstructed,
                target,
                valid,
                quality,
                variance,
                mu,
                logvar,
                beta=beta,
                precision_scale=loss_configuration["precision_scale"],
                variance_floor=loss_configuration["variance_floor"],
                maximum_relative_precision=loss_configuration[
                    "maximum_relative_precision"
                ],
            )
            if training:
                losses["total"].backward()
                optimizer.step()

        batch_size = encoder_input.shape[0]
        sample_count += batch_size
        for name in totals:
            totals[name] += float(losses[name].detach().cpu()) * batch_size

    if sample_count == 0:
        raise ValueError("data loader contains no samples")
    return {name: value / sample_count for name, value in totals.items()}


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
):
    """Train with separate, consistently reduced reconstruction and KL logs."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
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

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta,
            training=True,
            loss_configuration=loss_configuration,
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
):
    """Save everything required to reload the model and identify its inputs."""

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_type": "wic_si12_vae_run_v1",
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
    if artifact.get("artifact_type") != "wic_si12_vae_run_v1":
        raise ValueError("unsupported run artifact")
    model_configuration = artifact["model_configuration"]
    if model_configuration.get("model") == "SplineWICSI12VAE":
        model = SplineWICSI12VAE.from_configuration(model_configuration)
    else:
        model = WICSI12VAE.from_configuration(model_configuration)
    model.load_state_dict(artifact["model_state"])
    model.loss_configuration = artifact.get("training_configuration", {}).get(
        "uncertainty_weighting"
    )
    model.to(device)
    model.eval()
    transform = ChannelTransform.from_dict(artifact["transform"])
    return model, transform, artifact
