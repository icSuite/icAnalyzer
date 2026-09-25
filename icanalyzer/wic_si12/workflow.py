"""Training and artifact helpers for the corrected-WIC/SI12 VAE."""

#%% Imports

from pathlib import Path
import random

import numpy as np
import torch

from .data import ChannelTransform
from .model import WICSI12VAE, vae_loss


#%% Reproducibility

def set_random_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#%% Training

def _run_epoch(model, loader, optimizer, device, beta, training):
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
            reconstructed, mu, logvar = model(encoder_input)
            losses = vae_loss(
                reconstructed,
                target,
                valid,
                quality,
                variance,
                mu,
                logvar,
                beta=beta,
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
):
    """Train with separate, consistently reduced reconstruction and KL logs."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    set_random_seed(seed)
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, optimizer, device, beta, training=True
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
            )
        history.append(record)
    return model, history


def evaluate_vae(model, loader, beta=1e-3, device="cpu"):
    """Evaluate the same reduced loss without changing model parameters."""

    return _run_epoch(
        model,
        loader,
        optimizer=None,
        device=torch.device(device),
        beta=beta,
        training=False,
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
    model = WICSI12VAE.from_configuration(artifact["model_configuration"])
    model.load_state_dict(artifact["model_state"])
    model.to(device)
    model.eval()
    transform = ChannelTransform.from_dict(artifact["transform"])
    return model, transform, artifact
