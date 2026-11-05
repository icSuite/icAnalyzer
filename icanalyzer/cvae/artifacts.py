"""Save and load supported Hall/Pedersen and WIC CVAE artifacts."""

from pathlib import Path

import torch

from .data import ChannelTransform
from .model import SplineConditionalVAE


CONDUCTANCE_CVAE_ARTIFACT_TYPE = "conductance_vae_run_v1"
WIC_CVAE_ARTIFACT_TYPE = "wic_cvae_run_v1"
SUPPORTED_ARTIFACT_TYPES = (
    CONDUCTANCE_CVAE_ARTIFACT_TYPE,
    WIC_CVAE_ARTIFACT_TYPE
)


def save_run_artifact(
    filename,
    model,
    transform,
    history,
    training_configuration,
    data_provenance,
    split_files,
    evaluation=None,
    artifact_type=WIC_CVAE_ARTIFACT_TYPE
):
    """Save everything required to reload a supported automatic CVAE."""

    if artifact_type not in SUPPORTED_ARTIFACT_TYPES:
        raise ValueError(f"unsupported current CVAE artifact type {artifact_type!r}")
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
        "evaluation": {} if evaluation is None else dict(evaluation)
    }
    torch.save(artifact, filename)


def load_run_artifact(filename, device="cpu"):
    """Load current artifacts and pre-rename spline-model artifacts."""

    artifact = torch.load(filename, map_location=device, weights_only=False)
    if artifact.get("artifact_type") not in SUPPORTED_ARTIFACT_TYPES:
        raise ValueError("expected a Hall/Pedersen or WIC automatic CVAE artifact")
    configuration = dict(artifact["model_configuration"])
    if configuration.get("model") not in ("SplineConditionalVAE", "SplineWICSI12VAE"):
        raise ValueError("current automatic CVAEs require the spline conditional model")
    if (
        artifact["artifact_type"] == CONDUCTANCE_CVAE_ARTIFACT_TYPE
        and configuration.get("field_names") in (None, ["wic_corrected", "si12"])
    ):
        # Old conductance runs inherited the former class's WIC/SI12 labels.
        # Only the labels were wrong; the trained state is the same two-channel model.
        configuration["field_names"] = ["Hall", "Pedersen"]
    model = SplineConditionalVAE.from_configuration(configuration)
    model.load_state_dict(artifact["model_state"])
    training = artifact.get("training_configuration", {})
    model.loss_configuration = training.get(
        "loss_configuration", training.get("uncertainty_weighting")
    )
    model.to(device)
    model.eval()
    transform = ChannelTransform.from_dict(artifact["transform"])
    return model, transform, artifact
