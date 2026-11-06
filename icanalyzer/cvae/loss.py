"""Current CVAE reconstruction losses and optional coverage balancing."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def coverage_bin_indices(kp, edges):
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all():
        raise ValueError("coverage Kp edges must be a finite increasing sequence from 0 to 9")
    if edges[0] != 0 or edges[-1] != 9 or np.any(np.diff(edges) <= 0):
        raise ValueError("coverage Kp edges must increase strictly from 0 to 9")
    kp = np.asarray(kp)
    if np.any(~np.isfinite(kp)) or np.any((kp < 0) | (kp > 9)):
        raise ValueError("coverage weighting requires finite Kp within 0–9")
    return np.searchsorted(edges[1:-1], kp, side="right")


def validate_coverage_weighting(configuration):
    edges = np.asarray(configuration.get("kp_bin_edges"), dtype=float)
    coverage_bin_indices([], edges)
    factors = np.asarray(configuration.get("factors"), dtype=float)
    mode = configuration.get("balance_mode", "capped")
    if mode not in ("capped", "full"):
        raise ValueError("coverage balance mode must be capped or full")
    maximum = configuration.get("maximum_boost", np.nan)
    if mode == "full":
        if maximum is not None:
            raise ValueError("full coverage balance must have no maximum boost")
    elif maximum is None or not np.isfinite(maximum) or maximum < 1:
        raise ValueError("maximum coverage boost must be finite and at least 1")
    if factors.ndim != 3 or factors.shape[0] != len(edges) - 1 or min(factors.shape[1:]) < 1:
        raise ValueError("coverage factors must have shape (Kp bins, y, x)")
    if np.any(~np.isfinite(factors)) or np.any(factors < 1):
        raise ValueError("coverage factors must be finite and at least 1")
    if mode == "capped" and np.any(factors > maximum * (1 + np.finfo(np.float32).eps)):
        raise ValueError("coverage factors exceed the maximum boost")


def smooth_coverage_weighting(quality_sum, sigma_pixels=1.5, maximum_boost=3., kp_bin_edges=(0, 9), balance_mode="capped"):
    """Fit one map per Kp bin; unsupported training cells receive no boost."""
    coverage_bin_indices([], kp_bin_edges)
    quality_sum = np.asarray(quality_sum, dtype=np.float64)
    if quality_sum.ndim != 3 or quality_sum.shape[0] != len(kp_bin_edges) - 1:
        raise ValueError("training quality sums must have shape (Kp bins, y, x)")
    if np.any(~np.isfinite(quality_sum)) or np.any(quality_sum < 0):
        raise ValueError("training coverage must be finite and nonnegative")
    if not np.isfinite(sigma_pixels) or sigma_pixels < 0:
        raise ValueError("coverage smoothing width must be finite and nonnegative")
    if balance_mode not in ("capped", "full"):
        raise ValueError("coverage balance mode must be capped or full")
    if balance_mode == "full":
        maximum_boost = None
    elif maximum_boost is None or not np.isfinite(maximum_boost) or maximum_boost < 1:
        raise ValueError("maximum coverage boost must be finite and at least 1")

    # Normalize the Gaussian at unsupported cells and grid edges. A constant
    # observed coverage stays constant, without an artificial boundary boost.
    smoothed = np.zeros_like(quality_sum)
    factors = np.ones_like(quality_sum, dtype=np.float32)
    references = []
    for index, coverage in enumerate(quality_sum):
        supported = coverage > 0
        if not supported.any():
            raise ValueError(f"coverage Kp bin {index} has no positive training observations; use broader bins")
        numerator = gaussian_filter(coverage, sigma_pixels, mode="constant")
        denominator = gaussian_filter(supported.astype(float), sigma_pixels, mode="constant")
        np.divide(numerator, denominator, out=smoothed[index], where=denominator > 0)
        reference = float(smoothed[index][supported].max())
        inverse = reference / smoothed[index][supported]
        factors[index, supported] = inverse if balance_mode == "full" else np.minimum(inverse, maximum_boost)
        references.append(reference)

    configuration = dict(method="smoothed_training_quality_sum", sigma_pixels=float(sigma_pixels),
        balance_mode=balance_mode, maximum_boost=maximum_boost, kp_bin_edges=list(kp_bin_edges), factors=factors.tolist(),
        reference_coverage=references, unsupported_policy="no boost; original validity mask remains authoritative",
        normalization="existing loss divides by boosted batch weight sum")
    validate_coverage_weighting(configuration)
    return configuration, smoothed

LEGACY_LOSS_CONFIGURATION = {
    "kind": "legacy_weighted_mse_v1",
    "reconstruction_reduction": "global_weight_sum_times_state_size",
    "kl_reduction": "sum_batch_and_latent",
    "measurement_uncertainty_in_loss": False
}

MEASUREMENT_UNCERTAINTY_LOSS_KIND = "training_median_precision_v1"
GAUSSIAN_UNCERTAINTY_LOSS_KIND = "quality_weighted_gaussian_v1"


def gaussian_uncertainty_configuration(model_error_std=(0.5, 0.5)):
    """Describe fixed normalized model error combined with measurement variance."""

    return validate_loss_configuration({
        "kind": GAUSSIAN_UNCERTAINTY_LOSS_KIND,
        "measurement_uncertainty_in_loss": True,
        "model_error_std": list(model_error_std),
        "model_error_units": "standardized_log10(value + 1)",
        "variance_model": "measurement_variance_plus_fixed_model_variance",
        "likelihood_scale": "twice_model_variance_per_channel",
        "reconstruction_reduction": "global_quality_sum_times_state_size",
        "kl_reduction": "sum_batch_and_latent"
    })


def validate_loss_configuration(configuration):
    """Validate the loss modes supported by the current automatic CVAEs."""

    if configuration is None:
        raise ValueError("loss_configuration is required so loss semantics are explicit")
    kind = configuration.get("kind")
    supported = ("legacy_weighted_mse_v1", MEASUREMENT_UNCERTAINTY_LOSS_KIND, GAUSSIAN_UNCERTAINTY_LOSS_KIND)
    if kind not in supported:
        raise ValueError(f"unsupported current CVAE loss configuration {kind!r}")
    if configuration.get("coverage_weighting") is not None:
        validate_coverage_weighting(configuration["coverage_weighting"])
    if kind == GAUSSIAN_UNCERTAINTY_LOSS_KIND:
        error_std = np.asarray(configuration.get("model_error_std"), dtype=float)
        if error_std.ndim != 1 or not len(error_std) or np.any(~np.isfinite(error_std)) or np.any(error_std <= 0):
            raise ValueError("loss model_error_std must contain finite positive values")
    if kind == MEASUREMENT_UNCERTAINTY_LOSS_KIND:
        required = ("precision_scale", "variance_floor", "maximum_relative_precision")
        missing = [name for name in required if name not in configuration]
        if missing:
            raise ValueError("loss configuration is missing: " + ", ".join(missing))
        precision_scale = np.asarray(configuration["precision_scale"], dtype=float)
        if precision_scale.ndim != 1 or not len(precision_scale) or np.any(~np.isfinite(precision_scale)) or np.any(precision_scale <= 0):
            raise ValueError("loss precision_scale must contain finite positive values")
        for name in ("variance_floor", "maximum_relative_precision"):
            value = float(configuration[name])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"loss {name} must be finite and positive")
    return dict(configuration)


#%% Reconstruction and KL loss

def vae_loss(
    reconstructed,
    target,
    mask,
    weight,
    mu,
    logvar,
    beta=1.0,
    transformed_variance=None,
    precision_scale=None,
    variance_floor=1e-3,
    maximum_relative_precision=100.0,
    model_error_std=None,
):
    """Return the legacy-scaled reconstruction loss and summed KL term.

    When transformed measurement variance is supplied, inverse variance
    modifies the channel-specific residual weights. Dividing the resulting
    weight sum by the channel count preserves the literal legacy reduction
    when all relative precisions equal one.

    With model_error_std, use fixed-reference Gaussian terms
    tau**2 * (error**2 / (variance + tau**2) + log1p(variance / tau**2)).
    Divide by quality sum, not precision sum. This is a scaled likelihood,
    not an untempered ELBO; it exactly matches legacy MSE at zero variance.
    """

    if reconstructed.ndim == 4:
        if reconstructed.shape != target.shape:
            raise ValueError("reconstructed and target must share shape (batch, channel, y, x)")
    elif reconstructed.ndim == 5:
        expected = (target.shape[0], reconstructed.shape[1], *target.shape[1:])
        if reconstructed.shape != expected or reconstructed.shape[1] < 1:
            raise ValueError("multi-sample reconstruction must have shape (batch, sample, channel, y, x)")
    else:
        raise ValueError("reconstructed values must contain one or more latent samples")
    expected_support_shape = (target.shape[0], 1, *target.shape[-2:])
    if mask.shape != expected_support_shape or weight.shape != expected_support_shape:
        raise ValueError("mask and weight must have shape (batch, 1, y, x)")

    mask = mask.to(dtype=torch.bool)
    weight = torch.where(
        mask & torch.isfinite(weight),
        weight.to(dtype=reconstructed.dtype),
        torch.zeros_like(weight, dtype=reconstructed.dtype),
    )
    weight = torch.clamp(weight, min=0.0)
    likelihood_offset = reconstructed.new_zeros(())
    if model_error_std is not None:
        if transformed_variance is None or transformed_variance.shape != target.shape:
            raise ValueError("Gaussian uncertainty requires variance with the target shape")
        if precision_scale is not None:
            raise ValueError("Gaussian uncertainty cannot also use median precision scaling")
        error_std = torch.as_tensor(model_error_std, dtype=reconstructed.dtype, device=reconstructed.device)
        if error_std.shape != (target.shape[1],) or not torch.all(torch.isfinite(error_std) & (error_std > 0)):
            raise ValueError("model_error_std must contain one finite positive value per channel")
        variance = transformed_variance.to(dtype=reconstructed.dtype)
        finite_variance = torch.isfinite(variance) & (variance >= 0)
        if torch.any((weight > 0) & ~finite_variance):
            raise ValueError("Gaussian uncertainty received invalid measurement variance on weighted cells")
        variance = torch.where(finite_variance, variance, torch.zeros_like(variance))
        model_variance = error_std.square().reshape(1, -1, 1, 1)

        # Fixed reference scaling of Gaussian NLL: zero measurement variance
        # reproduces legacy MSE, and measurement errors only attenuate weight.
        effective_weight = weight * model_variance / (variance + model_variance)
        weight_sum = weight.sum()
        likelihood_offset = torch.sum(weight * model_variance * torch.log1p(variance / model_variance))
    elif transformed_variance is None:
        effective_weight = weight
        weight_sum = effective_weight.sum()
    else:
        if transformed_variance.shape != target.shape:
            raise ValueError(
                "transformed_variance must have the same shape as the target"
            )
        variance_floor = float(variance_floor)
        maximum_relative_precision = float(maximum_relative_precision)
        if not np.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("variance_floor must be finite and positive")
        if (
            not np.isfinite(maximum_relative_precision)
            or maximum_relative_precision <= 0
        ):
            raise ValueError(
                "maximum_relative_precision must be finite and positive"
            )

        if precision_scale is None:
            raise ValueError(
                "precision_scale is required with transformed_variance"
            )
        precision_scale = torch.as_tensor(
            precision_scale,
            dtype=reconstructed.dtype,
            device=reconstructed.device,
        )
        if precision_scale.shape != (target.shape[1],):
            raise ValueError("precision_scale must contain one value per channel")
        if not torch.all(torch.isfinite(precision_scale) & (precision_scale > 0)):
            raise ValueError("precision_scale values must be finite and positive")

        variance = transformed_variance.to(dtype=reconstructed.dtype)
        finite_variance = torch.isfinite(variance) & (variance >= 0)
        precision = torch.where(
            finite_variance,
            1.0 / torch.clamp(variance, min=variance_floor),
            torch.zeros_like(variance),
        )
        relative_precision = precision / precision_scale.reshape(1, -1, 1, 1)
        relative_precision = torch.clamp(
            relative_precision, max=maximum_relative_precision
        )
        effective_weight = weight * relative_precision
        weight_sum = effective_weight.sum() / target.shape[1]

    if reconstructed.ndim == 5:
        squared_error = torch.mean(
            (reconstructed - target[:, None])**2, dim=1
        )
    else:
        squared_error = (reconstructed - target)**2
    state_size = int(np.prod(target.shape[1:]))
    reconstruction_loss = (
        (torch.sum(effective_weight * squared_error) + likelihood_offset)
        / (weight_sum + 1e-8)
        * state_size
    )
    kl_loss = -0.5 * torch.sum(
        1.0 + logvar - mu.pow(2) - logvar.exp()
    )
    total = reconstruction_loss + float(beta) * kl_loss
    return {
        "total": total,
        "reconstruction": reconstruction_loss,
        "kl": kl_loss,
    }
