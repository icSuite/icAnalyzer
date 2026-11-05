"""Opt-in count-space likelihood for clipped corrected-WIC/SI12 products."""

import numpy as np
import torch

from .data import ChannelTransform


COUNT_UNCERTAINTY_LOSS_KIND = "quality_weighted_censored_counts_v1"


#%% Training-only channel scales

def validate_count_configuration(configuration):
    if configuration.get("kind") != COUNT_UNCERTAINTY_LOSS_KIND:
        raise ValueError("unsupported count loss configuration")
    for name in ("channel_scale_counts", "model_error_fraction"):
        values = np.asarray(configuration.get(name), dtype=float)
        if values.shape != (2,) or np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"count loss {name} must contain two finite positive values")
    transform = ChannelTransform.from_dict(configuration["transform"])
    if transform.kind != ChannelTransform.LOG10_KIND:
        raise ValueError("count likelihood requires the canonical log10(value + 1) transform")
    if np.any(~np.isfinite(transform.center)) or np.any(~np.isfinite(transform.spread)):
        raise ValueError("count likelihood requires finite transform parameters")
    return dict(configuration)


def fit_count_uncertainty_configuration(dataset, model_error_fraction=(0.5, 0.5)):
    """Fit quality-weighted RMS counts on retained training frames only.

    Channel scales remove the WIC/SI12 count-unit disparity, not measurement
    information. Model discrepancy is a fixed fraction of each fitted scale.
    """

    scales = []
    censored_fraction = []
    for channel in range(2):
        values = dataset.values[:, channel].astype(np.float64)
        weight = dataset.weight[:, 0].astype(np.float64)
        supported = dataset.common_valid[:, 0] & np.isfinite(values) & (weight > 0)
        weight = np.where(supported, weight, 0.0)
        values = np.where(supported, values, 0.0)
        weight_sum = weight.sum()
        if weight_sum <= 0:
            raise ValueError("count scale requires positive training support in each channel")
        scales.append(float(np.sqrt(np.sum(weight * values**2) / weight_sum)))
        censored_fraction.append(float(np.sum(weight * (values == 0)) / weight_sum))

    return validate_count_configuration({
        "kind": COUNT_UNCERTAINTY_LOSS_KIND,
        "measurement_uncertainty_in_loss": True,
        "channel_scale_counts": scales,
        "channel_scale_fit": "quality_weighted_rms_retained_training_frames",
        "model_error_fraction": list(model_error_fraction),
        "model_error_units": "fraction_of_training_channel_scale_counts",
        "transform": dataset.transform.to_dict(),
        "zero_policy": "all_observed_zeros_left_censored_at_zero",
        "observation_model": "CS_level_Gaussian_approximation_not_raw_Poisson",
        "prediction_mean": "unclipped_inverse_log10_preclipping_counts",
        "likelihood_scale": "twice_squared_model_error_fraction_per_channel",
        "reconstruction_reduction": "global_quality_sum_times_state_size",
        "kl_reduction": "sum_batch_and_latent",
        "training_quality_weighted_zero_fraction": censored_fraction
    })


#%% Count likelihood, with a stable log-CDF for zero measurements

def count_space_reconstruction_loss(reconstructed, observed, standard_deviation, mask, weight, configuration):
    """Quality-weighted, channel-reference-scaled censored Gaussian NLL.

    Positive counts use Gaussian density; zero counts use P(preclip_count <= 0).
    Uncertainties remain in counts: no log-space delta-method variance is used.
    The decoder's unclipped inverse supplies a pre-clipping corrected-count
    mean. It can approach -1 count, avoiding a dead gradient below zero; saved
    physical reconstructions still use the existing inverse's zero clipping.

    This is an approximate CS-level observation model: native-pixel WIC was
    clipped BEFORE area averaging, whose exact likelihood is unavailable here.
    """

    validate_count_configuration(configuration)
    if observed is None or standard_deviation is None:
        raise ValueError("count likelihood requires physical observations and standard deviations")
    if observed.shape != reconstructed.shape or standard_deviation.shape != observed.shape:
        raise ValueError("count observations and uncertainties must match the reconstruction shape")
    if reconstructed.ndim != 4 or reconstructed.shape[1] != 2:
        raise ValueError("count reconstruction must have shape (batch, 2, y, x)")
    support_shape = (observed.shape[0], 1, *observed.shape[-2:])
    if mask.shape != support_shape or weight.shape != support_shape:
        raise ValueError("count mask and weight must have shape (batch, 1, y, x)")
    weight = torch.where(mask.bool() & torch.isfinite(weight), weight.clamp(min=0), 0.0)
    active = (weight > 0).expand_as(observed)
    valid = torch.isfinite(observed) & (observed >= 0)
    valid &= torch.isfinite(standard_deviation) & (standard_deviation >= 0)
    valid &= torch.isfinite(reconstructed)
    if torch.any(active & ~valid):
        raise ValueError("count likelihood received invalid data on weighted cells")

    # Neutralize missing cells BEFORE exponentiation and residual arithmetic.
    decoded = torch.where(active, reconstructed, 0.0)
    observed = torch.where(active, observed, 0.0)
    sigma = torch.where(active, standard_deviation, 0.0)
    def channel_tensor(values):
        return torch.as_tensor(values, dtype=decoded.dtype, device=decoded.device).reshape(1, 2, 1, 1)

    center = channel_tensor(configuration["transform"]["center"])
    spread = channel_tensor(configuration["transform"]["spread"])
    scale = channel_tensor(configuration["channel_scale_counts"])
    reference_variance = channel_tensor(configuration["model_error_fraction"]).square()
    if torch.any(~torch.isfinite(reference_variance) | (reference_variance <= 0)):
        raise ValueError("count model variance is not representable in the reconstruction dtype")
    mean_counts = torch.expm1(np.log(10.0) * (decoded * spread + center))
    mean_counts = torch.where(active, mean_counts, 0.0)
    if torch.any(active & ~torch.isfinite(mean_counts)):
        raise ValueError("count likelihood inverse transform overflowed on weighted cells")

    # Normalize counts and their errors together; original sigma information
    # is retained. Fixed discrepancy prevents infinite precision at sigma=0.
    mean = mean_counts / scale
    observation = observed / scale
    variance = (sigma / scale).square() + reference_variance
    if torch.any(active & ~torch.isfinite(variance)):
        raise ValueError("count likelihood variance overflowed on weighted cells")
    density_loss = reference_variance * ((mean - observation).square() / variance + torch.log(variance / reference_variance))
    censored_loss = -2.0 * reference_variance * torch.special.log_ndtr(-mean / torch.sqrt(variance))
    cell_loss = torch.where(observed == 0, censored_loss, density_loss)
    cell_loss = torch.where(active, cell_loss, 0.0)
    if torch.any(~torch.isfinite(cell_loss)):
        raise ValueError("count likelihood overflowed; inspect decoder counts and configured scales")
    state_size = int(np.prod(observed.shape[1:]))
    return torch.sum(weight * cell_loss) / (weight.sum() + 1e-8) * state_size
