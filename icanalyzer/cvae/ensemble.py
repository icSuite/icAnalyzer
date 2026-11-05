"""Latent sampling and ensemble statistics for supported automatic CVAEs."""

#%% Imports

import numpy as np
import torch


#%% Latent sampling

def nonnegative_wic_floor(transform):
    """Return the first float32 transformed value safely above physical zero."""

    zero = np.float32(-transform.center[0] / transform.spread[0])
    return float(np.nextafter(zero, np.float32(np.inf)))


def correct_wic_roundoff(samples, transform):
    """Set only float32 roundoff below the physical WIC boundary to zero."""

    samples = np.asarray(samples)
    if samples.ndim != 4 or samples.shape[1] < 1:
        raise ValueError("decoded samples must have shape (sample, channel, y, x)")

    wic = samples[:, 0]
    negative = wic < 0
    corrected_count = int(np.count_nonzero(negative))
    tolerance = float(
        8 * np.finfo(np.float32).eps * max(abs(float(transform.scale[0])), 1.0)
    )
    minimum = float(np.min(wic))
    if minimum < -tolerance:
        fraction = corrected_count / wic.size
        raise ValueError(
            "decoded corrected WIC contains materially negative values "
            f"(minimum {minimum:.8g} counts; {fraction:.3%} of cells); "
            "do not clip silently"
        )
    if corrected_count:
        samples = samples.copy()
        samples[:, 0][negative] = 0.0

    diagnostics = {
        "roundoff_corrected_cell_count": corrected_count,
        "roundoff_corrected_fraction": corrected_count / wic.size,
        "minimum_before_correction_counts": minimum,
        "roundoff_tolerance_counts": tolerance,
    }
    return samples, diagnostics

def sample_decoded_ensemble(
    model,
    transform,
    n_samples,
    batch_size=256,
    device="cpu",
    seed=0,
    return_latent=False,
    support_mask=None,
    kp=None,
):
    """Decode exactly ``n_samples`` conditional field samples."""

    if n_samples < 2:
        raise ValueError("at least two latent samples are required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    model = model.to(device)
    model.eval()
    field_names = getattr(model, "field_names", None)
    if field_names is not None and len(field_names) != transform.channel_count:
        raise ValueError("model output fields and channel transform do not align")
    conditioned = getattr(model, "kp_conditioning", None) is not None
    if conditioned and kp is None:
        raise ValueError("Kp-conditioned sampling requires an explicit requested Kp")
    if not conditioned and kp is not None:
        raise ValueError("cannot request Kp from an unconditional model")
    if conditioned:
        kp = np.asarray(kp, dtype=np.float32)
        if kp.ndim == 0:
            kp = np.full(n_samples, kp, dtype=np.float32)
        if kp.shape != (n_samples,) or not np.isfinite(kp).all() or np.any((kp < 0) | (kp > 9)):
            raise ValueError("sample Kp must be a finite 0–9 scalar or one value per member")
    if support_mask is not None:
        support_mask = np.asarray(support_mask, dtype=bool)
        if support_mask.shape != tuple(model.image_shape) or not support_mask.any():
            raise ValueError("support_mask must match the grid and contain supported cells")
        supported = torch.as_tensor(support_mask, device=device)[None, None]
        physical_zero = transform.transform(np.zeros((1, transform.channel_count, 1, 1), dtype=np.float32))
        physical_zero = torch.as_tensor(physical_zero, device=device)

    physical = np.empty(
        (n_samples, transform.channel_count, *model.image_shape), dtype=np.float32
    )
    latent_samples = (
        np.empty((n_samples, model.latent_dim), dtype=np.float32)
        if return_latent else None
    )
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            current_size = min(batch_size, n_samples - start)
            stop = start + current_size
            latent = torch.randn(
                current_size,
                model.latent_dim,
                generator=generator,
                device=device,
            )
            if return_latent:
                latent_samples[start:stop] = latent.cpu().numpy()
            condition_arguments = {"kp": torch.as_tensor(kp[start:stop], device=device)} if conditioned else {}
            transformed = model.decode(latent, **condition_arguments)
            if support_mask is not None:
                # Numeric placeholders only: never fabricate zero-valued training targets.
                # Mask BEFORE exponentiation to avoid overflow outside learned support.
                transformed = torch.where(supported, transformed, physical_zero)
            physical[start:stop] = transform.inverse(transformed).cpu().numpy()
            if support_mask is not None:
                physical[start:stop, :, ~support_mask] = 0.0

    if not return_latent:
        return physical
    return physical, latent_samples


#%% Ensemble moments

def ensemble_statistics(samples):
    """Return a mean field and normalized anomaly factor.

    If ``samples`` has shape ``(sample, channel, y, x)``, the anomaly factor
    has shape ``(channel*y*x, sample)`` and satisfies ``cov = A @ A.T``.
    """

    samples = np.asarray(samples)
    if samples.ndim < 2 or samples.shape[0] < 2:
        raise ValueError("samples must contain at least two ensemble members")
    if not np.issubdtype(samples.dtype, np.floating):
        samples = samples.astype(np.float32)
    if np.any(~np.isfinite(samples)):
        raise ValueError("ensemble statistics require finite samples")

    mean = samples.mean(axis=0, dtype=np.float64)
    flattened = samples.reshape(samples.shape[0], -1)
    anomaly_factor = flattened.T.copy()
    anomaly_factor -= mean.reshape(-1, 1)
    anomaly_factor /= np.sqrt(samples.shape[0] - 1)
    variance = np.einsum(
        "ij,ij->i", anomaly_factor, anomaly_factor,
        dtype=np.float64, optimize=True
    ).reshape(samples.shape[1:])
    return {
        "mean": mean,
        "anomaly_factor": anomaly_factor,
        "state_shape": samples.shape[1:],
        "sample_count": samples.shape[0],
        "variance": variance,
    }


