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

def _sample_model_prior(model, count, generator, condition_arguments):
    """Sample the model-owned prior, retaining legacy normal compatibility."""

    sampler = getattr(model, "sample_prior", None)
    if callable(sampler):
        return sampler(
            count, generator=generator, **condition_arguments
        )
    reference = next(model.parameters())
    return torch.randn(
        count, model.latent_dim, dtype=reference.dtype,
        device=reference.device, generator=generator,
    )


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
    conditioning=None,
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
    kp_conditioned = getattr(model, "kp_conditioning", None) is not None
    named_conditioned = getattr(model, "conditioning", None) is not None
    if kp_conditioned and kp is None:
        raise ValueError("Kp-conditioned sampling requires an explicit requested Kp")
    if not kp_conditioned and kp is not None:
        raise ValueError("cannot request Kp from an unconditional model")
    if named_conditioned and conditioning is None:
        raise ValueError("named-conditioned sampling requires explicit conditions")
    if not named_conditioned and conditioning is not None:
        raise ValueError("model does not accept named conditions")
    if kp_conditioned:
        kp = np.asarray(kp, dtype=np.float32)
        if kp.ndim == 0:
            kp = np.full(n_samples, kp, dtype=np.float32)
        if kp.shape != (n_samples,) or not np.isfinite(kp).all() or np.any((kp < 0) | (kp > 9)):
            raise ValueError("sample Kp must be a finite 0–9 scalar or one value per member")
    if named_conditioned:
        conditioning = np.asarray(conditioning, dtype=np.float32)
        feature_count = len(model.conditioning["feature_names"])
        if conditioning.shape == (feature_count,):
            conditioning = np.broadcast_to(
                conditioning, (n_samples, feature_count)
            ).copy()
        if (
            conditioning.shape != (n_samples, feature_count)
            or not np.isfinite(conditioning).all()
        ):
            raise ValueError(
                "sample conditions must be one finite vector or one row per member"
            )
    if support_mask is not None:
        support_mask = np.asarray(support_mask, dtype=bool)
        grid_shape = tuple(model.image_shape)
        if support_mask.shape == grid_shape:
            support_mask = np.broadcast_to(
                support_mask, (transform.channel_count, *grid_shape)
            ).copy()
        if (
            support_mask.shape != (transform.channel_count, *grid_shape)
            or not np.all(support_mask.reshape(transform.channel_count, -1).any(axis=1))
        ):
            raise ValueError(
                "support_mask must contain one nonempty grid per output channel"
            )
        supported = torch.as_tensor(support_mask, device=device)[None]
        physical_zero = transform.transform(
            np.zeros((1, transform.channel_count, 1, 1), dtype=np.float32)
        )
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
            condition_arguments = {}
            if kp_conditioned:
                condition_arguments["kp"] = torch.as_tensor(
                    kp[start:stop], device=device
                )
            if named_conditioned:
                condition_arguments["conditioning"] = torch.as_tensor(
                    conditioning[start:stop], device=device
                )
            latent = _sample_model_prior(
                model, current_size, generator, condition_arguments
            )
            if return_latent:
                latent_samples[start:stop] = latent.cpu().numpy()
            transformed = model.decode(latent, **condition_arguments)
            if support_mask is not None:
                # Numeric placeholders only: never fabricate zero-valued training targets.
                # Mask BEFORE exponentiation to avoid overflow outside learned support.
                transformed = torch.where(supported, transformed, physical_zero)
            physical[start:stop] = transform.inverse(transformed).cpu().numpy()
            if support_mask is not None:
                physical[start:stop] = np.where(
                    support_mask[None], physical[start:stop], 0.0
                )

    if not return_latent:
        return physical
    return physical, latent_samples


def sample_decoded_members_to_disk(
    model,
    transform,
    n_samples,
    members_filename,
    latent_filename,
    batch_size=256,
    device="cpu",
    seed=0,
    support_mask=None,
    kp=None,
    conditioning=None,
):
    """Decode an ensemble into state-by-member disk arrays.

    The state-major layout makes exact per-pixel quantiles and anomaly-factor
    construction contiguous without ever materializing the full ensemble in
    memory. The latent sequence matches :func:`sample_decoded_ensemble` for
    the same model, device, seed and conditioning.
    """

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

    kp_conditioned = getattr(model, "kp_conditioning", None) is not None
    named_conditioned = getattr(model, "conditioning", None) is not None
    if kp_conditioned and kp is None:
        raise ValueError("Kp-conditioned sampling requires an explicit requested Kp")
    if not kp_conditioned and kp is not None:
        raise ValueError("cannot request Kp from an unconditional model")
    if named_conditioned and conditioning is None:
        raise ValueError("named-conditioned sampling requires explicit conditions")
    if not named_conditioned and conditioning is not None:
        raise ValueError("model does not accept named conditions")

    if kp_conditioned:
        kp = np.asarray(kp, dtype=np.float32)
        if kp.ndim == 0:
            kp = np.full(n_samples, kp, dtype=np.float32)
        if (
            kp.shape != (n_samples,) or not np.isfinite(kp).all()
            or np.any((kp < 0) | (kp > 9))
        ):
            raise ValueError(
                "sample Kp must be a finite 0–9 scalar or one value per member"
            )
    if named_conditioned:
        conditioning = np.asarray(conditioning, dtype=np.float32)
        feature_count = len(model.conditioning["feature_names"])
        if conditioning.shape == (feature_count,):
            conditioning = np.broadcast_to(
                conditioning, (n_samples, feature_count)
            ).copy()
        if (
            conditioning.shape != (n_samples, feature_count)
            or not np.isfinite(conditioning).all()
        ):
            raise ValueError(
                "sample conditions must be one finite vector or one row per member"
            )

    grid_shape = tuple(model.image_shape)
    if support_mask is not None:
        support_mask = np.asarray(support_mask, dtype=bool)
        if support_mask.shape == grid_shape:
            support_mask = np.broadcast_to(
                support_mask, (transform.channel_count, *grid_shape)
            ).copy()
        if (
            support_mask.shape != (transform.channel_count, *grid_shape)
            or not np.all(
                support_mask.reshape(transform.channel_count, -1).any(axis=1)
            )
        ):
            raise ValueError(
                "support_mask must contain one nonempty grid per output channel"
            )
        supported = torch.as_tensor(support_mask, device=device)[None]
        physical_zero = transform.transform(
            np.zeros((1, transform.channel_count, 1, 1), dtype=np.float32)
        )
        physical_zero = torch.as_tensor(physical_zero, device=device)

    state_size = transform.channel_count * int(np.prod(grid_shape))
    members = np.lib.format.open_memmap(
        members_filename, mode="w+", dtype=np.float32,
        shape=(state_size, n_samples)
    )
    latent_samples = np.lib.format.open_memmap(
        latent_filename, mode="w+", dtype=np.float32,
        shape=(n_samples, model.latent_dim)
    )

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            condition_arguments = {}
            if kp_conditioned:
                condition_arguments["kp"] = torch.as_tensor(
                    kp[start:stop], device=device
                )
            if named_conditioned:
                condition_arguments["conditioning"] = torch.as_tensor(
                    conditioning[start:stop], device=device
                )
            latent = _sample_model_prior(
                model, stop - start, generator, condition_arguments
            )
            transformed = model.decode(latent, **condition_arguments)
            if support_mask is not None:
                transformed = torch.where(
                    supported, transformed, physical_zero
                )
            physical = transform.inverse(transformed).cpu().numpy()
            if support_mask is not None:
                physical = np.where(
                    support_mask[None], physical, 0.0
                )
            members[:, start:stop] = physical.reshape(stop - start, -1).T
            latent_samples[start:stop] = latent.cpu().numpy()

    members.flush()
    latent_samples.flush()
    return members, latent_samples


def state_major_summaries(
    members, state_shape, quantiles=(0.1, 0.5, 0.9), working_mib=256
):
    """Compute exact means and quantiles from a state-by-member disk array."""

    members = np.asanyarray(members)
    state_shape = tuple(state_shape)
    state_size = int(np.prod(state_shape))
    quantiles = np.asarray(quantiles, dtype=float)
    if (
        members.ndim != 2 or members.shape[0] != state_size
        or members.shape[1] < 2 or working_mib <= 0
        or quantiles.ndim != 1 or not len(quantiles)
        or not np.isfinite(quantiles).all()
        or np.any((quantiles < 0) | (quantiles > 1))
    ):
        raise ValueError("state-major ensemble summary inputs are invalid")

    member_count = members.shape[1]
    bytes_per_state = member_count * members.dtype.itemsize
    states_per_block = max(
        1, int(working_mib * 1024**2 // max(bytes_per_state, 1))
    )
    mean = np.empty(state_size, dtype=np.float64)
    values = np.empty((len(quantiles), state_size), dtype=np.float64)
    for start in range(0, state_size, states_per_block):
        stop = min(start + states_per_block, state_size)
        block = np.asarray(members[start:stop])
        if not np.isfinite(block).all() or np.any(block < 0):
            raise ValueError("ensemble members must be finite and nonnegative")
        mean[start:stop] = block.mean(axis=1, dtype=np.float64)
        values[:, start:stop] = np.quantile(block, quantiles, axis=1)
    return mean.reshape(state_shape), values.reshape(
        len(quantiles), *state_shape
    )


def state_major_anomaly_factor_in_place(
    members, mean, working_mib=256
):
    """Replace state-major members with normalized anomalies in bounded blocks."""

    members = np.asanyarray(members)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    if (
        members.ndim != 2 or members.shape[0] != len(mean)
        or members.shape[1] < 2 or working_mib <= 0
    ):
        raise ValueError("state-major anomaly inputs are invalid")
    bytes_per_state = members.shape[1] * members.dtype.itemsize
    states_per_block = max(
        1, int(working_mib * 1024**2 // max(bytes_per_state, 1))
    )
    normalization = np.float32(np.sqrt(members.shape[1] - 1))
    for start in range(0, len(mean), states_per_block):
        stop = min(start + states_per_block, len(mean))
        block = members[start:stop]
        block -= mean[start:stop, None]
        block /= normalization
    if hasattr(members, "flush"):
        members.flush()
    return members


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


