"""Generic data transforms and reproducible orbit splitting for CVAEs."""

from pathlib import Path
import re

import numpy as np
import torch


def _orbit_from_path(path):
    match = re.search(r"or_(\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def split_orbit_files(filenames, validation_fraction=0.15, test_fraction=0.15, seed=0):
    """Split complete orbit files reproducibly without frame leakage."""

    paths = [Path(path) for path in filenames]
    if not paths:
        raise ValueError("at least one orbit file is required")
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below one")

    orbit_path_pairs = []
    for path in paths:
        orbit = _orbit_from_path(path)
        if orbit is None:
            raise ValueError(f"cannot determine orbit from {path.name!r}")
        orbit_path_pairs.append((orbit, path))
    if len({orbit for orbit, _ in orbit_path_pairs}) != len(orbit_path_pairs):
        raise ValueError("each orbit must appear in exactly one input file")

    generator = np.random.default_rng(seed)
    order = generator.permutation(len(paths))
    shuffled = [orbit_path_pairs[index][1] for index in order]

    n_files = len(shuffled)
    n_test = int(np.floor(n_files * test_fraction))
    n_validation = int(np.floor(n_files * validation_fraction))
    if test_fraction > 0 and n_files >= 3:
        n_test = max(1, n_test)
    if validation_fraction > 0 and n_files - n_test >= 2:
        n_validation = max(1, n_validation)

    return {
        "test": shuffled[:n_test],
        "validation": shuffled[n_test:n_test + n_validation],
        "train": shuffled[n_test + n_validation:],
    }


def fit_kp_conditioning(kp):
    """Fit scalar Kp normalization on retained training images only."""

    kp = np.asarray(kp, dtype=float)
    if kp.ndim != 1 or len(kp) < 2 or not np.isfinite(kp).all() or np.any((kp < 0) | (kp > 9)):
        raise ValueError("training Kp must be a finite 0–9 vector with at least two images")
    scale = float(kp.std())
    if scale <= 0:
        raise ValueError("Kp-conditioned training requires more than one Kp value")
    return {
        "center": float(kp.mean()),
        "scale": scale,
        "training_min": float(kp.min()),
        "training_max": float(kp.max()),
        "embedding_dim": 8
    }


#%% Channel transformation

class ChannelTransform:
    """Versioned channel transform followed by channel-wise standardization."""

    ASINH_KIND = "channel_asinh_standardization_v1"
    LOG10_KIND = "channel_log10p_standardization_v1"

    def __init__(self, scale, center, spread, kind=ASINH_KIND):
        self.scale = np.asarray(scale, dtype=np.float32)
        self.center = np.asarray(center, dtype=np.float32)
        self.spread = np.asarray(spread, dtype=np.float32)
        self.kind = str(kind)
        if self.scale.ndim != 1 or not self.scale.size or self.center.shape != self.scale.shape or self.spread.shape != self.scale.shape:
            raise ValueError("scale, center, and spread must be nonempty and have one value per channel")
        if np.any(self.scale <= 0) or np.any(self.spread <= 0):
            raise ValueError("scale and spread must be positive")
        if self.kind not in (self.ASINH_KIND, self.LOG10_KIND):
            raise ValueError(f"unsupported channel transform {self.kind!r}")

    @property
    def channel_count(self):
        return int(self.scale.size)

    def _channel_shape(self, values):
        if values.ndim < 2 or values.shape[1] != self.channel_count:
            raise ValueError(f"values must have {self.channel_count} channels on axis 1")
        return (1, self.channel_count) + (1,) * (values.ndim - 2)

    def transform(self, values):
        if torch.is_tensor(values):
            scale = torch.as_tensor(self.scale, device=values.device, dtype=values.dtype)
            center = torch.as_tensor(self.center, device=values.device, dtype=values.dtype)
            spread = torch.as_tensor(self.spread, device=values.device, dtype=values.dtype)
            shape = self._channel_shape(values)
            if self.kind == self.LOG10_KIND:
                if torch.any(values < 0):
                    raise ValueError("log10 channel transform requires nonnegative values")
                transformed = torch.log10(values + 1.0)
            else:
                transformed = torch.asinh(values / scale.reshape(shape))
            return (
                transformed - center.reshape(shape)
            ) / spread.reshape(shape)

        values = np.asarray(values)
        shape = self._channel_shape(values)
        if self.kind == self.LOG10_KIND:
            if np.any(values < 0):
                raise ValueError("log10 channel transform requires nonnegative values")
            transformed = np.log10(values + 1.0)
        else:
            transformed = np.arcsinh(values / self.scale.reshape(shape))
        return (
            transformed - self.center.reshape(shape)
        ) / self.spread.reshape(shape)

    def inverse(self, transformed):
        if torch.is_tensor(transformed):
            scale = torch.as_tensor(self.scale, device=transformed.device, dtype=transformed.dtype)
            center = torch.as_tensor(self.center, device=transformed.device, dtype=transformed.dtype)
            spread = torch.as_tensor(self.spread, device=transformed.device, dtype=transformed.dtype)
            shape = self._channel_shape(transformed)
            restored = transformed * spread.reshape(shape) + center.reshape(shape)
            if self.kind == self.LOG10_KIND:
                return torch.clamp(torch.pow(10.0, restored) - 1.0, min=0.0)
            return scale.reshape(shape) * torch.sinh(restored)

        transformed = np.asarray(transformed)
        shape = self._channel_shape(transformed)
        restored = transformed * self.spread.reshape(shape) + self.center.reshape(shape)
        if self.kind == self.LOG10_KIND:
            return np.maximum(np.power(10.0, restored) - 1.0, 0.0)
        return self.scale.reshape(shape) * np.sinh(restored)

    def variance(self, values, standard_deviation):
        values = np.asarray(values, dtype=np.float32)
        standard_deviation = np.asarray(standard_deviation, dtype=np.float32)
        if standard_deviation.shape != values.shape:
            raise ValueError("standard deviation must match the channel values")
        shape = self._channel_shape(values)
        scale = self.scale.reshape(shape)
        spread = self.spread.reshape(shape)
        if self.kind == self.LOG10_KIND:
            if np.any(values < 0):
                raise ValueError("log10 variance transform requires nonnegative values")
            derivative = 1.0 / ((values + 1.0) * np.log(10.0) * spread)
        else:
            derivative = 1.0 / (
                scale * np.sqrt(1.0 + (values / scale)**2) * spread
            )
        return (standard_deviation * derivative)**2

    def to_dict(self):
        return {
            "kind": self.kind,
            "scale": self.scale.tolist(),
            "center": self.center.tolist(),
            "spread": self.spread.tolist(),
        }

    @classmethod
    def from_dict(cls, values):
        return cls(
            values["scale"], values["center"], values["spread"],
            kind=values.get("kind", cls.ASINH_KIND)
        )


def fit_channel_transform(values, valid):
    """Fit the legacy log10(value + 1) transform using training data only."""

    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("values and valid must have shape (sample, 2, y, x)")
    if np.any(values[valid] < 0):
        raise ValueError(
            "log10 channel transform received negative values; "
            "clip them at the ingestion boundary"
        )

    scale = np.empty(2, dtype=np.float32)
    center = np.empty(2, dtype=np.float32)
    spread = np.empty(2, dtype=np.float32)
    for channel in range(2):
        selected = values[:, channel][valid[:, channel]]
        if selected.size == 0:
            raise ValueError(f"channel {channel} has no valid training values")
        scale[channel] = 1.0
        transformed = np.log10(selected + 1.0)
        center[channel] = np.mean(transformed)
        spread[channel] = np.std(transformed)
        if not np.isfinite(spread[channel]) or spread[channel] <= 0:
            raise ValueError(f"channel {channel} has zero transformed spread")
    return ChannelTransform(
        scale, center, spread, kind=ChannelTransform.LOG10_KIND
    )


def fit_precision_weighting(
    product_data,
    transform,
    variance_floor=1e-3,
    maximum_relative_precision=100.0,
    maximum_scale_samples=2_000_000,
):
    """Fit one transformed-space precision scale per channel on training data.

    The scale is the median valid precision from a deterministic frame sample.
    It is fitted once, rather than independently for every image, so absolute
    uncertainty differences between training frames remain in the loss.
    """

    values = np.asarray(product_data["values"], dtype=np.float32)
    standard_deviation = np.asarray(
        product_data["standard_deviation"], dtype=np.float32
    )
    valid = np.asarray(product_data["valid"], dtype=bool)
    quality = np.asarray(product_data["quality"], dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("training values must have shape (sample, 2, y, x)")
    for name, array in (
        ("standard_deviation", standard_deviation),
        ("valid", valid),
        ("quality", quality),
    ):
        if array.shape != values.shape:
            raise ValueError(f"{name} shape {array.shape} does not match {values.shape}")

    variance_floor = float(variance_floor)
    maximum_relative_precision = float(maximum_relative_precision)
    maximum_scale_samples = int(maximum_scale_samples)
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be finite and positive")
    if not np.isfinite(maximum_relative_precision) or maximum_relative_precision <= 0:
        raise ValueError("maximum_relative_precision must be finite and positive")
    if maximum_scale_samples <= 0:
        raise ValueError("maximum_scale_samples must be positive")

    possible_values_per_channel = int(values.shape[0] * np.prod(values.shape[2:]))
    frame_stride = max(
        1,
        int(np.ceil(possible_values_per_channel / maximum_scale_samples)),
    )
    selection = slice(None, None, frame_stride)
    sampled_variance = transform.variance(
        values[selection], standard_deviation[selection]
    )
    sampled_valid = valid[selection]
    sampled_quality = quality[selection]

    precision_scale = np.empty(2, dtype=np.float32)
    scale_sample_count = []
    relative_precision_diagnostics = []
    for channel in range(2):
        support = (
            sampled_valid[:, channel]
            & np.isfinite(sampled_quality[:, channel])
            & (sampled_quality[:, channel] > 0)
            & np.isfinite(sampled_variance[:, channel])
            & (sampled_variance[:, channel] >= 0)
        )
        variance = sampled_variance[:, channel][support]
        if variance.size == 0:
            raise ValueError(f"channel {channel} has no valid uncertainty samples")
        precision = 1.0 / np.maximum(variance, variance_floor)
        precision_scale[channel] = np.median(precision)
        if not np.isfinite(precision_scale[channel]) or precision_scale[channel] <= 0:
            raise ValueError(f"channel {channel} has invalid fitted precision scale")
        scale_sample_count.append(int(precision.size))
        relative_precision = precision / precision_scale[channel]
        clipped_precision = np.minimum(
            relative_precision, maximum_relative_precision
        )
        relative_precision_diagnostics.append({
            "median": float(np.median(relative_precision)),
            "mean_before_cap": float(np.mean(relative_precision)),
            "mean_after_cap": float(np.mean(clipped_precision)),
            "fraction_capped": float(np.mean(
                relative_precision > maximum_relative_precision
            )),
        })

    return {
        "kind": "training_median_precision_v1",
        "measurement_uncertainty_in_loss": True,
        "precision_scale": precision_scale.tolist(),
        "variance_floor": variance_floor,
        "maximum_relative_precision": maximum_relative_precision,
        "maximum_scale_samples": maximum_scale_samples,
        "frame_stride": frame_stride,
        "scale_sample_count": scale_sample_count,
        "relative_precision_diagnostics": relative_precision_diagnostics,
        "loss_reduction": "global_effective_weight_sum_times_state_size",
    }

