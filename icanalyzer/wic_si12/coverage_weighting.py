"""Smooth training coverage before taking capped or full inverse weights."""

import numpy as np
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
