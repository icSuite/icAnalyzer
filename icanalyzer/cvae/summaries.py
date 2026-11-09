"""Coherent distribution summaries for WIC spatial ensembles."""

#%% Imports and morphology contract

import numpy as np


WIC_MORPHOLOGY_FEATURE_NAMES = (
    "log_total_wic",
    "brightness_centroid_mlat",
    "brightness_width_mlat",
    "exceedance_area_fraction",
    "night_brightness_fraction",
    "dawn_brightness_fraction",
    "day_brightness_fraction",
    "dusk_brightness_fraction",
)


def _morphology_features(samples, support, mlat, mlt, threshold):
    """Reduce coherent WIC fields to brightness and oval-shape descriptors."""

    values = np.where(support[None], samples, 0.0).astype(np.float64)
    total = values.sum(axis=(1, 2))
    denominator = np.maximum(total, np.finfo(float).tiny)
    centroid = np.sum(values * mlat[None], axis=(1, 2)) / denominator
    width = np.sqrt(
        np.sum(
            values * (mlat[None] - centroid[:, None, None])**2,
            axis=(1, 2)
        ) / denominator
    )
    area_fraction = np.sum(
        (samples > threshold) & support[None], axis=(1, 2)
    ) / support.sum()

    sectors = (
        (mlt >= 21) | (mlt < 3),
        (mlt >= 3) & (mlt < 9),
        (mlt >= 9) & (mlt < 15),
        (mlt >= 15) & (mlt < 21),
    )
    fractions = [
        np.sum(values * sector[None], axis=(1, 2)) / denominator
        for sector in sectors
    ]
    features = np.column_stack((
        np.log10(1 + total), centroid, width, area_fraction, *fractions
    ))
    if not np.isfinite(features).all():
        raise ValueError("WIC morphology features must be finite")
    return features


def _summarize_batches(
    sample_count, get_batch, support, mlat, mlt, threshold, batch_size
):
    support = np.asarray(support, dtype=bool)
    mlat = np.asarray(mlat, dtype=float)
    mlt = np.asarray(mlt, dtype=float)
    if (
        sample_count < 2 or batch_size < 1
        or support.ndim != 2 or not support.any()
        or mlat.shape != support.shape or mlt.shape != support.shape
        or not np.isfinite(threshold) or threshold < 0
    ):
        raise ValueError("WIC summary inputs are invalid")

    exceedance_count = np.zeros(support.shape, dtype=np.int64)
    feature_chunks = []
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        samples = np.asarray(get_batch(start, stop), dtype=np.float32)
        if samples.shape != (stop - start, *support.shape):
            raise ValueError("WIC ensemble batches must match the summary grid")
        finite_values = samples[:, support]
        roundoff_tolerance = (
            16 * np.finfo(np.float32).eps
            * max(1.0, float(np.max(np.abs(finite_values))))
        )
        if (
            not np.isfinite(finite_values).all()
            or float(np.min(finite_values)) < -roundoff_tolerance
        ):
            raise ValueError("WIC ensemble batches must be finite and nonnegative")
        if np.any(samples < 0):
            samples = np.maximum(samples, 0)
        exceedance_count += np.sum(samples > threshold, axis=0)
        feature_chunks.append(
            _morphology_features(samples, support, mlat, mlt, threshold)
        )

    features = np.concatenate(feature_chunks)
    scale = np.quantile(features, .75, axis=0) - np.quantile(
        features, .25, axis=0
    )
    standard_deviation = features.std(axis=0)
    scale = np.where(scale > 0, scale, standard_deviation)
    scale = np.where(scale > 0, scale, 1.0)
    standardized = features / scale
    center = standardized.mean(axis=0)
    distance = np.sum((standardized - center)**2, axis=1)
    medoid_index = int(np.argmin(distance))
    medoid = np.asarray(
        get_batch(medoid_index, medoid_index + 1)[0], dtype=np.float32
    ).copy()
    probability = exceedance_count.astype(np.float64) / sample_count
    probability[~support] = np.nan

    return {
        "medoid": medoid,
        "medoid_index": medoid_index,
        "medoid_feature_names": WIC_MORPHOLOGY_FEATURE_NAMES,
        "medoid_feature_values": features[medoid_index].astype(np.float32),
        "medoid_method": (
            "actual ensemble member minimizing total pairwise squared "
            "distance in robust-scaled morphology space"
        ),
        "exceedance_probability": probability.astype(np.float32),
        "exceedance_threshold": float(threshold),
    }


#%% Public raw-sample and saved-factor paths

def summarize_wic_samples(
    samples, support, mlat, mlt, threshold=100.0, batch_size=2048
):
    """Summarize an in-memory WIC ensemble without breaking field coherence."""

    samples = np.asarray(samples)
    if samples.ndim != 3:
        raise ValueError("WIC samples must have shape (sample, y, x)")
    return _summarize_batches(
        len(samples), lambda start, stop: samples[start:stop],
        support, mlat, mlt, threshold, batch_size
    )


def summarize_wic_anomaly_factor(
    mean, anomaly_factor, support, mlat, mlt,
    threshold=100.0, batch_size=2048
):
    """Reconstruct coherent batches from saved mean/anomaly representation."""

    mean = np.asarray(mean, dtype=np.float64)
    anomaly_factor = np.asarray(anomaly_factor)
    if (
        mean.ndim != 2 or anomaly_factor.ndim != 2
        or anomaly_factor.shape[0] != mean.size
        or anomaly_factor.shape[1] < 2
    ):
        raise ValueError("saved WIC mean and anomaly factor do not align")
    sample_count = anomaly_factor.shape[1]
    normalization = np.sqrt(sample_count - 1)

    def reconstruct(start, stop):
        flattened = (
            mean.reshape(-1, 1)
            + anomaly_factor[:, start:stop] * normalization
        )
        return flattened.T.reshape(stop - start, *mean.shape)

    return _summarize_batches(
        sample_count, reconstruct, support, mlat, mlt, threshold, batch_size
    )
