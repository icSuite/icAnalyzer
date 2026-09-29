"""WIC/SI12 VAE, ensemble propagation, and sparse reconstruction."""

from .data import (
    ChannelTransform,
    WICSI12ArrayDataset,
    fit_channel_transform,
    load_good_orbit_intervals,
    load_precipitation_cs,
    load_precipitation_cs_files,
    select_annotated_orbit_files,
    split_orbit_files,
)
from .comparison import (
    exact_time_indices,
    load_paired_files,
    load_paired_orbit,
    pair_orbit_files,
)
from .diagnostics import (
    best_validation_epoch,
    covariance_spectrum,
    joint_correlation_maps,
    leading_covariance_modes,
    maximum_variance_anchor,
    reconstruct_ensemble,
)
from .ensemble import (
    ensemble_statistics,
    sample_decoded_ensemble,
    sparse_ensemble_reconstruction,
)
from .model import WICSI12VAE, vae_loss
from .physics import conductance_from_corrected_wic

__all__ = [
    "ChannelTransform",
    "WICSI12ArrayDataset",
    "WICSI12VAE",
    "best_validation_epoch",
    "conductance_from_corrected_wic",
    "covariance_spectrum",
    "ensemble_statistics",
    "fit_channel_transform",
    "exact_time_indices",
    "load_good_orbit_intervals",
    "load_precipitation_cs",
    "load_precipitation_cs_files",
    "load_paired_files",
    "load_paired_orbit",
    "joint_correlation_maps",
    "leading_covariance_modes",
    "maximum_variance_anchor",
    "pair_orbit_files",
    "reconstruct_ensemble",
    "select_annotated_orbit_files",
    "sample_decoded_ensemble",
    "sparse_ensemble_reconstruction",
    "split_orbit_files",
    "vae_loss",
]
