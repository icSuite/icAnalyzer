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
    "conductance_from_corrected_wic",
    "ensemble_statistics",
    "fit_channel_transform",
    "load_good_orbit_intervals",
    "load_precipitation_cs",
    "load_precipitation_cs_files",
    "select_annotated_orbit_files",
    "sample_decoded_ensemble",
    "sparse_ensemble_reconstruction",
    "split_orbit_files",
    "vae_loss",
]
