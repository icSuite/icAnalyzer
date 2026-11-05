"""Legacy-style joint spline-coefficient PCA from a saved model and latent draws."""

#%% Imports

from pathlib import Path
import tempfile

import numpy as np
import torch

from icanalyzer.wic_si12.data import ChannelTransform
from icanalyzer.wic_si12.model import SplineWICSI12VAE
from icanalyzer.wic_si12.workflow import load_run_artifact
from scripts.IMAGE.vae_pipeline.plot_wic_si12_covariance import decompose_factor


#%% Restore log-count coefficients, then PCA before projecting onto the grid

def coefficient_prior(run_directory, n_modes=50, batch_size=64, device="cpu"):
    if batch_size <= 0:
        raise ValueError("decode batch size must be positive")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run_directory = Path(run_directory).expanduser()
    model, transform, _artifact = load_run_artifact(run_directory / "wic_si12_vae.pt", device=device)
    if not isinstance(model, SplineWICSI12VAE) or transform.kind != ChannelTransform.LOG10_KIND:
        raise ValueError("coefficient reconstruction requires a saved log10 spline VAE")
    with np.load(run_directory / "ensemble_products.npz", allow_pickle=False) as archive:
        latent = archive["latent_samples"]
        state_shape = archive["fuv_mean"].shape
    if latent.ndim != 2 or latent.shape[1] != model.latent_dim or latent.shape[0] < 2:
        raise ValueError("saved latent draws do not match the model or contain fewer than two samples")
    if not np.isfinite(latent).all() or state_shape != (2, *model.image_shape):
        raise ValueError("saved ensemble and model have incompatible shapes or nonfinite latent draws")
    coefficient_shape = (2, model.spline_control_points, model.spline_control_points)
    if not 1 <= n_modes <= min(np.prod(coefficient_shape), len(latent) - 1):
        raise ValueError("requested PCA mode count exceeds the coefficient/ensemble rank limit")
    row_basis = model.row_basis.cpu().numpy().astype(np.float64)
    column_basis = model.column_basis.cpu().numpy().astype(np.float64)
    if not np.allclose(row_basis.sum(axis=1), 1, atol=2e-6) or not np.allclose(column_basis.sum(axis=1), 1, atol=2e-6):
        raise ValueError("spline bases must reproduce constants to restore coefficient units")

    with tempfile.TemporaryDirectory(prefix="wic-si12-coefficient-pca-") as workspace:
        # Only one disk-backed coefficient ensemble; no physical image ensemble.
        factor = np.lib.format.open_memmap(Path(workspace) / "factor.npy", mode="w+", dtype=np.float64, shape=(int(np.prod(coefficient_shape)), len(latent)), fortran_order=True)
        coefficient_mean = np.zeros(coefficient_shape, dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(latent), batch_size):
                stop = min(start + batch_size, len(latent))
                batch = torch.as_tensor(latent[start:stop], dtype=torch.float32, device=device)
                coefficients = model.decode_coefficients(batch).cpu().numpy().astype(np.float64)
                # Unlike the legacy decoder, the saved pipeline outputs standardized
                # coefficients. Undo this BEFORE joint PCA to retain log-count units.
                coefficients *= transform.spread[None, :, None, None]
                coefficients += transform.center[None, :, None, None]
                if not np.isfinite(coefficients).all():
                    raise ValueError("decoded log-count spline coefficients contain nonfinite values")
                coefficient_mean += coefficients.sum(axis=0)
                factor[:, start:stop] = coefficients.reshape(stop - start, -1).T
        coefficient_mean /= len(latent)
        for start in range(0, len(latent), batch_size):
            block = factor[:, start:start + batch_size]
            block -= coefficient_mean.ravel()[:, None]
            block /= np.sqrt(len(latent) - 1)
        factor.flush()
        pca = decompose_factor(factor, coefficient_shape, n_modes=n_modes)
        del factor

    coefficient_patterns = pca["patterns"]
    mean = np.einsum("xi,cij,yj->cyx", column_basis, coefficient_mean, row_basis)
    patterns = np.einsum("xi,mcij,yj->mcyx", column_basis, coefficient_patterns, row_basis)
    # This projected factor implements G Q G.T, with Q truncated in coefficient
    # space. It is NOT a fresh PCA of pixel maps. No clipping occurs before PCA.
    return {
        **pca,
        "mean": mean,
        "patterns": patterns,
        "coefficient_mean": coefficient_mean,
        "coefficient_patterns": coefficient_patterns,
        "row_basis": row_basis,
        "column_basis": column_basis,
        "decode_device": str(device)
    }
