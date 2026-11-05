"""Fit retained observations through a frozen decoder; local latent uncertainty."""

#%% Imports

from pathlib import Path

import numpy as np
from scipy import linalg, optimize
import torch

from icanalyzer.wic_si12.data import ChannelTransform
from icanalyzer.wic_si12.workflow import load_run_artifact


#%% Load only the saved decoder and training transformation, not the ensemble

def load_decoder(run_directory, device="cpu"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform, _artifact = load_run_artifact(Path(run_directory).expanduser() / "wic_si12_vae.pt", device=device)
    if transform.kind != ChannelTransform.LOG10_KIND:
        raise ValueError("decoder reconstruction requires a saved log10 model")
    model.eval().double().requires_grad_(False)
    return model, transform


#%% Nonlinear MAP: observations plus a standard-normal latent prior

def fit_decoder(model, transform, values, retained, observation_std=0.1, prior_weight=1.0, restarts=8, max_iterations=2000, seed=1):
    if transform.kind != ChannelTransform.LOG10_KIND:
        raise ValueError("decoder fitting requires a log10 transformation")
    if not np.isfinite([observation_std, prior_weight]).all() or observation_std <= 0 or prior_weight <= 0:
        raise ValueError("observation STD and latent prior weight must be finite and positive")
    if restarts < 1 or max_iterations < 1:
        raise ValueError("latent restarts and maximum iterations must be positive")
    values = np.asarray(values)
    if values.shape != (2, *model.image_shape):
        raise ValueError("held-out image shape differs from the saved decoder grid")
    indices = np.flatnonzero(np.broadcast_to(retained, values.shape))
    observations = values.ravel()[indices]
    if indices.size == 0 or not np.isfinite(observations).all() or np.any(observations < 0):
        raise ValueError("retained observations must be finite, nonnegative and nonempty")
    device = next(model.parameters()).device
    model.eval().double().requires_grad_(False)
    index_tensor = torch.as_tensor(indices, device=device)
    target = torch.as_tensor(np.log10(observations + 1), dtype=torch.float64, device=device)
    center = torch.as_tensor(transform.center, dtype=torch.float64, device=device)[:, None, None]
    spread = torch.as_tensor(transform.spread, dtype=torch.float64, device=device)[:, None, None]

    def log_field(latent):
        # Undo training standardization without physical clipping before fitting.
        return (model.decode(latent[None])[0] * spread + center).reshape(-1)

    def objective(latent_array):
        latent = torch.tensor(latent_array, dtype=torch.float64, device=device, requires_grad=True)
        residual = (log_field(latent)[index_tensor] - target) / observation_std
        loss = 0.5 * (residual.square().sum() + prior_weight * latent.square().sum())
        gradient = torch.autograd.grad(loss, latent)[0]
        loss_value, gradient_value = loss.item(), gradient.detach().cpu().numpy()
        if not np.isfinite(loss_value) or not np.isfinite(gradient_value).all():
            raise ValueError("decoder fitting produced a nonfinite objective or gradient")
        return loss_value, gradient_value

    # Local RNG; no encoder input, quality map, hidden truth or new ensemble.
    generator = np.random.default_rng(seed)
    starts = generator.normal(size=(restarts, model.latent_dim))
    starts[0] = 0
    fits = []
    for start in starts:
        result = optimize.minimize(objective, start, method="L-BFGS-B", jac=True, options={"maxiter": max_iterations, "ftol": 1e-10, "gtol": 1e-6})
        fits.append(result)
    best_index = int(np.argmin([fit.fun for fit in fits]))
    best = fits[best_index]
    latent = torch.tensor(best.x, dtype=torch.float64, device=device, requires_grad=True)
    fitted_log = log_field(latent).detach().cpu().numpy()

    # Gauss-Newton/Laplace covariance around ONLY the selected latent solution.
    # Forward-mode needs latent_dim derivatives, not one per image pixel.
    jacobian = torch.autograd.functional.jacobian(log_field, latent, vectorize=True, strategy="forward-mode").detach().cpu().numpy()
    observed_jacobian = jacobian[indices] / observation_std
    precision = prior_weight * np.eye(model.latent_dim) + observed_jacobian.T @ observed_jacobian
    cholesky = linalg.cho_factor(precision, lower=True)
    latent_covariance = linalg.cho_solve(cholesky, np.eye(model.latent_dim))
    conditional_factor = linalg.solve_triangular(cholesky[0], jacobian.T, lower=True).T
    log_variance = np.sum(conditional_factor**2, axis=1)
    # Same counts-space calculation as OI, but under a LOCAL linearization.
    reconstruction = np.power(10.0, fitted_log) - 1
    natural_log_variance = np.log(10)**2 * log_variance
    uncertainty = np.sqrt(np.expm1(natural_log_variance)) * np.exp(np.log(10) * fitted_log + 0.5 * natural_log_variance)
    if not np.isfinite(reconstruction).all() or not np.isfinite(uncertainty).all():
        raise ValueError("decoder reconstruction or local uncertainty back-transformation overflowed")
    posterior = {
        "reconstruction": np.maximum(reconstruction, 0).reshape(values.shape),
        "uncertainty": uncertainty.reshape(values.shape),
        "posterior_log_mean": fitted_log.reshape(values.shape),
        "posterior_log_std": np.sqrt(log_variance).reshape(values.shape),
        "latent_map": best.x,
        "latent_covariance": latent_covariance,
        "negative_median_cells_clipped": int(np.count_nonzero(reconstruction < 0))
    }
    diagnostics = {
        "objective": "0.5*sum(((decoder_log10(z)-observed_log10)/observation_std)**2) + 0.5*latent_prior_weight*sum(z**2)",
        "latent_prior_weight": float(prior_weight),
        "seed": int(seed),
        "restarts": int(restarts),
        "max_iterations": int(max_iterations),
        "best_restart": best_index,
        "best_fit_converged": bool(best.success),
        "optimizer_message": str(best.message),
        "restart_objectives": [float(fit.fun) for fit in fits],
        "restart_iterations": [int(fit.nit) for fit in fits],
        "restart_converged": [bool(fit.success) for fit in fits],
        "latent_norm": float(np.linalg.norm(best.x)),
        "observed_log_rmse_dex": float(np.sqrt(np.mean((fitted_log[indices] - target.cpu().numpy())**2))),
        "decode_device": str(device),
        "reconstruction_statistic": "decoder at fitted latent MAP; not full nonlinear posterior mean",
        "uncertainty": "local Gauss-Newton/Laplace latent covariance propagated through decoder Jacobian; shifted-lognormal counts STD; excludes other latent solutions, model discrepancy and calibration"
    }
    return posterior, diagnostics
