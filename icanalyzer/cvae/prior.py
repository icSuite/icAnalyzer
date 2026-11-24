"""Learned latent priors for conditional variational autoencoders."""

import numpy as np
import torch
from torch import nn


def disable_optional_native_bmm_override(covariance_rank, device):
    """Use ordinary ATen BMM when an optional Triton override is unsafe."""

    device = torch.device(device)
    backend = getattr(torch.backends, "python_native", None)
    if (
        device.type != "cuda" or int(covariance_rank) == 0
        or backend is None or not hasattr(backend, "disable_operations")
    ):
        return False
    backend.disable_operations("bmm")
    return True


class ConditionalGaussianMixturePrior(nn.Module):
    """Kp-conditioned diagonal or low-rank mixture over a joint latent row."""

    def __init__(
        self, latent_dim, components=6, hidden_dim=64,
        kp_center=3.0, kp_scale=2.0,
        minimum_logvar=-8.0, maximum_logvar=4.0,
        covariance_rank=0, maximum_factor=2.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.components = int(components)
        self.hidden_dim = int(hidden_dim)
        self.kp_center = float(kp_center)
        self.kp_scale = float(kp_scale)
        self.minimum_logvar = float(minimum_logvar)
        self.maximum_logvar = float(maximum_logvar)
        self.covariance_rank = int(covariance_rank)
        self.maximum_factor = float(maximum_factor)
        if (
            self.latent_dim < 1 or self.components < 1 or self.hidden_dim < 1
            or not np.isfinite((self.kp_center, self.kp_scale)).all()
            or self.kp_scale <= 0
            or not self.minimum_logvar < self.maximum_logvar
            or self.covariance_rank < 0
            or self.covariance_rank > self.latent_dim
            or not np.isfinite(self.maximum_factor) or self.maximum_factor <= 0
        ):
            raise ValueError("invalid conditional-prior configuration")

        self.condition_network = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.logit_head = nn.Linear(self.hidden_dim, self.components)
        self.mean_head = nn.Linear(
            self.hidden_dim, self.components * self.latent_dim
        )
        self.logvar_head = nn.Linear(
            self.hidden_dim, self.components * self.latent_dim
        )
        self.factor_head = None
        if self.covariance_rank:
            self.factor_head = nn.Linear(
                self.hidden_dim,
                self.components * self.latent_dim * self.covariance_rank,
            )
        nn.init.normal_(self.mean_head.weight, mean=0, std=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.normal_(self.logvar_head.weight, mean=0, std=0.01)
        nn.init.zeros_(self.logvar_head.bias)
        if self.factor_head is not None:
            nn.init.normal_(self.factor_head.weight, mean=0, std=0.001)
            nn.init.zeros_(self.factor_head.bias)

    def _distribution_parameters_at(self, kp, validate=True):
        """Return all parameters, including optional low-rank loadings."""

        reference = next(self.parameters())
        kp = torch.as_tensor(kp, dtype=reference.dtype, device=reference.device)
        if kp.ndim == 0:
            kp = kp[None]
        if kp.ndim != 1:
            raise ValueError("Kp must be a finite vector within 0--9")
        if validate and (
            not torch.isfinite(kp).all()
            or torch.any((kp < 0) | (kp > 9))
        ):
            raise ValueError("Kp must be a finite vector within 0--9")
        normalized = ((kp - self.kp_center) / self.kp_scale)[:, None]
        hidden = self.condition_network(normalized)
        means = self.mean_head(hidden).reshape(
            len(kp), self.components, self.latent_dim
        )
        logvars = self.logvar_head(hidden).reshape_as(means).clamp(
            self.minimum_logvar, self.maximum_logvar
        )
        factors = None
        if self.factor_head is not None:
            factors = self.factor_head(hidden).reshape(
                len(kp), self.components, self.latent_dim, self.covariance_rank
            )
            factors = self.maximum_factor * torch.tanh(
                factors / self.maximum_factor
            )
        return self.logit_head(hidden), means, logvars, factors

    def parameters_at(self, kp):
        """Return logits, means and diagonal log variances for each row."""

        logits, means, logvars, _factors = self._distribution_parameters_at(kp)
        return logits, means, logvars

    def factors_at(self, kp):
        """Return low-rank covariance loadings, or None for a diagonal prior."""

        return self._distribution_parameters_at(kp)[3]

    def log_prob(self, latent, kp, validate=True):
        """Evaluate the standardized joint latent log density."""

        latent = torch.as_tensor(latent)
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("latent values do not match prior dimension")
        logits, means, logvars, factors = self._distribution_parameters_at(
            kp, validate=validate
        )
        if len(logits) != len(latent):
            raise ValueError("Kp and latent batches must align")
        residual = latent[:, None] - means
        inverse_diagonal = torch.exp(-logvars)
        quadratic = (residual.square() * inverse_diagonal).sum(dim=-1)
        log_determinant = logvars.sum(dim=-1)
        if factors is not None:
            scaled_factors = factors * inverse_diagonal[..., None]
            gram = torch.einsum(
                "bkdr,bkds->bkrs", factors, scaled_factors
            )
            identity = torch.eye(
                self.covariance_rank, dtype=gram.dtype, device=gram.device
            )
            cholesky = torch.linalg.cholesky(gram + identity)
            projected = torch.einsum(
                "bkdr,bkd->bkr", scaled_factors, residual
            )
            solved = torch.cholesky_solve(
                projected[..., None], cholesky
            ).squeeze(-1)
            quadratic = (
                quadratic - (projected * solved).sum(dim=-1)
            ).clamp_min(0)
            log_determinant = log_determinant + 2 * torch.log(
                torch.diagonal(cholesky, dim1=-2, dim2=-1)
            ).sum(dim=-1)
        component_log_prob = -0.5 * (
            self.latent_dim * np.log(2 * np.pi)
            + log_determinant + quadratic
        )
        return torch.logsumexp(
            torch.log_softmax(logits, dim=-1) + component_log_prob,
            dim=-1,
        )

    def sample(self, kp, generator=None):
        """Draw one standardized complete latent row for every supplied Kp."""

        logits, means, logvars, factors = self._distribution_parameters_at(kp)
        probabilities = torch.softmax(logits, dim=-1)
        components = torch.multinomial(
            probabilities, 1, replacement=True, generator=generator
        ).squeeze(1)
        rows = torch.arange(len(components), device=components.device)
        selected_mean = means[rows, components]
        selected_logvar = logvars[rows, components]
        noise = torch.randn(
            selected_mean.shape, dtype=selected_mean.dtype,
            device=selected_mean.device, generator=generator,
        )
        latent = selected_mean + noise * torch.exp(0.5 * selected_logvar)
        if factors is not None:
            selected_factor = factors[rows, components]
            factor_noise = torch.randn(
                (len(selected_mean), self.covariance_rank),
                dtype=selected_mean.dtype, device=selected_mean.device,
                generator=generator,
            )
            latent = latent + torch.einsum(
                "bdr,br->bd", selected_factor, factor_noise
            )
        return latent, components

    def configuration(self):
        return {
            "latent_dim": self.latent_dim,
            "components": self.components,
            "hidden_dim": self.hidden_dim,
            "kp_center": self.kp_center,
            "kp_scale": self.kp_scale,
            "minimum_logvar": self.minimum_logvar,
            "maximum_logvar": self.maximum_logvar,
            "covariance_rank": self.covariance_rank,
            "maximum_factor": self.maximum_factor,
        }

    @classmethod
    def from_configuration(cls, configuration):
        return cls(**configuration)


def aggregate_posterior_standardization(mu, logvar, rows, chunk_size=8192):
    """Return per-dimension mean and standard deviation of a posterior mixture."""

    mu = np.asarray(mu)
    logvar = np.asarray(logvar)
    rows = np.asarray(rows, dtype=np.int64)
    if (
        mu.ndim != 2 or logvar.shape != mu.shape or rows.ndim != 1
        or not len(rows) or np.any((rows < 0) | (rows >= len(mu)))
        or chunk_size < 1
    ):
        raise ValueError("invalid posterior arrays or selected rows")
    total = np.zeros(mu.shape[1], dtype=np.float64)
    total_second = np.zeros_like(total)
    for start in range(0, len(rows), chunk_size):
        selected = rows[start:start + chunk_size]
        selected_mu = mu[selected].astype(np.float64)
        selected_logvar = logvar[selected].astype(np.float64)
        total += selected_mu.sum(axis=0)
        total_second += (
            np.square(selected_mu) + np.exp(selected_logvar)
        ).sum(axis=0)
    center = total / len(rows)
    variance = np.maximum(total_second / len(rows) - np.square(center), 1e-8)
    return center.astype(np.float32), np.sqrt(variance).astype(np.float32)
