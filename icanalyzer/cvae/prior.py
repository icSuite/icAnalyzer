"""Learned latent priors for conditional variational autoencoders."""

import numpy as np
import torch
from torch import nn


class ConditionalGaussianMixturePrior(nn.Module):
    """Kp-conditioned diagonal mixture over one complete joint latent row."""

    def __init__(
        self, latent_dim, components=6, hidden_dim=64,
        kp_center=3.0, kp_scale=2.0,
        minimum_logvar=-8.0, maximum_logvar=4.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.components = int(components)
        self.hidden_dim = int(hidden_dim)
        self.kp_center = float(kp_center)
        self.kp_scale = float(kp_scale)
        self.minimum_logvar = float(minimum_logvar)
        self.maximum_logvar = float(maximum_logvar)
        if (
            self.latent_dim < 1 or self.components < 1 or self.hidden_dim < 1
            or not np.isfinite((self.kp_center, self.kp_scale)).all()
            or self.kp_scale <= 0
            or not self.minimum_logvar < self.maximum_logvar
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
        nn.init.normal_(self.mean_head.weight, mean=0, std=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.normal_(self.logvar_head.weight, mean=0, std=0.01)
        nn.init.zeros_(self.logvar_head.bias)

    def parameters_at(self, kp):
        """Return logits, means and log variances for one Kp per row."""

        reference = next(self.parameters())
        kp = torch.as_tensor(kp, dtype=reference.dtype, device=reference.device)
        if kp.ndim == 0:
            kp = kp[None]
        if (
            kp.ndim != 1 or not torch.isfinite(kp).all()
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
        return self.logit_head(hidden), means, logvars

    def log_prob(self, latent, kp):
        """Evaluate the standardized joint latent log density."""

        latent = torch.as_tensor(latent)
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("latent values do not match prior dimension")
        logits, means, logvars = self.parameters_at(kp)
        if len(logits) != len(latent):
            raise ValueError("Kp and latent batches must align")
        residual = latent[:, None] - means
        component_log_prob = -0.5 * (
            np.log(2 * np.pi) + logvars + residual.square() * torch.exp(-logvars)
        ).sum(dim=-1)
        return torch.logsumexp(
            torch.log_softmax(logits, dim=-1) + component_log_prob,
            dim=-1,
        )

    def sample(self, kp, generator=None):
        """Draw one standardized complete latent row for every supplied Kp."""

        logits, means, logvars = self.parameters_at(kp)
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
