import numpy as np
import torch

from icanalyzer.cvae.prior import (
    ConditionalGaussianMixturePrior,
    aggregate_posterior_standardization,
)


def test_single_component_zero_network_matches_standard_normal():
    prior = ConditionalGaussianMixturePrior(
        latent_dim=3, components=1, hidden_dim=4
    )
    for parameter in prior.parameters():
        torch.nn.init.zeros_(parameter)
    latent = torch.tensor([[0.0, 1.0, -2.0], [0.5, 0.0, 0.25]])
    kp = torch.tensor([0.0, 6.0])

    actual = prior.log_prob(latent, kp)
    expected = torch.distributions.Normal(0, 1).log_prob(latent).sum(-1)

    torch.testing.assert_close(actual, expected)


def test_mixture_sampling_returns_complete_aligned_rows():
    prior = ConditionalGaussianMixturePrior(
        latent_dim=5, components=3, hidden_dim=4
    )
    latent, component = prior.sample(
        torch.tensor([0.0, 2.0, 4.0, 6.0]),
        generator=torch.Generator().manual_seed(3),
    )

    assert latent.shape == (4, 5)
    assert component.shape == (4,)
    assert torch.isfinite(latent).all()
    assert torch.all((component >= 0) & (component < 3))


def test_aggregate_posterior_standardization_includes_encoder_variance():
    mu = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    logvar = np.zeros_like(mu)

    center, scale = aggregate_posterior_standardization(
        mu, logvar, rows=[0, 1], chunk_size=1
    )

    np.testing.assert_allclose(center, [1.0, 3.0])
    np.testing.assert_allclose(scale, np.sqrt(2), rtol=1e-6)
