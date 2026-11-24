import numpy as np
import torch

from icanalyzer.cvae.prior import (
    ConditionalGaussianMixturePrior,
    aggregate_posterior_standardization,
)
from scripts.CVAE.evaluate_posthoc_conditional_prior import (
    field_metrics,
    rows_near_kp,
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


def test_rows_near_kp_restricts_the_saved_validation_population():
    bank = {"kp": np.array([0, 1.667, 2, 2.333, 4, 6], dtype=np.float32)}

    rows = rows_near_kp(bank, [0, 1, 2, 4, 5], target=2, tolerance=0.34)

    np.testing.assert_array_equal(rows, [1, 2])


def test_field_metrics_preserve_intensity_and_grid_coordinates():
    fields = np.zeros((1, 2, 2, 2), dtype=np.float32)
    fields[0, 0, 1, 0] = 4
    fields[0, 1, 0, 1] = 2
    support = np.ones((2, 2, 2), dtype=bool)
    mlat = np.array([[60, 60], [70, 70]], dtype=float)
    mlt = np.array([[0, 6], [12, 18]], dtype=float)

    metrics = field_metrics(fields, support, mlat, mlt, thresholds=(1, 1))

    np.testing.assert_allclose(metrics["wic_corrected"]["maximum"], [4])
    np.testing.assert_allclose(metrics["wic_corrected"]["supported_mean"], [1])
    np.testing.assert_allclose(metrics["wic_corrected"]["active_fraction"], [0.25])
    np.testing.assert_allclose(metrics["wic_corrected"]["weighted_mlat"], [70])
    np.testing.assert_allclose(metrics["wic_corrected"]["mlat_width"], [0])
    np.testing.assert_allclose(metrics["wic_corrected"]["weighted_mlt"], [12])
    np.testing.assert_allclose(metrics["si12"]["weighted_mlat"], [60])
    np.testing.assert_allclose(metrics["si12"]["weighted_mlt"], [6])
