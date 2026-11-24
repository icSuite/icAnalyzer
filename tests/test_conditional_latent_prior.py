import numpy as np
import torch
from torch import nn

from icanalyzer.cvae.loss import (
    LEGACY_LOSS_CONFIGURATION,
    conditional_monte_carlo_kl,
)
from icanalyzer.cvae.prior import (
    ConditionalGaussianMixturePrior,
    aggregate_posterior_standardization,
    disable_optional_native_bmm_override,
)
from icanalyzer.cvae.training import _run_epoch
from scripts.CVAE.evaluate_posthoc_conditional_prior import (
    evaluation_candidate_labels,
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


def test_conditional_monte_carlo_kl_uses_supplied_latent_and_has_gradients():
    prior = ConditionalGaussianMixturePrior(
        latent_dim=3, components=1, hidden_dim=4, covariance_rank=0
    )
    for parameter in prior.parameters():
        nn.init.zeros_(parameter)
    mu = torch.tensor([[0.2, -0.4, 0.7]], requires_grad=True)
    logvar = torch.tensor([[0.1, -0.2, 0.3]], requires_grad=True)
    noise = torch.tensor([[0.5, -1.0, 0.25]])
    latent = mu + torch.exp(0.5 * logvar) * noise
    kp = torch.tensor([2.0])

    actual = conditional_monte_carlo_kl(
        mu, logvar, latent, prior, kp
    )
    log_q = -0.5 * (
        np.log(2 * np.pi) + logvar
        + (latent - mu).square() * torch.exp(-logvar)
    ).sum()
    log_p = -0.5 * (
        3 * np.log(2 * np.pi) + latent.square().sum()
    )
    torch.testing.assert_close(actual, log_q - log_p)

    actual.backward()
    assert mu.grad.abs().sum() > 0
    assert logvar.grad.abs().sum() > 0
    assert prior.mean_head.bias.grad.abs().sum() > 0


def test_training_updates_joint_prior_and_omits_group_kl_metrics():
    class TinyJointPriorVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_mean = nn.Parameter(torch.tensor([0.4, -0.3]))
            self.encoder_logvar = nn.Parameter(torch.tensor([-0.2, 0.1]))
            self.decoder_scale = nn.Parameter(torch.tensor(0.1))
            self.latent_prior = ConditionalGaussianMixturePrior(
                latent_dim=2, components=2, hidden_dim=4,
                covariance_rank=1,
            )
            self.kp_conditioning = {"center": 3.0, "scale": 2.0}
            self.conditioning = None
            self.latent_groups = {"global": (0, 1), "spatial": (1, 2)}

        def forward(
            self, encoder_input, valid=None, kp=None, return_latent=False
        ):
            batch = len(encoder_input)
            mu = self.encoder_mean[None].expand(batch, -1)
            logvar = self.encoder_logvar[None].expand_as(mu)
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            reconstructed = (
                self.decoder_scale * latent[:, :1, None, None]
            ).expand(batch, 1, 2, 2)
            if return_latent:
                return reconstructed, mu, logvar, latent
            return reconstructed, mu, logvar

    torch.manual_seed(12)
    model = TinyJointPriorVAE()
    before = model.latent_prior.mean_head.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch = {
        "encoder_input": torch.zeros(3, 1, 2, 2),
        "target": torch.zeros(3, 1, 2, 2),
        "mask": torch.ones(3, 1, 2, 2, dtype=torch.bool),
        "weight": torch.ones(3, 1, 2, 2),
        "kp": torch.tensor([0.0, 3.0, 6.0]),
    }

    metrics = _run_epoch(
        model, [batch], optimizer=optimizer, device="cpu", beta=0.1,
        training=True, loss_configuration=LEGACY_LOSS_CONFIGURATION,
    )

    assert set(metrics) == {"total", "reconstruction", "kl"}
    assert not torch.equal(before, model.latent_prior.mean_head.weight)


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


def test_low_rank_log_density_matches_full_covariance_distribution():
    prior = ConditionalGaussianMixturePrior(
        latent_dim=3, components=1, hidden_dim=4,
        covariance_rank=2, maximum_factor=5,
    ).double()
    for parameter in prior.parameters():
        torch.nn.init.zeros_(parameter)
    mean = torch.tensor([0.2, -0.5, 0.7], dtype=torch.float64)
    logvar = torch.log(torch.tensor([0.6, 1.2, 0.8], dtype=torch.float64))
    raw_factor = torch.tensor(
        [[0.30, -0.10], [0.20, 0.25], [-0.15, 0.35]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        prior.mean_head.bias.copy_(mean)
        prior.logvar_head.bias.copy_(logvar)
        prior.factor_head.bias.copy_(raw_factor.ravel())
    latent = torch.tensor(
        [[0.1, -0.2, 0.4], [1.0, -1.0, 0.0]], dtype=torch.float64
    )
    factor = prior.factors_at(torch.tensor([2.0], dtype=torch.float64))[0, 0]
    covariance = torch.diag(torch.exp(logvar)) + factor @ factor.T
    expected = torch.distributions.MultivariateNormal(
        mean, covariance_matrix=covariance
    ).log_prob(latent)

    actual = prior.log_prob(latent, torch.tensor([2.0, 2.0], dtype=torch.float64))

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)


def test_low_rank_sampling_and_configuration_roundtrip():
    prior = ConditionalGaussianMixturePrior(
        latent_dim=5, components=2, hidden_dim=4,
        covariance_rank=3, maximum_factor=1.5,
    )

    restored = ConditionalGaussianMixturePrior.from_configuration(
        prior.configuration()
    )
    latent, component = restored.sample(
        torch.tensor([0.0, 2.0, 4.0, 6.0]),
        generator=torch.Generator().manual_seed(7),
    )

    assert restored.covariance_rank == 3
    assert restored.maximum_factor == 1.5
    assert latent.shape == (4, 5)
    assert component.shape == (4,)
    assert torch.isfinite(latent).all()


def test_low_rank_factor_receives_finite_training_gradient():
    torch.manual_seed(9)
    prior = ConditionalGaussianMixturePrior(
        latent_dim=6, components=2, hidden_dim=5, covariance_rank=2
    )
    latent = torch.randn(12, 6)
    kp = torch.linspace(0, 6, len(latent))

    loss = -prior.log_prob(latent, kp).mean()
    loss.backward()

    gradient = prior.factor_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_legacy_diagonal_configuration_remains_loadable():
    original = ConditionalGaussianMixturePrior(
        latent_dim=4, components=2, hidden_dim=3
    )
    configuration = original.configuration()
    configuration.pop("covariance_rank")
    configuration.pop("maximum_factor")
    restored = ConditionalGaussianMixturePrior.from_configuration(configuration)

    restored.load_state_dict(original.state_dict())

    assert restored.covariance_rank == 0
    assert restored.factor_head is None


def test_low_rank_cuda_fit_disables_optional_native_bmm(monkeypatch):
    calls = []
    backend = type("Backend", (), {
        "disable_operations": lambda self, name: calls.append(name),
    })()
    monkeypatch.setattr(torch.backends, "python_native", backend, raising=False)

    disabled = disable_optional_native_bmm_override(8, torch.device("cuda"))

    assert disabled
    assert calls == ["bmm"]


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


def test_low_rank_evaluation_uses_matched_diagonal_baseline_even_if_it_wins():
    summary = {
        "selected_components": 6,
        "selected_candidate": {
            "label": "K=6", "components": 6, "covariance_rank": 0,
        },
        "candidates": [
            {
                "label": "K=6", "components": 6, "covariance_rank": 0,
                "validation": {"mean_log_density": -10},
            },
            {
                "label": "K=6, rank=8", "components": 6,
                "covariance_rank": 8,
                "validation": {"mean_log_density": -11},
            },
        ],
    }

    baseline, comparison, selected = evaluation_candidate_labels(summary)

    assert baseline == "K=6"
    assert comparison == "K=6, rank=8"
    assert selected == "K=6"


def test_legacy_evaluation_compares_selected_mixture_with_gaussian():
    summary = {
        "selected_components": 6,
        "candidates": [
            {"components": 1, "validation": {"mean_log_density": -12}},
            {"components": 6, "validation": {"mean_log_density": -10}},
        ],
    }

    baseline, comparison, selected = evaluation_candidate_labels(summary)

    assert baseline == "K=1"
    assert comparison == "K=6"
    assert selected == "K=6"


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
