"""Check the interactive new2 decoder without importing or running its workflow."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import BSpline
import torch
from torch import nn
from torch.nn import functional as F


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/IMAGE/step_1_vae_wic_s12_new2.py"


def script_namespace():
    module = ast.parse(SCRIPT.read_text())
    names = {"PartialConv2d", "SplineVAE2D", "Spline", "probe_latent_space", "loss_function"}
    nodes = [node for node in module.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = {"np": np, "torch": torch, "nn": nn, "F": F, "BSpline": BSpline,
                 "CONFIG": {"IMG_HEIGHT": 46, "IMG_WIDTH": 46, "DEVICE": torch.device("cpu"), "USE_MEASUREMENT_UNCERTAINTY": False},
                 "H_mu": .3, "P_mu": .1, "H_std": .8, "P_std": .4}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return module, namespace


@pytest.mark.parametrize("uncertainty", [False, True])
def test_original_support_uses_training_subset_before_artificial_padding(uncertainty):
    module, namespace = script_namespace()
    namespace["CONFIG"]["USE_MEASUREMENT_UNCERTAINTY"] = uncertainty
    for name in ("H", "P", "W", "dH", "dP"):
        namespace[name] = np.ones((10001, 2, 2))
    for name in ("H", "P", "W"):
        namespace[name][:10000, 0, 0] = np.nan
    namespace["dH"][:10000, 1, 1] = np.nan
    main = next(node for node in module.body if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test))
    start = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "subset_size")
    stop = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "H" and "maximum" in ast.unparse(node.value))
    exec(compile(ast.Module(body=main.body[start:stop], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    support = namespace["spatial_support"].copy()
    assert not support[0, 0]  # A later, excluded frame cannot activate this pixel.
    assert support[1, 1] == (not uncertainty)
    for name in ("H", "P", "W"):
        namespace[name][:, 0, 0] = 0.5  # Subsequent artificial filling cannot change support.
    np.testing.assert_array_equal(namespace["spatial_support"], support)


def decoder_namespace():
    _, namespace = script_namespace()
    xi, eta = np.meshgrid(np.linspace(-.7, .7, 46), np.linspace(-.7, .7, 46))
    spline = namespace["Spline"](SimpleNamespace(xi=xi, eta=eta), 40, 3)
    support = np.ones((46, 46), dtype=bool)
    for row in (slice(0, 4), slice(-4, None)):
        for column in (slice(0, 4), slice(-4, None)):
            support[row, column] = False
    namespace["G"] = spline.G
    namespace["ncp"] = spline.ncp
    namespace["active_coefficients"] = np.any(spline.G[support.ravel()] > 0, axis=0)
    assert namespace["active_coefficients"].any() and (~namespace["active_coefficients"]).any()
    return namespace, support


def test_hard_mask_preserves_supported_predictions_and_zeroes_both_channels():
    namespace, support = decoder_namespace()
    model = namespace["SplineVAE2D"](latent_dim=36).eval()
    baseline = copy.deepcopy(model)
    baseline.coefficient_mask.fill_(1)
    z = torch.randn(3, 36)
    with torch.no_grad():
        reconstruction, c_h, c_p = model.decode(z, torch.device("cpu"))
        original, original_h, original_p = baseline.decode(z, torch.device("cpu"))
    active = namespace["active_coefficients"]
    for masked, unmasked in ((c_h, original_h), (c_p, original_p)):
        assert torch.all(masked[:, ~active] == 0)
        torch.testing.assert_close(masked[:, active], unmasked[:, active], rtol=0, atol=0)
    torch.testing.assert_close(reconstruction[:, :2, support], original[:, :2, support], rtol=0, atol=0)
    assert model.flatten_dim == 64 * 6 * 6
    assert c_h.shape == c_p.shape == (3, 36 * 36)
    assert "coefficient_mask" not in dict(model.named_parameters())


def test_mask_blocks_output_gradients_and_is_in_model_state():
    namespace, _ = decoder_namespace()
    model = namespace["SplineVAE2D"](latent_dim=36).to("cpu")
    captured = []

    def remember_coefficients(module, inputs, output):
        output.retain_grad()
        captured.append(output)

    hook = model.spline_conv.register_forward_hook(remember_coefficients)
    _, c_h, c_p = model.decode(torch.randn(2, 36), torch.device("cpu"))
    (c_h.sum() + c_p.sum()).backward()
    gradient = captured[0].grad.reshape(2, 2, -1)
    active = namespace["active_coefficients"]
    assert torch.all(gradient[:, :, ~active] == 0)
    assert torch.all(gradient[:, :, active] == 1)
    hook.remove()
    restored = namespace["SplineVAE2D"](latent_dim=36)
    restored.coefficient_mask.fill_(1)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.coefficient_mask, model.coefficient_mask)
    assert model.coefficient_mask.device == next(model.parameters()).device


def test_training_step_with_existing_loss_keeps_fixed_coefficients_zero():
    namespace, _ = decoder_namespace()
    model = namespace["SplineVAE2D"](latent_dim=36)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(2, 3, 46, 46)
    x[:, 2] = 1.
    mask = torch.ones(2, 1, 46, 46)
    reconstructed, mu, logvar = model(x, mask)
    loss = namespace["loss_function"](reconstructed, x, mask, mu, logvar, beta=1.6)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        _, c_h, c_p = model.decode(torch.randn(2, 36), torch.device("cpu"))
    inactive = ~namespace["active_coefficients"]
    assert torch.all(c_h[:, inactive] == 0) and torch.all(c_p[:, inactive] == 0)


def test_prior_sampler_returns_fixed_coefficients_with_zero_variance():
    namespace, _ = decoder_namespace()
    model = namespace["SplineVAE2D"](latent_dim=36)
    c_h, c_p = namespace["probe_latent_space"](model, 36, n_samples=10, chunk_size=5)
    inactive = ~namespace["active_coefficients"]
    for coefficients in (c_h, c_p):
        assert coefficients.shape == (10, 36 * 36)
        np.testing.assert_array_equal(coefficients[:, inactive], 0)
        np.testing.assert_array_equal(np.var(coefficients[:, inactive], axis=0), 0)
