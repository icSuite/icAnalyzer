"""Gradient, connectivity and missing-data checks for the WIC topology prior."""

import numpy as np
import pytest
import torch

pytest.importorskip("gudhi")

from scripts.automatic_masking.topology import persistence_pairs, shape_counts, topology_loss
from scripts.automatic_masking.model2_unet import joint_loss, masked_loss


def tensor_logits(values):
    return torch.logit(torch.tensor(values, dtype=torch.float64)).unsqueeze(0).requires_grad_()


@pytest.mark.parametrize("kind", ["empty", "full", "single", "diagonal"])
def test_valid_shapes_have_no_topology_penalty(kind):
    values = np.full((5, 8), .12)
    if kind == "full":
        values[:] = .88
    elif kind == "single":
        values[1:4, 2:5] = .88
    elif kind == "diagonal":
        values[1, 2] = .88
        values[2, 3] = .88
    logits = tensor_logits(values)
    loss = topology_loss(logits)
    assert loss.item() == 0
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_two_regions_have_correct_persistence_and_gradient_indices():
    values = np.full((5, 8), .12)
    values[1, 1] = .93
    values[3, 6] = .84
    values[2, 1:7] = np.linspace(.20, .31, 6)
    logits = tensor_logits(values)
    regions, holes = persistence_pairs(values)
    assert len(regions) == 1 and len(holes) == 0
    flat = values.T.reshape(-1)
    expected = (flat[regions[0, 0]] - flat[regions[0, 1]]) ** 2
    loss = topology_loss(logits).sum()
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    birth, death = regions[0]
    grad = logits.grad[0].transpose(0, 1).reshape(-1)
    assert grad[birth] > 0 and grad[death] < 0
    changed = (logits - .5 * logits.grad).detach()
    assert topology_loss(changed).item() < loss.item()


def test_ring_penalizes_the_hole_with_a_useful_gradient():
    values = np.full((5, 8), .11)
    values[1:4, 2:6] = .91
    values[2, 3:5] = [.22, .31]
    logits = tensor_logits(values)
    _, pairs = persistence_pairs(values)
    assert len(pairs) == 1
    loss = topology_loss(logits).sum()
    assert loss.item() > 0
    loss.backward()
    assert topology_loss((logits - .5 * logits.grad).detach()).item() < loss.item()


def test_geometric_prior_never_turns_a_missing_observation_into_a_rejected_hole():
    values = np.full((5, 8), .88)
    logits = tensor_logits(values)
    target = torch.ones_like(logits)
    domain = torch.ones_like(logits)
    domain[:, 2, 4] = 0
    classes = torch.tensor([0])
    expected = masked_loss(logits, target, domain)
    actual = joint_loss(logits, None, target, domain, classes, topology_weight=1)
    torch.testing.assert_close(actual, expected)
    finite = domain[0].numpy().astype(bool)
    assert shape_counts(np.ones_like(finite), finite) == (1, 0)


def test_observation_gaps_can_bridge_regions_and_are_not_holes():
    values = np.ones((5, 8), dtype=bool)
    domain = np.ones_like(values)
    domain[:, 4] = False
    domain[2, 1] = False
    assert shape_counts(values & domain)[0] == 2
    assert shape_counts(values, domain) == (1, 0)
    assert shape_counts(np.zeros_like(values), domain) == (0, 0)


def test_full_domain_binary_topology_matches_persistence_for_asymmetric_patterns():
    rng = np.random.default_rng(12026)
    for _ in range(30):
        mask = rng.random((5, 8)) > .55
        regions, holes = shape_counts(mask)
        pairs = persistence_pairs(mask.astype(float))
        assert len(pairs[0]) == max(0, regions - 1)
        assert len(pairs[1]) == holes

