"""Persistent-homology penalty and discrete shape metrics for full-grid masks."""

#%% Shape prior: at most one foreground region, no enclosed background holes

import numpy as np
import torch
from scipy.ndimage import binary_dilation, label

FOREGROUND = np.ones((3, 3), dtype=bool)
BACKGROUND = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def persistence_pairs(probability):
    """Coface indices follow Fortran ordering; the essential H0 region is allowed."""
    import gudhi
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=1 - probability)
    complex_.compute_persistence(homology_coeff_field=2, min_persistence=0)
    regular, _ = complex_.cofaces_of_persistence_pairs()
    return [regular[dimension] if dimension < len(regular) else np.empty((0, 2), dtype=np.int64) for dimension in (0, 1)]


def topology_loss(logits):
    """Per-image sum of squared finite H0/H1 persistence, with useful gradients.

    PH identifies critical cells on detached probabilities; PyTorch differentiates
    their persistence values. The pairs are piecewise constant. Compute on the
    full mask, never probability * finite-WIC domain: data gaps are not rejection.
    Empty masks are permitted because essential H0 is never forced to persist.
    """
    probability = logits.sigmoid()
    losses = []
    for image in probability:
        pairs = persistence_pairs(image.detach().cpu().numpy())
        flat = image.transpose(0, 1).reshape(-1)
        value = image.sum() * 0
        for indices in pairs:
            if len(indices):
                cells = torch.as_tensor(indices, dtype=torch.long, device=image.device)
                persistence = flat[cells[:, 0]] - flat[cells[:, 1]]
                value = value + persistence.square().sum()
        losses.append(value)
    return torch.stack(losses)


def shape_counts(mask, domain=None):
    """8-connected foreground and enclosed 4-connected background regions.

    Without domain: complete geometric mask. With domain: unknown pixels can
    bridge foreground, and rejected components touching unknown pixels are open.
    Unknown-only foreground components never count as accepted regions.
    """
    mask = np.asarray(mask, dtype=bool)
    if domain is None:
        domain = np.ones_like(mask)
    observed = mask & domain
    unknown = ~domain
    components, _ = label(observed | unknown, structure=FOREGROUND)
    accepted_regions = len(np.unique(components[observed])) if observed.any() else 0
    rejected = domain & ~mask
    components, count = label(rejected, structure=BACKGROUND)
    open_pixels = np.zeros_like(mask)
    open_pixels[[0, -1], :] = True
    open_pixels[:, [0, -1]] = True
    open_pixels |= binary_dilation(unknown, structure=BACKGROUND)
    open_components = set(components[open_pixels & rejected].tolist())
    holes = sum(component not in open_components for component in range(1, count + 1))
    return accepted_regions, holes


def shape_scores(masks, domains=None):
    values = [shape_counts(mask, None if domains is None else domains[i]) for i, mask in enumerate(masks)]
    regions, holes = np.array(values).T
    return dict(accepted_regions=regions, rejected_holes=holes, multiple_regions=regions > 1, has_holes=holes > 0, any_violation=(regions > 1) | (holes > 0))

