"""Model1 whole-frame acceptance CNN, training helpers, and routing rules."""

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.automatic_masking.model2_unet import convolution_block, sample_weights
from scripts.automatic_masking.prepare_model1_training_data import CORE, physical_transform, write_json


def acceptance_labels(labels, targets, domains):
    """Manual Partial is Accepted when every observed WIC pixel is accepted."""
    labels = np.asarray(labels)
    observed = np.asarray(domains, dtype=bool)
    keeps_all_observed = observed.any(axis=(1, 2)) & np.all(~observed | targets, axis=(1, 2))
    effective = labels.copy()
    effective[(labels == "partial") & keeps_all_observed] = "full"
    return effective


class FullFrameClassifier(nn.Module):
    """Small encoder; 2x2 pooled tiles preserve coarse day/night structure."""
    def __init__(self, in_channels=8):
        super().__init__()
        self.first = convolution_block(in_channels, 8)
        self.second = convolution_block(8, 16)
        self.bottom = convolution_block(16, 32)
        self.output = nn.Linear(32 * 4, 1)

    def forward(self, values):
        first = self.first(values)
        second = self.second(F.max_pool2d(first, 2))
        bottom = self.bottom(F.max_pool2d(second, 2))
        tiles = F.adaptive_avg_pool2d(bottom, (2, 2)).flatten(1)
        return self.output(tiles)[:, 0]


class FrameDataset(Dataset):
    """Single-frame acceptance inputs from the frozen reviewed population."""

    def __init__(self, inputs, channels, indices, labels, weights):
        self.inputs = inputs
        self.channels = channels
        self.indices = indices
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        values = self.inputs[index, self.channels].copy()
        target = self.labels[index] == "full"
        return torch.from_numpy(values), torch.tensor(target, dtype=torch.float32), torch.tensor(self.weights[position], dtype=torch.float32)


def binary_weights(rows, indices):
    weights = sample_weights(rows, indices)
    full = np.array([rows[index]["label"] == "full" for index in indices])
    weights *= np.where(full, 1.5, .75).astype(np.float32)
    return weights / weights.mean()


def channels_for(parent_fields, normalization):
    """Return value channels followed by their matching finite flags."""

    fields = normalization["fields"]
    indices = [
        index
        for parent in parent_fields
        for index, candidate in enumerate(normalization.get("parent_fields", fields))
        if candidate == parent
    ]
    return indices + [index + len(fields) for index in indices]


def normalized_core(images, normalization):
    """Apply the frozen Model2 physical transforms and finite flags."""

    channels = []
    for position, field in enumerate(CORE):
        transformed, _, _ = physical_transform(images[:, position], field)
        scaled = (transformed - normalization["centre"][position]) / normalization["scale"][position]
        channels.append(np.nan_to_num(np.clip(scaled, -6, 6), nan=0).astype(np.float32))
    finite = np.isfinite(images).astype(np.float32)
    return np.concatenate((np.stack(channels, axis=1), finite), axis=1)


def classifier_scores(model, inputs, channels, indices, device="cpu"):
    result = np.empty(len(indices), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            batch = indices[start:start + 128]
            values = np.asarray(inputs[batch[:, None], channels])
            result[start:start + len(batch)] = model(torch.from_numpy(values).to(device)).sigmoid().cpu().numpy()
    return result


def train_acceptance(seed, output, inputs, channels, labels, rows, split, normalization, device="cpu"):
    """Fit the single-frame acceptance CNN and save its best validation state."""

    directory = output / f"seed{seed}"
    directory.mkdir()
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_indices, validation = split["train"], split["validation"]
    train_weights = binary_weights(rows, train_indices)
    validation_weights = binary_weights(rows, validation)
    loader = DataLoader(
        FrameDataset(inputs, channels, train_indices, labels, train_weights),
        batch_size=64, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed)
    )
    model = FullFrameClassifier(len(channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    started = time.monotonic()
    for epoch in range(60):
        model.train()
        total = 0.
        for values, target, weight in loader:
            optimizer.zero_grad()
            logits = model(values.to(device))
            loss = (F.binary_cross_entropy_with_logits(logits, target.to(device), reduction="none") * weight.to(device)).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite acceptance-classifier loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(values)
        probability = np.clip(classifier_scores(model, inputs, channels, validation, device), 1e-6, 1 - 1e-6)
        target = (labels[validation] == "full").astype(np.float32)
        validation_loss = float(np.average(-target * np.log(probability) - (1 - target) * np.log1p(-probability), weights=validation_weights))
        history.append(dict(epoch=epoch + 1, training_loss=total / len(train_indices), validation_loss=validation_loss))
        if validation_loss < best_loss - .0001:
            best_loss, best_epoch = validation_loss, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        write_json(directory / "history.json", history)
        print(f"accept seed{seed} epoch{epoch + 1}: train{history[-1]['training_loss']:.4f}, val{validation_loss:.4f}, best{best_epoch}", flush=True)
        if epoch + 1 - best_epoch >= 10:
            break
    model.load_state_dict(best_state)
    record = dict(seed=seed, best_epoch=best_epoch, epochs=len(history), validation_loss=best_loss, parameters=sum(parameter.numel() for parameter in model.parameters()), seconds=time.monotonic() - started)
    state = {name: value.cpu() for name, value in best_state.items()}
    torch.save(dict(state_dict=state, input_fields=CORE, normalization=normalization, architecture="8/16/32-channel encoder,2x2 tile pooling,binary linear output", summary=record), directory / "frame_classifier.pt")
    write_json(directory / "summary.json", record)
    return model, record


def apply_frame_gate(current, scores, threshold):
    passed = np.zeros(len(scores), dtype=bool) if threshold is None else scores >= threshold
    gated = current.copy()
    gated[passed] = True
    assert not np.any(current & ~gated)
    assert np.array_equal(gated[~passed], current[~passed])
    return gated, passed


def route_two_stage(current, accept_scores, accept_cutoff, observed):
    """Model1 releases nonempty accepted frames; Model2 remains the fallback."""
    if not np.isfinite(accept_scores).all():
        raise ValueError("Nonfinite Model1 acceptance score")
    released = (accept_scores >= accept_cutoff) & observed.any(axis=(1, 2))
    combined = current.copy()
    combined[released] = True
    return combined, released


def selection_curve(scores, labels, orbits, good_gain, target_precision=.99, min_frames=50, min_orbits=3):
    """Keep score ties together; select only validation-supported useful gates."""
    scores = np.asarray(scores)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Classifier scores must be finite in [0,1]")
    if not np.isfinite(good_gain).all() or np.any(np.asarray(good_gain) < 0):
        raise ValueError("Recovery gain must be finite and nonnegative")
    if not len(scores):
        return [], None
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0) + 1, len(scores)]
    true_full = np.cumsum(labels[order] == "full")
    gains = np.cumsum(np.asarray(good_gain)[order])
    counts, covered_orbits, curve = 0, set(), []
    for end in ends:
        covered_orbits.update(orbits[order[counts:end]].tolist())
        counts = int(end)
        good = int(true_full[end-1])
        curve.append(dict(threshold=float(ordered_scores[end-1]), accepted_frames=counts, accepted_orbits=len(covered_orbits), true_full=good, false_full=counts-good, precision=good/counts, recall=good/int((labels=="full").sum()) if np.any(labels=="full") else 0., good_gain=float(gains[end-1])))
    eligible = [r for r in curve if r["precision"] >= target_precision and r["accepted_frames"] >= min_frames and r["accepted_orbits"] >= min_orbits and r["good_gain"] > 0]
    chosen = max(eligible, key=lambda r: (r["good_gain"], r["precision"], r["threshold"])) if eligible else None
    return curve, chosen


def frame_metrics(scores, labels, orbits, threshold):
    passed = np.zeros(len(scores), dtype=bool) if threshold is None else scores >= threshold
    full = labels == "full"
    accepted = int(passed.sum())
    return dict(frames=len(scores), passed=accepted, true_full=int((passed & full).sum()), false_full=int((passed & ~full).sum()), precision=float(full[passed].mean()) if accepted else None, recall=float(passed[full].mean()) if full.any() else None, passed_orbits=int(len(np.unique(orbits[passed]))), passed_by_class={label: int((passed & (labels == label)).sum()) for label in ("full","bad","partial")})


def precision_interval(scores, labels, orbits, threshold, seed=20261014, draws=2000):
    """Paired cluster resampling describes inspected orbit variability only."""
    if threshold is None:
        return None
    passed = scores >= threshold
    identifiers = np.unique(orbits)
    count = np.array([passed[orbits == orbit].sum() for orbit in identifiers])
    success = np.array([(passed & (labels == "full"))[orbits == orbit].sum() for orbit in identifiers])
    rng = np.random.default_rng(seed)
    selection = rng.integers(len(identifiers), size=(draws,len(identifiers)))
    total = count[selection].sum(axis=1)
    good = success[selection].sum(axis=1)
    fractions = good[total > 0] / total[total > 0]
    return dict(interval=np.quantile(fractions,[.025,.975]).tolist() if len(fractions) else None, nonempty_draws=len(fractions), draws=draws, interpretation="Whole-orbit resampling of inspected data, not a population99% guarantee; boundary zero-error intervals can be degenerate")
