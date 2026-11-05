"""Reusable WIC mask U-Net architectures, transforms, losses, and metrics."""

#%% A small image model; no annotation information enters its input channels

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

CLASSES = ("full", "bad", "partial")


def convolution_block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU(), nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU())


class SmallUNet(nn.Module):
    def __init__(self, input_channels=10, class_head=False, match_joint_initialization=False):
        super().__init__()
        self.first = convolution_block(input_channels, 8)
        self.second = convolution_block(8, 16)
        self.bottom = convolution_block(16, 32)
        self.up_second = convolution_block(48, 16)
        self.up_first = convolution_block(24, 8)
        self.output = nn.Conv2d(8, 1, 1)
        self.classifier = nn.Linear(32, 3) if class_head or match_joint_initialization else None
        if not class_head:
            # Match the joint run's RNG draws without retaining a class output.
            self.classifier = None

    def forward(self, inputs, return_class=False):
        first = self.first(inputs)
        second = self.second(F.max_pool2d(first, 2))
        bottom = self.bottom(F.max_pool2d(second, 2))
        second_up = F.interpolate(bottom, size=second.shape[-2:], mode="bilinear", align_corners=False)
        second_up = self.up_second(torch.cat((second, second_up), dim=1))
        first_up = F.interpolate(second_up, size=first.shape[-2:], mode="bilinear", align_corners=False)
        mask_logits = self.output(self.up_first(torch.cat((first, first_up), dim=1)))[:, 0]
        if return_class:
            class_logits = self.classifier(bottom.mean(dim=(2, 3))) if self.classifier is not None else None
            return mask_logits, class_logits
        return mask_logits


class PartialUNet(SmallUNet):
    """Same U-Net weights, with VAE-style validity-aware encoder convolutions."""

    def __init__(self):
        super().__init__(input_channels=8)
        self.register_buffer("support_kernel", torch.ones(1, 1, 3, 3))

    def partial_convolution(self, values, valid, convolution):
        kernel = self.support_kernel.to(dtype=values.dtype)
        support = F.conv2d(valid, kernel, padding=1)
        footprint = F.conv2d(torch.ones_like(valid), kernel, padding=1)
        updated = (support > 0).to(values.dtype)
        raw = convolution(values * valid)
        bias = convolution.bias.reshape(1, -1, 1, 1)
        normalized = (raw - bias) * (footprint / support.clamp_min(1)) + bias
        return normalized * updated, updated

    def partial_block(self, values, valid, block):
        values, valid = self.partial_convolution(values, valid, block[0])
        values = block[2](block[1](values)) * valid
        values, valid = self.partial_convolution(values, valid, block[3])
        values = block[5](block[4](values)) * valid
        return values, valid

    def forward(self, inputs, return_class=False):
        valid = inputs[:, 4:5]  # finite corrected WIC, never the annotation target
        first, first_valid = self.partial_block(inputs, valid, self.first)
        second, second_valid = self.partial_block(F.max_pool2d(first, 2), F.max_pool2d(first_valid, 2), self.second)
        bottom, _ = self.partial_block(F.max_pool2d(second, 2), F.max_pool2d(second_valid, 2), self.bottom)
        second_up = F.interpolate(bottom, size=second.shape[-2:], mode="bilinear", align_corners=False)
        second_up = self.up_second(torch.cat((second, second_up), dim=1))
        first_up = F.interpolate(second_up, size=first.shape[-2:], mode="bilinear", align_corners=False)
        mask_logits = self.output(self.up_first(torch.cat((first, first_up), dim=1)))[:, 0]
        return (mask_logits, None) if return_class else mask_logits


def widen_for_context(model, window_size):
    """Start with identical center-frame predictions; neighbors learn from zero weights."""
    if window_size == 1:
        return model
    initial = model.first[0]
    state = torch.get_rng_state()
    wider = nn.Conv2d(window_size * (initial.in_channels + 1), initial.out_channels, 3, padding=1)
    torch.set_rng_state(state)
    with torch.no_grad():
        wider.weight.zero_()
        center = (window_size // 2) * (initial.in_channels + 1)
        wider.weight[:, center:center + initial.in_channels] = initial.weight
        wider.bias.copy_(initial.bias)
    model.first[0] = wider
    return model


def window_inputs(inputs, windows, indices, lag_seconds=None):
    channels, height, width = inputs.shape[1:]
    stride = channels + int(lag_seconds is not None)
    result = np.zeros((len(indices), windows.shape[1] * stride, height, width), dtype=np.float32)
    for position in range(windows.shape[1]):
        neighbors = windows[indices, position]
        available = neighbors >= 0
        start = position * stride
        result[available, start:start + channels] = inputs[neighbors[available]]
        if lag_seconds is not None:
            result[available, start + channels] = lag_seconds[indices[available], position, None, None] / 300
    return result


class FrameWindowDataset(Dataset):
    """Build small input batches without duplicating five whole image corpora."""
    def __init__(self, inputs, windows, indices, targets, domains, classes, weights, lag_seconds=None):
        self.inputs, self.windows, self.indices = inputs, windows, indices
        self.targets, self.domains, self.classes = targets, domains, classes
        self.weights, self.lag_seconds = weights, lag_seconds

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local):
        index = self.indices[local]
        values = window_inputs(self.inputs, self.windows, np.array([index]), self.lag_seconds)[0]
        return torch.from_numpy(values), torch.from_numpy(self.targets[index].astype(np.float32)), torch.from_numpy(self.domains[index].astype(np.float32)), int(self.classes[index]), torch.tensor(self.weights[local], dtype=torch.float32)


def prepare_inputs(images, train, fields):
    values = images.copy()
    transforms = []
    for channel, field in enumerate(fields):
        transform = field
        if field == "wic_corrected":
            values[:, channel] = np.arcsinh(values[:, channel] / 100)
            transform = "asinh(WIC/100)"
        elif field == "dwic_corrected":
            values[:, channel] = np.log1p(np.maximum(values[:, channel], 0) / 10)
            transform = "log1p(std/10)"
        elif field in ("sza", "dza"):
            divisor = 180 if field == "sza" else 90
            values[:, channel] /= divisor
            transform = f"{field}/{divisor}"
        transforms.append(transform)
    observed = np.isfinite(values)
    rng = np.random.default_rng(12026)
    sample = rng.choice(train, min(512, len(train)), replace=False)
    centre, scale = [], []
    for channel in range(len(fields)):
        pixels = values[sample, channel].ravel()
        pixels = pixels[np.isfinite(pixels)]
        low, median, high = np.percentile(pixels, [10, 50, 90])
        centre.append(float(median))
        scale.append(float(max(high - low, 0.05)))
        values[:, channel] = (values[:, channel] - median) / scale[-1]
    values = np.nan_to_num(np.clip(values, -6, 6), nan=0)
    return np.concatenate((values, observed.astype(np.float32)), axis=1), dict(fields=fields, centre=centre, scale=scale, fit_sample_indices=sample.tolist(), transforms=transforms)


def sample_weights(rows, indices):
    segments = {}
    orbit_segments = {}
    class_orbits = {}
    for index in indices:
        row = rows[index]
        label = row.get("label", "partial")
        segments[row["segment"]] = segments.get(row["segment"], 0) + 1
        orbit_segments.setdefault((label, row["orbit"]), set()).add(row["segment"])
        class_orbits.setdefault(label, set()).add(row["orbit"])
    weights = []
    for index in indices:
        row = rows[index]
        label = row.get("label", "partial")
        weight = 1 / (segments[row["segment"]] * len(orbit_segments[label, row["orbit"]]) * len(class_orbits[label]))
        weights.append(weight)
    weights = np.asarray(weights, dtype=np.float32)
    return weights / weights.mean()


def masked_loss(logits, target, domain):
    area = domain.sum(dim=(1, 2)).clamp_min(1)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * domain).sum(dim=(1, 2)) / area
    probability = logits.sigmoid() * domain
    kept = target * domain
    dice_loss = 1 - (2 * (probability * kept).sum(dim=(1, 2)) + 1) / (probability.sum(dim=(1, 2)) + kept.sum(dim=(1, 2)) + 1)
    return bce + dice_loss


def joint_loss(mask_logits, class_logits, target, domain, classes, class_weight=1, topology_weight=0):
    loss = masked_loss(mask_logits, target, domain)
    if class_logits is not None:
        loss = loss + class_weight * F.cross_entropy(class_logits, classes, reduction="none")
    if topology_weight > 0:
        from scripts.automatic_masking.topology import topology_loss
        loss = loss + topology_weight * topology_loss(mask_logits)
    return loss


def predict(model, inputs, indices, device, windows=None, lag_seconds=None):
    predictions, class_predictions = [], []
    if windows is None:
        windows = np.arange(len(inputs))[:, None]
    model.eval()
    with torch.no_grad():
        for batch in np.array_split(indices, max(1, int(np.ceil(len(indices) / 128)))):
            values = window_inputs(inputs, windows, batch, lag_seconds)
            mask_logits, class_logits = model(torch.from_numpy(values).to(device), return_class=True)
            predictions.append(mask_logits.sigmoid().cpu().numpy())
            if class_logits is not None:
                class_predictions.append(class_logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(class_predictions) if class_predictions else None


def pixel_scores(prediction, target, domain):
    tp = (prediction & target & domain).sum(axis=(1, 2))
    fn = (~prediction & target & domain).sum(axis=(1, 2))
    fp = (prediction & ~target & domain).sum(axis=(1, 2))
    tn = (~prediction & ~target & domain).sum(axis=(1, 2))
    divide = lambda a, b: np.divide(a, b, out=np.full(len(a), np.nan, dtype=float), where=b > 0)
    dice, iou = divide(2 * tp, 2 * tp + fp + fn), divide(tp, tp + fp + fn)
    dice[(tp + fp + fn) == 0] = 1
    iou[(tp + fp + fn) == 0] = 1
    return dict(dice=dice, iou=iou, useful_discarded=divide(fn, tp + fn), rejected_retained=divide(fp, fp + tn), correction_fraction=divide(fp + fn, tp + fn + fp + tn), tp=tp, fn=fn, fp=fp, tn=tn)


def summarize(scores, orbits):
    result = {}
    for name, values in scores.items():
        if name in ("tp", "fn", "fp", "tn"):
            continue
        if not np.isfinite(values).any():
            result[name] = dict(frame_mean=None, orbit_mean=None, frame_median=None)
            continue
        means = [np.nanmean(values[orbits == orbit]) for orbit in np.unique(orbits) if np.isfinite(values[orbits == orbit]).any()]
        result[name] = dict(frame_mean=float(np.nanmean(values)), orbit_mean=float(np.mean(means)), frame_median=float(np.nanmedian(values)))
    return result


def class_scores(scores, orbits, labels):
    return {label: dict(frames=int((labels == label).sum()), orbits=int(len(np.unique(orbits[labels == label]))), metrics=summarize({key: value[labels == label] for key, value in scores.items()}, orbits[labels == label])) for label in np.unique(labels)}


def validation_correction(scores, orbits, labels):
    by_class = class_scores(scores, orbits, labels)
    return float(np.mean([result["metrics"]["correction_fraction"]["orbit_mean"] for result in by_class.values()]))
