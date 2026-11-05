"""Compare encoder reconstruction with target-informed latent and spline fits."""

#%% Imports
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq
import torch
from torch.utils.data import DataLoader, Subset

from icanalyzer.wic_si12.model import SplineWICSI12VAE
from icanalyzer.wic_si12.workflow import load_run_artifact
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import _load_heldout_dataset

METHODS = ["encoder", "optimized_latent", "free_spline"]
LABELS = ["Encoder mean", "Optimized latent (oracle)", "Free spline (oracle)"]


#%% Fixed orbit-balanced frame selection
def select_frames(orbits, count, seed):
    if not 0 < count <= len(orbits):
        raise ValueError("frame count must be positive and not exceed the usable split")
    generator = np.random.default_rng(seed)
    pools = [list(generator.permutation(np.flatnonzero(orbits == orbit))) for orbit in np.unique(orbits)]
    chosen = []
    while len(chosen) < count:
        for pool in generator.permutation(len(pools)):
            if pools[pool]:
                chosen.append(int(pools[pool].pop()))
                if len(chosen) == count:
                    break
    return np.sort(chosen)


def selection_record(dataset, artifact, split, count, seed, saved=None):
    identity = {"orbit": dataset.frame_orbit.tolist(), "time": dataset.frame_time.tolist(),
                "files": artifact["split_files"][split], "transform": artifact["transform"],
                "minimum_sza": artifact["training_configuration"].get("minimum_sza_degrees"),
                "maximum_missing_fraction": dataset.maximum_missing_fraction}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if saved is None:
        indices = select_frames(dataset.frame_orbit, count, seed)
    else:
        if saved["dataset_sha256"] != fingerprint:
            raise ValueError(f"saved selection does not match the reloaded {split} data/transform")
        indices = np.asarray(saved["indices"], dtype=int)
        if len(indices) == 0 or len(np.unique(indices)) != len(indices) or np.any((indices < 0) | (indices >= len(dataset))):
            raise ValueError("saved selection contains invalid or repeated indices")
        if (dataset.frame_orbit[indices].tolist() != saved["orbit"] or dataset.frame_time[indices].tolist() != saved["time"]):
            raise ValueError("saved selection frame identities differ")
    frame_data = hashlib.sha256()
    for array in (dataset.target, dataset.common_valid, dataset.weight):
        frame_data.update(np.ascontiguousarray(array[indices]).tobytes())
    if saved is not None and saved["frame_data_sha256"] != frame_data.hexdigest():
        raise ValueError("saved frame targets, masks or quality weights have changed")
    return {"dataset_sha256": fingerprint, "frame_data_sha256": frame_data.hexdigest(), "indices": indices.tolist(),
            "orbit": dataset.frame_orbit[indices].tolist(), "time": dataset.frame_time[indices].tolist()}


#%% Legacy reconstruction objective and unconstrained spline floor
def reconstruction_terms(prediction, target, mask, weight):
    q = torch.where(mask.bool() & torch.isfinite(weight), weight.clamp(min=0), 0.0)
    active = q > 0
    error = torch.where(active, prediction - target, 0.0)
    squared_error = (q * error.square()).sum(dim=(1, 2, 3))
    return squared_error, q.sum(dim=(1, 2, 3))


def legacy_reconstruction(prediction, target, mask, weight):
    squared_error, quality_mass = reconstruction_terms(prediction, target, mask, weight)
    return squared_error.sum() / (quality_mass.sum() + 1e-8) * np.prod(target.shape[1:])


def fit_free_splines(target, mask, weight, row_basis, column_basis, rcond):
    """Fit both channels with the saved model's exact coefficient-axis order.

    The decoder uses column_basis[x,i] * coefficients[c,i,j] * row_basis[y,j].
    QR least squares supplies a numerical, target-informed approximation floor.
    Rank-deficient/gap regions are not independently constrained by this fit.
    """
    design = np.einsum("xi,yj->yxij", column_basis, row_basis).reshape(np.prod(target.shape[-2:]), -1)
    prediction = np.empty_like(target, dtype=np.float64)
    coefficients, ranks = [], []
    for frame in range(len(target)):
        q = np.where(mask[frame, 0] & np.isfinite(weight[frame, 0]), np.maximum(weight[frame, 0], 0), 0).ravel()
        active = q > 0
        if not np.any(active):
            raise ValueError("free spline fit requires positive common quality support")
        root_weight = np.sqrt(q[active].astype(np.float64))
        rhs = target[frame].reshape(2, -1)[:, active].T * root_weight[:, None]
        fitted, _, rank, _ = lstsq(design[active] * root_weight[:, None], rhs, cond=rcond, lapack_driver="gelsy")
        prediction[frame] = (design @ fitted).T.reshape(target.shape[1:])
        coefficients.append(fitted.T.reshape(2, column_basis.shape[1], row_basis.shape[1]))
        ranks.append(int(rank))
    if not np.all(np.isfinite(prediction)):
        raise ValueError("nonfinite free-spline predictions; inspect weighted design conditioning")
    return prediction, np.asarray(coefficients), ranks


#%% Fit latent vectors only; the encoder/decoder weights and buffers are frozen
def optimize_latents(model, batch, encoded_mu, steps, starts, learning_rate, seed, batch_size, progress=False):
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("latent optimization requires an eval-mode model with frozen parameters")
    device = encoded_mu.device
    frame_count = len(encoded_mu)
    generator = torch.Generator().manual_seed(seed)
    random_starts = torch.randn(starts - 1, frame_count, model.latent_dim, generator=generator).to(device)
    with torch.no_grad():
        best_prediction = model.decode(encoded_mu).detach().clone()
        error, mass = reconstruction_terms(best_prediction, batch["target"], batch["mask"], batch["weight"])
        best_error = error / (mass + 1e-8)
    if torch.any(mass <= 0):
        raise ValueError("latent fitting requires positive common quality support for every frame")
    best_z = encoded_mu.detach().clone()
    winning_start = torch.full((frame_count,), -1, dtype=torch.long, device=device)
    curves = np.empty((starts, steps + 1, frame_count), dtype=np.float64)
    state_size = np.prod(batch["target"].shape[1:])
    for first in range(0, frame_count, batch_size):
        last = min(first + batch_size, frame_count)
        local = {key: value[first:last] for key, value in batch.items()}
        for start in range(starts):
            initial = encoded_mu[first:last] if start == 0 else random_starts[start - 1, first:last]
            z = initial.detach().clone().requires_grad_()
            optimizer = torch.optim.Adam([z], lr=learning_rate)
            if progress:
                print(f"  Frames {first + 1}-{last}, latent start {start + 1}/{starts}, {steps} steps", flush=True)
            for step in range(steps + 1):
                predicted = model.decode(z)
                error, qsum = reconstruction_terms(predicted, local["target"], local["mask"], local["weight"])
                score = error / (qsum + 1e-8)
                if not torch.all(torch.isfinite(score)):
                    raise ValueError("nonfinite latent objective; reduce learning rate or inspect decoder")
                curves[start, step, first:last] = score.detach().cpu().double().numpy() * state_size
                improved = score.detach() < best_error[first:last]
                with torch.no_grad():
                    best_error[first:last][improved] = score.detach()[improved]
                    best_prediction[first:last][improved] = predicted.detach()[improved]
                    best_z[first:last][improved] = z.detach()[improved]
                    winning_start[first:last][improved] = start
                if step != steps:
                    optimizer.zero_grad()
                    loss = error.sum() / (qsum.sum() + 1e-8) * state_size
                    loss.backward()
                    optimizer.step()
    return best_prediction, best_z, winning_start, curves


#%% Matched metrics and readable reconstruction/convergence figures
def reconstruction_metrics(prediction, target, mask, weight):
    valid = np.broadcast_to(mask.astype(bool), target.shape)
    if not np.all(np.isfinite(prediction[valid])):
        raise ValueError("nonfinite prediction on supported cells")
    q = np.where(mask & np.isfinite(weight), np.maximum(weight, 0), 0)
    error = np.where(valid, prediction - target, 0.0)
    weighted = np.where(q > 0, error, 0.0)**2 * q
    per_frame_mse = weighted.sum(axis=(2, 3)) / (q.sum(axis=(2, 3)) + 1e-8)
    quality_rmse = np.sqrt(weighted.sum(axis=(0, 2, 3)) / q.sum())
    all_valid_rmse = np.sqrt((error**2).sum(axis=(0, 2, 3)) / valid.sum(axis=(0, 2, 3)))
    return {"quality_weighted_rmse": quality_rmse.tolist(), "all_valid_rmse": all_valid_rmse.tolist(),
            "per_frame_quality_rmse": np.sqrt(per_frame_mse).tolist(),
            "legacy_reconstruction": float(weighted.sum() / (q.sum() + 1e-8) * np.prod(target.shape[1:]))}


def plot_split(split, arrays, record, transform, output, example_count):
    qsum = arrays["weight"].sum(axis=(1, 2, 3))
    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    for start, curves in enumerate(arrays["curves"]):
        mean = np.sum(curves * qsum[None], axis=1) / qsum.sum()
        axis.plot(np.arange(len(mean)), mean, label="Encoder initialization" if start == 0 else f"Prior initialization {start}")
    retained = np.minimum.accumulate(arrays["curves"], axis=1).min(axis=0)
    axis.plot(np.arange(len(retained)), (retained * qsum[None]).sum(axis=1) / qsum.sum(), color="tab:green", linewidth=2, label="Best retained across starts/steps")
    for method, style in (("encoder", "--"), ("free_spline", ":")):
        axis.axhline(record["metrics"][method]["legacy_reconstruction"], linestyle=style, color="black", label=method)
    axis.set(xlabel="Adam latent steps", ylabel="Legacy weighted reconstruction loss", yscale="log", title=f"{split}: oracle fitting convergence (no KL)")
    axis.legend(fontsize=8)
    figure.savefig(output / f"convergence_{split}.png", dpi=150)
    plt.close(figure)

    examples = np.linspace(0, len(arrays["target"]) - 1, min(example_count, len(arrays["target"])), dtype=int)
    figure, axes = plt.subplots(2 * len(examples), 4, figsize=(13, 4 * len(examples)), squeeze=False, layout="constrained")
    center = np.asarray(transform.center).reshape(2, 1, 1)
    spread = np.asarray(transform.spread).reshape(2, 1, 1)
    for pair, frame in enumerate(examples):
        support = arrays["mask"][frame, 0] & (arrays["weight"][frame, 0] > 0)
        truth = arrays["target"][frame] * spread + center
        for channel, field in enumerate(("WIC", "SI12")):
            maximum = max(.1, float(np.percentile(truth[channel][support], 99)))
            for column, method in enumerate(["target", *METHODS]):
                values = arrays[method][frame] * spread + center
                shown = np.where(support, np.maximum(values[channel], 0), np.nan)
                title = "Observed" if column == 0 else LABELS[column - 1]
                if column:
                    rmse = record["metrics"][method]["per_frame_quality_rmse"][frame][channel]
                    title += f"\nweighted RMSE {rmse:.3f}"
                image = axes[2 * pair + channel, column].imshow(shown, origin="lower", cmap="viridis", vmin=0, vmax=maximum)
                axes[2 * pair + channel, column].set_title(title, fontsize=9)
                axes[2 * pair + channel, column].set_xlabel("CS column")
                if column == 0:
                    axes[2 * pair + channel, column].set_ylabel(f"{field}\norbit {record['selection']['orbit'][frame]}\nCS row")
            figure.colorbar(image, ax=list(axes[2 * pair + channel]), shrink=.8, label="log10(counts + 1)")
    figure.suptitle(f"{split}: positive-quality support; oracle columns use the targets\nDisplay floor at zero only; metrics use unclipped standardized-log predictions")
    figure.savefig(output / f"reconstruction_examples_{split}.png", dpi=150)
    plt.close(figure)


def plot_summary(results, output):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for channel, axis in enumerate(axes):
        for method_index, (method, label) in enumerate(zip(METHODS, LABELS)):
            values = [result["metrics"][method]["quality_weighted_rmse"][channel] for result in results.values()]
            axis.bar(np.arange(len(results)) + .25 * (method_index - 1), values, width=.25, label=label)
        axis.set_xticks(range(len(results)), list(results))
        axis.set(title=("WIC", "SI12")[channel], ylabel="Quality-weighted standardized-log RMSE")
        axis.legend(fontsize=7)
    figure.savefig(output / "reconstruction_rmse.png", dpi=150)
    plt.close(figure)


#%% Run the three-way diagnostic without updating the saved model
def run(arguments):
    output = arguments.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty output directory; previous diagnostics are preserved")
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if arguments.device == "auto" and torch.cuda.is_available() else arguments.device
    device = "cpu" if device == "auto" else device
    artifact_path = arguments.run_directory / "wic_si12_vae.pt"
    before = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    model, transform, artifact = load_run_artifact(artifact_path, device=device)
    if not isinstance(model, SplineWICSI12VAE) or transform.kind != transform.LOG10_KIND:
        raise ValueError("diagnostic requires the saved partial-convolution/spline log10 model")
    if artifact["training_configuration"]["loss_configuration"]["kind"] != "legacy_weighted_mse_v1":
        raise ValueError("use a legacy-loss control; uncertainty integration is currently deferred")
    original_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.eval().requires_grad_(False)
    reused = None if arguments.selection is None else json.loads(arguments.selection.read_text())
    results, saved_arrays, selections = {}, {}, {}
    for split_index, split in enumerate(arguments.splits):
        dataset, _ = _load_heldout_dataset(artifact, transform, requested_split=split)
        selection = selection_record(dataset, artifact, split, arguments.frame_count, arguments.seed + split_index,
                                     None if reused is None else reused["splits"][split])
        selections[split] = selection
        selected = next(iter(DataLoader(Subset(dataset, selection["indices"]), batch_size=len(selection["indices"]))))
        batch = {key: selected[key].to(device) for key in ("encoder_input", "target", "mask", "weight")}
        del dataset, selected
        print(f"Diagnosing {split}: {len(selection['indices'])} matched frames on {device}", flush=True)
        with torch.no_grad():
            mu, _ = model.encode(batch["encoder_input"], batch["mask"])
            encoder = model.decode(mu)
        target, mask, weight = [batch[key].cpu().numpy() for key in ("target", "mask", "weight")]
        spline, coefficients, ranks = fit_free_splines(target, mask, weight, model.row_basis.cpu().numpy().astype(float),
                                                       model.column_basis.cpu().numpy().astype(float), arguments.spline_rcond)
        latent, fitted_z, winning_start, curves = optimize_latents(model, batch, mu, arguments.steps, arguments.starts,
                                                                  arguments.learning_rate, arguments.seed + 1009 * split_index,
                                                                  arguments.batch_size, progress=True)
        arrays = {"target": target, "mask": mask, "weight": weight, "encoder": encoder.cpu().numpy(),
                  "optimized_latent": latent.cpu().numpy(), "free_spline": spline, "spline_coefficients": coefficients,
                  "encoded_mu": mu.cpu().numpy(), "optimized_z": fitted_z.cpu().numpy(), "curves": curves}
        metrics = {method: reconstruction_metrics(arrays[method], target, mask, weight) for method in METHODS}
        qsum = weight.sum(axis=(1, 2, 3))
        retained = np.minimum.accumulate(curves, axis=1).min(axis=0)
        retained_mean = (retained * qsum[None]).sum(axis=1) / qsum.sum()
        window = max(1, arguments.steps // 10)
        record = {"selection": selection, "metrics": metrics, "spline_numerical_rank": ranks,
                  "winning_start": winning_start.cpu().tolist(), "encoder_latent_norm": mu.norm(dim=1).cpu().tolist(),
                  "optimized_latent_norm": fitted_z.norm(dim=1).cpu().tolist(),
                  "convergence": {"last_window_steps": window, "best_loss_window_start": float(retained_mean[-1 - window]),
                                  "best_loss_final": float(retained_mean[-1]),
                                  "fractional_improvement_last_window": float(1 - retained_mean[-1] / max(retained_mean[-1 - window], 1e-30))}}
        results[split] = record
        saved_arrays.update({f"{split}_{name}": value for name, value in arrays.items()})
        plot_split(split, arrays, record, transform, output, arguments.examples)
        print(f"  Quality RMSE WIC/SI12: {[(method, metrics[method]['quality_weighted_rmse']) for method in METHODS]}", flush=True)
    if before != hashlib.sha256(artifact_path.read_bytes()).hexdigest():
        raise RuntimeError("saved model artifact changed during diagnosis")
    if any(not torch.equal(value.cpu(), original_state[name]) for name, value in model.state_dict().items()):
        raise RuntimeError("model weights or buffers changed during diagnosis")
    report = {"run": str(arguments.run_directory.resolve()), "artifact_sha256": before,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "device": device,
              "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()},
              "model_configuration": artifact["model_configuration"], "transform": artifact["transform"], "splits": results,
              "limitations": ["Spline and optimized-latent columns use target data: not deployable reconstruction or assimilation",
              "No KL, latent regularizer or prior constraint in oracle fits; fitting ability is not prior/covariance skill",
              "Finite multi-start optimization is an achieved fit, not proof of the decoder's global approximation floor",
              "Free spline floor depends on numerical rank tolerance; zero-quality/gap extrapolation is unconstrained",
              "Eval-mode full inputs, not train-mode stochastic/holed optimization; no model or default changes",
              "No subset retraining or uncertainty integration in this three-way diagnostic"]}
    plot_summary(results, output)
    np.savez_compressed(output / "comparisons.npz", **saved_arrays)
    (output / "selection.json").write_text(json.dumps({"policy": "seeded orbit round robin; reuse exact identities", "splits": selections}, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "validation"), default=["train", "validation"])
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--starts", type=int, default=3, help="Total starts: encoder mean first, then seeded standard-normal vectors")
    parser.add_argument("--learning-rate", type=float, default=.03)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--spline-rcond", type=float, default=1e-10)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--selection", type=Path, help="Reuse saved selection.json; overrides frame-count/seed for frame selection")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args(argv)
    if any(getattr(arguments, name) <= 0 for name in ("frame_count", "steps", "starts", "batch_size", "examples")):
        parser.error("frame count, steps, starts, batch size and example count must be positive")
    if not np.isfinite(arguments.learning_rate) or arguments.learning_rate <= 0 or not 0 < arguments.spline_rcond < 1:
        parser.error("learning rate must be finite/positive and spline-rcond between zero and one")
    if len(set(arguments.splits)) != len(arguments.splits) or arguments.seed < 0:
        parser.error("splits must be unique and seed nonnegative")
    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    run(arguments)
    print(f"Saved matched diagnostic to {arguments.output}")
