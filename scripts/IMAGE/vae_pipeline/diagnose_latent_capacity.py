"""Overfit a matched training subset deterministically; never update the source VAE."""

#%% Imports
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from icanalyzer.wic_si12.model import SplineWICSI12VAE
from icanalyzer.wic_si12.workflow import load_run_artifact
from scripts.IMAGE.vae_pipeline.diagnose_reconstruction_bottleneck import (
    fit_free_splines, legacy_reconstruction, reconstruction_metrics, selection_record
)
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import _load_heldout_dataset


#%% Deterministic predictions and best eval-mode subset checkpoint
def predict_fields(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            mu, _ = model.encode(batch["encoder_input"].to(device), batch["mask"].to(device))
            predictions.append(model.decode(mu).cpu().numpy())
    return np.concatenate(predictions)


def fit_subset(model, subset, target, mask, weight, arguments, device):
    """From-scratch local training: no reparameterization, KL or input holes."""
    evaluation_loader = DataLoader(subset, batch_size=arguments.batch_size, shuffle=False)
    training_loader = DataLoader(subset, batch_size=arguments.batch_size, shuffle=True,
                                 generator=torch.Generator().manual_seed(arguments.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    prediction = predict_fields(model, evaluation_loader, device)
    metrics = reconstruction_metrics(prediction, target, mask, weight)
    best_loss, best_epoch = metrics["legacy_reconstruction"], 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [{"epoch": 0, "train_mode_batch_mean": None, "eval_mode_subset_loss": best_loss}]
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        losses = []
        for batch in training_loader:
            optimizer.zero_grad()
            mu, _ = model.encode(batch["encoder_input"].to(device), batch["mask"].to(device))
            reconstructed = model.decode(mu)
            loss = legacy_reconstruction(reconstructed, batch["target"].to(device),
                                          batch["mask"].to(device), batch["weight"].to(device))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite deterministic training loss; inspect data or learning rate")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        prediction = predict_fields(model, evaluation_loader, device)
        metrics = reconstruction_metrics(prediction, target, mask, weight)
        evaluated = metrics["legacy_reconstruction"]
        history.append({"epoch": epoch, "train_mode_batch_mean": float(np.mean(losses)), "eval_mode_subset_loss": evaluated})
        if evaluated < best_loss:
            best_loss, best_epoch = evaluated, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 100 == 0 or epoch == arguments.epochs:
            print(f"latent {model.latent_dim}, epoch {epoch}/{arguments.epochs}, eval loss {evaluated:.3f}, best {best_loss:.3f} at {best_epoch}", flush=True)
    model.load_state_dict(best_state)
    prediction = predict_fields(model, evaluation_loader, device)
    metrics = reconstruction_metrics(prediction, target, mask, weight)
    return prediction, metrics, best_epoch, history


#%% Matched examples, error comparison and train/eval convergence
def plot_results(fields, metrics, histories, target, mask, weight, transform, selection, output, examples):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    labels = list(fields)
    for channel, axis in enumerate(axes):
        axis.bar(range(len(labels)), [metrics[label]["quality_weighted_rmse"][channel] for label in labels])
        axis.set_xticks(range(len(labels)), labels, rotation=15, ha="right")
        axis.set(title=("WIC", "SI12")[channel], ylabel="Quality-weighted standardized-log RMSE")
    figure.savefig(output / "overfit_comparison.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    for label, history in histories.items():
        axis.plot([h["epoch"] for h in history], [h["eval_mode_subset_loss"] for h in history], label=f"{label}: eval")
        axis.plot([h["epoch"] for h in history[1:]], [h["train_mode_batch_mean"] for h in history[1:]], ":", label=f"{label}: train batch mean")
    reference_labels = [label for label in fields if label in ("Source encoder", "Free spline") or label.startswith("Matched overfit")]
    for label in reference_labels:
        axis.axhline(metrics[label]["legacy_reconstruction"], linestyle="--", label=label)
    axis.set(xlabel="Deterministic training epochs", ylabel="Legacy reconstruction loss", yscale="log", title="Same-frame overfit: beta=0, no sampling, no holes")
    axis.legend(fontsize=8)
    figure.savefig(output / "overfit_convergence.png", dpi=150)
    plt.close(figure)

    chosen = np.linspace(0, len(target) - 1, min(examples, len(target)), dtype=int)
    figure, axes = plt.subplots(2 * len(chosen), 1 + len(fields), figsize=(3 * (1 + len(fields)), 4 * len(chosen)), squeeze=False, layout="constrained")
    center = np.asarray(transform.center).reshape(2, 1, 1)
    spread = np.asarray(transform.spread).reshape(2, 1, 1)
    for pair, frame in enumerate(chosen):
        support = mask[frame, 0] & (weight[frame, 0] > 0)
        truth = target[frame] * spread + center
        panels = {"Observed": target, **fields}
        for channel, field in enumerate(("WIC", "SI12")):
            maximum = max(.1, float(np.percentile(truth[channel][support], 99)))
            for column, (label, predictions) in enumerate(panels.items()):
                values = np.maximum(predictions[frame] * spread + center, 0)
                shown = np.where(support, values[channel], np.nan)
                title = label
                if column:
                    title += f"\nRMSE {metrics[label]['per_frame_quality_rmse'][frame][channel]:.3f}"
                image = axes[2 * pair + channel, column].imshow(shown, origin="lower", cmap="viridis", vmin=0, vmax=maximum)
                axes[2 * pair + channel, column].set_title(title, fontsize=9)
                axes[2 * pair + channel, column].set_xlabel("CS column")
                if column == 0:
                    axes[2 * pair + channel, column].set_ylabel(f"{field}\norbit {selection['orbit'][frame]}\nCS row")
            figure.colorbar(image, ax=list(axes[2 * pair + channel]), shrink=.8, label="log10(counts + 1)")
    figure.suptitle("Exact matched training frames; positive-quality support\nOracles use the targets; overfit is training-set reconstruction, not generalization")
    figure.savefig(output / "overfit_reconstruction_examples.png", dpi=150)
    plt.close(figure)


#%% Source model, exact selection and paired E04 baselines
def run(arguments):
    run_directory = Path(arguments.run_directory)
    output = Path(arguments.output) if arguments.output else run_directory / "latent_capacity_sweep"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty output; existing diagnostics and models are preserved")
    output.mkdir(parents=True, exist_ok=True)
    device = arguments.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    source_path = run_directory / "wic_si12_vae.pt"
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    saved_model, transform, artifact = load_run_artifact(source_path, device=device)
    if not isinstance(saved_model, SplineWICSI12VAE) or transform.kind != transform.LOG10_KIND:
        raise ValueError("matched overfit requires the saved partial-convolution/spline log10 model")
    if artifact["training_configuration"]["loss_configuration"]["kind"] != "legacy_weighted_mse_v1":
        raise ValueError("matched overfit requires the legacy-loss control; uncertainty is deferred")
    source_state = {name: value.detach().cpu().clone() for name, value in saved_model.state_dict().items()}
    dataset, split_name = _load_heldout_dataset(artifact, transform, requested_split="train")
    selected_manifest = None if arguments.selection is None else json.loads(arguments.selection.read_text())
    saved_selection = None if selected_manifest is None else selected_manifest["splits"]["train"]
    selection = selection_record(dataset, artifact, "train", arguments.frame_count, arguments.seed, saved_selection)
    subset = Subset(dataset, selection["indices"])
    evaluation_loader = DataLoader(subset, batch_size=arguments.batch_size, shuffle=False)
    selected = next(iter(DataLoader(subset, batch_size=len(subset))))
    target, mask, weight = [selected[key].numpy() for key in ("target", "mask", "weight")]
    fields = {"Source encoder": predict_fields(saved_model, evaluation_loader, device)}
    baseline_origin = "refitted free spline with saved basis, rcond=1e-10"
    if arguments.selection is not None:
        parent = arguments.selection.parent
        baseline = json.loads((parent / "metrics.json").read_text())
        if baseline["artifact_sha256"] != source_hash or baseline["splits"]["train"]["selection"] != selection:
            raise ValueError("E04 report source model or training selection differs")
        with np.load(parent / "comparisons.npz") as arrays:
            for key, value in (("target", target), ("mask", mask), ("weight", weight)):
                if not np.array_equal(arrays[f"train_{key}"], value):
                    raise ValueError(f"E04 baseline {key} differs from the matched training data")
            fields["Optimized latent (oracle)"] = arrays["train_optimized_latent"].copy()
            fields["Free spline"] = arrays["train_free_spline"].copy()
        baseline_origin = str(parent.resolve())
    else:
        fields["Free spline"], _, _ = fit_free_splines(target, mask, weight, saved_model.row_basis.cpu().numpy().astype(float),
                                                       saved_model.column_basis.cpu().numpy().astype(float), 1e-10)
    metrics = {label: reconstruction_metrics(prediction, target, mask, weight) for label, prediction in fields.items()}

    # Train separate diagnostic models; default to the source's exact latent size.
    dimensions = arguments.latent_dimensions or [int(artifact["model_configuration"]["latent_dim"])]
    control_record = None
    if arguments.overfit_control is not None:
        control_path = arguments.overfit_control / "latent_capacity_sweep.json"
        control = json.loads(control_path.read_text())
        expected = {"source_artifact_sha256": source_hash, "selection": selection, "epochs": arguments.epochs,
                    "batch_size": arguments.batch_size, "learning_rate": arguments.learning_rate, "seed": arguments.seed,
                    "training": "from scratch, deterministic decode(mu), beta=0, no holes, no scheduler"}
        for key, value in expected.items():
            if control[key] != value:
                raise ValueError(f"matched overfit control differs in {key}")
        if len(control["results"]) != 1 or dimensions != [control["results"][0]["latent_dim"]]:
            raise ValueError("matched overfit control must contain one model with the same latent size")
        control_configuration = control["results"][0]["model_configuration"]
        normalized_control = SplineWICSI12VAE.from_configuration(control_configuration).configuration()
        if normalized_control != dict(saved_model.configuration(), latent_dim=int(dimensions[0])):
            raise ValueError("matched overfit control must use the source decoder architecture")
        control_label = next(iter(control["history"]))
        with np.load(arguments.overfit_control / "overfit_comparisons.npz") as arrays:
            for key, value in (("target", target), ("mask", mask), ("weight", weight)):
                if not np.array_equal(arrays[key], value):
                    raise ValueError(f"matched overfit control {key} differs from the selected data")
            control_prediction = arrays[control_label.replace(" ", "_")].copy()
        label = f"Matched overfit latent {dimensions[0]}, decoder {saved_model.decoder_feature_size}x{saved_model.decoder_feature_size}"
        fields[label] = control_prediction
        metrics[label] = reconstruction_metrics(control_prediction, target, mask, weight)
        if not np.isclose(metrics[label]["legacy_reconstruction"], control["results"][0]["legacy_reconstruction"], rtol=1e-7):
            raise ValueError("matched overfit control predictions disagree with its reported loss")
        control_record = {"directory": str(arguments.overfit_control.resolve()),
                          "report_sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
                          "model_configuration": control_configuration, "best_epoch": control["results"][0]["best_epoch"],
                          "device": control["device"], "metrics": metrics[label]}
    results, histories = [], {}
    for latent_dim in dimensions:
        torch.manual_seed(arguments.seed)
        configuration = dict(artifact["model_configuration"], latent_dim=int(latent_dim))
        if arguments.decoder_feature_size is not None:
            configuration["decoder_feature_size"] = arguments.decoder_feature_size
        model = SplineWICSI12VAE.from_configuration(configuration).to(device)
        configuration = model.configuration()
        with torch.no_grad():
            model.row_basis.copy_(saved_model.row_basis)
            model.column_basis.copy_(saved_model.column_basis)
        print(f"Overfitting {len(subset)} exact training frames, latent {latent_dim}, decoder {model.decoder_feature_size}x{model.decoder_feature_size}, initialized from scratch", flush=True)
        prediction, fitted_metrics, best_epoch, history = fit_subset(model, subset, target, mask, weight, arguments, device)
        label = f"Overfit latent {latent_dim}"
        suffix = ""
        if model.decoder_feature_size != saved_model.decoder_feature_size:
            label += f", decoder {model.decoder_feature_size}x{model.decoder_feature_size}"
            suffix = f"_decoder{model.decoder_feature_size}"
        fields[label], metrics[label], histories[label] = prediction, fitted_metrics, history
        result = {"latent_dim": int(latent_dim), "model_configuration": configuration,
                  "decoder_feature_size": model.decoder_feature_size,
                  "architecture_changed": configuration != saved_model.configuration(),
                  "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                  "best_epoch": best_epoch, "initial_eval_loss": history[0]["eval_mode_subset_loss"],
                  "last_epoch_eval_loss": history[-1]["eval_mode_subset_loss"], **fitted_metrics}
        results.append(result)
        torch.save({"kind": "diagnostic_deterministic_subset_overfit", "model_configuration": configuration,
                    "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "transform": artifact["transform"], "source_artifact_sha256": source_hash,
                    "selection": selection, "best_epoch": best_epoch, "metrics": fitted_metrics},
                   output / f"deterministic_overfit_latent{latent_dim}{suffix}.pt")
    if source_hash != hashlib.sha256(source_path.read_bytes()).hexdigest():
        raise RuntimeError("source model artifact changed during diagnosis")
    if any(not torch.equal(value.cpu(), source_state[name]) for name, value in saved_model.state_dict().items()):
        raise RuntimeError("source model weights or buffers changed during diagnosis")
    plot_results(fields, metrics, histories, target, mask, weight, transform, selection, output, arguments.examples)
    np.savez_compressed(output / "overfit_comparisons.npz", target=target, mask=mask, weight=weight,
                        **{label.replace(" ", "_"): prediction for label, prediction in fields.items()})
    (output / "selection.json").write_text(json.dumps({"splits": {"train": selection}}, indent=2) + "\n")
    diagnostics = {"source_run": str(run_directory.resolve()), "source_split": split_name, "source_artifact_sha256": source_hash,
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "device": str(device),
                   "frame_count": len(subset), "selected_dataset_indices": selection["indices"], "selection": selection,
                   "epochs": arguments.epochs, "batch_size": arguments.batch_size, "learning_rate": arguments.learning_rate,
                   "seed": arguments.seed, "training": "from scratch, deterministic decode(mu), beta=0, no holes, no scheduler",
                   "checkpoint_selection": "minimum eval-mode reconstruction loss on the fixed TRAINING subset",
                   "overfit_control": control_record,
                   "baseline_origin": baseline_origin, "baselines": {label: metrics[label] for label in fields if not label.startswith("Overfit")},
                   "results": results, "history": histories,
                   "limitations": ["Training-set overfit, not held-out skill or geophysical covariance calibration",
                                    "Finite training failure does not prove an architectural fitting limit",
                                    "Oracle baselines use targets and free splines may fit measurement noise",
                                    "Training batch mean and full-subset eval loss differ in BatchNorm mode and reduction",
                                    "No production model, default, uncertainty integration or source artifact changes"]}
    (output / "latent_capacity_sweep.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return diagnostics


#%% Command line
def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("--output")
    parser.add_argument("--selection", type=Path, help="E04 selection.json; reuse exact training frames and paired oracle baselines")
    parser.add_argument("--latent-dimensions", nargs="+", type=int, help="Optional explicit sweep; default is the unchanged saved latent size")
    parser.add_argument("--decoder-feature-size", type=int, choices=(18, 30), help="Opt-in spatial-stage test: feature side length before dec_conv2; default preserves the saved model (legacy 18)")
    parser.add_argument("--overfit-control", type=Path, help="Completed matched overfit folder to include in plots; frames, data, epochs, optimizer settings and source must match")
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args(argv)
    if any(getattr(arguments, name) <= 0 for name in ("frame_count", "epochs", "batch_size", "examples")):
        parser.error("frame count, epochs, batch size and examples must be positive")
    if arguments.latent_dimensions is not None and (min(arguments.latent_dimensions) <= 0 or len(set(arguments.latent_dimensions)) != len(arguments.latent_dimensions)):
        parser.error("explicit latent dimensions must be positive and unique")
    if arguments.seed < 0 or not np.isfinite(arguments.learning_rate) or arguments.learning_rate <= 0:
        parser.error("seed must be nonnegative and learning rate finite/positive")
    return arguments


if __name__ == "__main__":
    report = run(parse_arguments())
    print(f"Completed deterministic overfit on {report['frame_count']} matched training frames")
