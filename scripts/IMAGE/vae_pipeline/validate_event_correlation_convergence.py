"""Check anchored physical correlations without storing full 50k ensembles."""

#%% Imports and streaming covariance rows

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from icanalyzer.wic_si12.workflow import load_run_artifact
from scripts.IMAGE.vae_pipeline.validate_conductance_prior import input_evidence
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import physical_decode


def update_moments(count, mean, variance_sum, cross_sum, fields, anchors):
    """Parallel Welford merge; cross_sum contains only selected covariance rows."""
    block_mean = fields.mean(axis=0)
    centered = fields - block_mean
    difference = block_mean - mean
    combined = count + len(fields)
    weight = count * len(fields) / combined
    variance_sum += np.sum(centered**2, axis=0) + difference**2 * weight
    cross_sum += centered[:, anchors].T @ centered + np.outer(difference[anchors], difference) * weight
    mean += difference * len(fields) / combined
    return combined


def correlations(variance, rows, anchors):
    denominator = np.sqrt(variance[anchors, None] * variance[None])
    result = np.divide(rows, denominator, out=np.full_like(rows, np.nan), where=denominator > 0)
    return np.clip(result, -1, 1)


def stream_correlations(model, transform, support, mu, logvar, noise, anchors, checkpoints):
    size = 2 * int(support.sum())
    count = 0
    mean = np.zeros(size)
    variance_sum = np.zeros(size)
    cross_sum = np.zeros((len(anchors), size))
    snapshots = {}
    with torch.no_grad():
        for checkpoint in checkpoints:
            while count < checkpoint:
                stop = min(count + 128, checkpoint)
                latent = mu[None] + np.exp(0.5 * logvar)[None] * noise[count:stop]
                decoded = model.decode(torch.as_tensor(latent, dtype=torch.float32))
                physical = physical_decode(decoded, transform, support)
                fields = physical[:, :, support].reshape(len(latent), -1).astype(float)
                if not np.isfinite(fields).all():
                    raise ValueError("nonfinite supported decoded field")
                count = update_moments(count, mean, variance_sum, cross_sum, fields, anchors)
            variance = variance_sum / (count - 1)
            rows = cross_sum / (count - 1)
            snapshots[count] = {"mean": mean.copy(), "variance": variance.copy(), "rows": rows.copy(), "correlation": correlations(variance, rows, anchors)}
            print(f"  {count} decoded draws", flush=True)
    return snapshots


def difference(first, second):
    valid = np.isfinite(first) & np.isfinite(second)
    values = first[valid] - second[valid]
    return {"finite_entries": int(valid.sum()), "RMS": float(np.sqrt(np.mean(values**2))), "max_absolute": float(np.max(np.abs(values)))}


#%% Matched-count, independent-seed and fixed-anchor diagnostics

def run(arguments):
    output = arguments.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    run_directory = arguments.run_directory.expanduser()
    model, transform, artifact = load_run_artifact(run_directory / "conductance_vae.pt", device="cpu")
    if getattr(model, "kp_conditioning", None) is not None:
        raise ValueError("this image-conditioned correlation audit requires an unconditional VAE; use Kp-matched diagnostics for conditional models")
    evidence = input_evidence(run_directory, artifact)
    event_report = json.loads((arguments.events / "event_covariance_test.json").read_text())
    if event_report["evidence"] != evidence:
        raise ValueError("saved event diagnostics refer to different model/source evidence")
    support = np.asarray(artifact["data_provenance"]["training_support"], dtype=bool)
    points = np.flatnonzero(support.ravel())
    # Three fixed cells; first preserves the original maximum-prior-Hall-STD anchor.
    locations = [tuple(event_report["configuration"]["anchor_row_column"]), (28, 34), (12, 23)]
    positions = [int(np.flatnonzero(points == np.ravel_multi_index(cell, support.shape))[0]) for cell in locations]
    anchors = np.array([position + channel * len(points) for position in positions for channel in (0, 1)])
    cases = {"prior": (np.zeros(model.latent_dim), np.zeros(model.latent_dim))}
    selected_orbits = arguments.orbits or [event["orbit"] for event in event_report["events"][:2]]
    for orbit in selected_orbits:
        with np.load(arguments.events / f"event_{orbit:04d}.npz") as event:
            if not np.array_equal(support, event["support"]):
                raise ValueError("saved event and model support differ")
            cases[f"orbit_{orbit}"] = (event["mu"], event["logvar"])
    checkpoints = (4096, 16384, 50000)
    results = {"draw_counts": checkpoints, "seeds": [12104, 12105], "anchors_row_column": locations, "cases": {}, "input_evidence": evidence}
    stored = {}
    for seed in results["seeds"]:
        noise = np.random.default_rng(seed).normal(size=(checkpoints[-1], model.latent_dim)).astype(np.float32)
        if seed == 12104 and not np.array_equal(noise[:4096], np.load(arguments.events / "base_gaussian_noise.npy")):
            raise ValueError("original 4096 noise prefix differs")
        for name, (mu, logvar) in cases.items():
            print(f"{name}, seed {seed}", flush=True)
            snapshots = stream_correlations(model, transform, support, mu, logvar, noise, anchors, checkpoints)
            stored[name, seed] = snapshots
            arrays = {f"{quantity}_{count}": value for count, snapshot in snapshots.items() for quantity, value in snapshot.items()}
            np.savez_compressed(output / f"{name}_seed{seed}.npz", support=support, anchor_indices=anchors, **arrays)
    for name in cases:
        first, second = stored[name, 12104], stored[name, 12105]
        result = {"4096_vs_50000": difference(first[4096]["correlation"], first[50000]["correlation"]),
                  "independent_50000": difference(first[50000]["correlation"], second[50000]["correlation"]),
                  "zero_variance_states_50000": int((first[50000]["variance"] == 0).sum()),
                  "anchor_STD_S_50000": np.sqrt(first[50000]["variance"][anchors]).tolist()}
        result["original_anchor"] = {"4096_vs_50000": difference(first[4096]["correlation"][:2], first[50000]["correlation"][:2]),
                                     "independent_50000": difference(first[50000]["correlation"][:2], second[50000]["correlation"][:2])}
        if name != "prior":
            result["difference_from_prior_50000"] = difference(first[50000]["correlation"], stored["prior", 12104][50000]["correlation"])
            result["original_anchor"]["difference_from_prior_50000"] = difference(first[50000]["correlation"][:2], stored["prior", 12104][50000]["correlation"][:2])
            # Reproduce the original conditional anchor rows, including state order.
            with np.load(arguments.events / f"event_{int(name.split('_')[1]):04d}.npz") as event:
                covariance = event["covariance_S2"]
                original = correlations(np.diag(covariance), covariance[anchors[:2]], anchors[:2])
                result["original_4096_reproduction"] = difference(first[4096]["correlation"][:2], original)
        # Far-field summary retains finite correlations only, with a stated pixel distance.
        rr, cc = np.indices(support.shape)
        far = ((rr - locations[0][0])**2 + (cc - locations[0][1])**2 >= 12**2)[support]
        hall_far = first[50000]["correlation"][0, :len(points)][far]
        hall_far = hall_far[np.isfinite(hall_far)]
        result["Hall_to_Hall_far_mean_absolute_r"] = float(np.mean(np.abs(hall_far)))
        results["cases"][name] = result
    if input_evidence(run_directory, artifact) != evidence:
        raise RuntimeError("source/model evidence changed")
    results["interpretation"] = "Within-image encoder q correlations and unconditional N(0,I) correlations refer to different distributions; sampling convergence does not calibrate physical uncertainty. Undefined correlations remain NaN."
    (output / "correlation_convergence.json").write_text(json.dumps(results, indent=2) + "\n")

    # Same anchor and color limits for a direct 4096/50000 comparison.
    figure, axes = plt.subplots(3, 3, figsize=(11, 10), layout="constrained")
    for row, name in enumerate(cases):
        small = stored[name, 12104][4096]["correlation"][0, :len(points)]
        large = stored[name, 12104][50000]["correlation"][0, :len(points)]
        for column, (values, title) in enumerate(zip((small, large, large-small), ("4096", "50000", "50000 − 4096"))):
            field = np.full(support.shape, np.nan)
            field[support] = values
            limit = 1 if column < 2 else 0.1
            im = axes[row, column].imshow(field, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[row, column].plot(locations[0][1], locations[0][0], "kx")
            axes[row, column].set_title(f"{name}: H anchor → H, {title}")
            figure.colorbar(im, ax=axes[row, column], shrink=0.75, extend="both" if column == 2 else "neither")
    figure.suptitle("Matched-count teleconnection check; difference panels use ±0.1")
    figure.savefig(output / "correlation_count_check.png", dpi=160)
    figure.savefig(output / "correlation_count_check.pdf")
    plt.close(figure)
    figure, axes = plt.subplots(3, 3, figsize=(11, 10), layout="constrained")
    for row, name in enumerate(cases):
        for column, location in enumerate(locations):
            field = np.full(support.shape, np.nan)
            field[support] = stored[name, 12104][50000]["correlation"][2*column, :len(points)]
            im = axes[row, column].imshow(field, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
            axes[row, column].plot(location[1], location[0], "kx")
            axes[row, column].set_title(f"{name}: H → H, anchor {location}")
            figure.colorbar(im, ax=axes[row, column], shrink=0.75)
    figure.suptitle("Fixed-anchor sensitivity; 50,000 draws per distribution")
    figure.savefig(output / "correlation_anchor_check.png", dpi=160)
    figure.savefig(output / "correlation_anchor_check.pdf")
    plt.close(figure)
    print(json.dumps(results["cases"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--orbits", type=int, nargs=2, help="two saved event orbits; defaults to first two in the event report")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
