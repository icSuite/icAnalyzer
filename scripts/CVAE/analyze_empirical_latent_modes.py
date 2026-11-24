"""Test Kp-local empirical CVAE latents for stable multimodal structure."""

#%% Imports

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupShuffleSplit

from icanalyzer.cvae.artifacts import load_run_artifact
from scripts.CVAE.sample_wic_at_kp import polar_contour


FIELD_NAMES = ("wic_corrected", "si12")
FIELD_LABELS = ("Corrected WIC", "SI12")


#%% Bank and held-out split validation

def load_bank(filename):
    with np.load(filename, allow_pickle=False) as stored:
        required = (
            "mu", "logvar", "kp", "dataset_index", "orbit",
            "source_frame", "time",
        )
        missing = [name for name in required if name not in stored]
        if missing:
            raise ValueError(
                "empirical latent bank is missing " + ", ".join(missing)
            )
        bank = {name: np.asarray(stored[name]) for name in required}
    count = len(bank["kp"])
    if (
        bank["mu"].ndim != 2
        or bank["logvar"].shape != bank["mu"].shape
        or any(bank[name].shape != (count,) for name in required[2:])
        or len(bank["mu"]) != count
        or not np.isfinite(bank["mu"]).all()
        or not np.isfinite(bank["logvar"]).all()
        or not np.isfinite(bank["kp"]).all()
    ):
        raise ValueError("empirical latent bank arrays do not align")
    return bank


def orbit_heldout_split(orbits, test_fraction, seed):
    orbits = np.asarray(orbits)
    if orbits.ndim != 1 or len(np.unique(orbits)) < 2:
        raise ValueError("mode analysis requires at least two source orbits")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=seed
    )
    train, test = next(splitter.split(np.zeros(len(orbits)), groups=orbits))
    if not len(train) or not len(test):
        raise ValueError("orbit-held-out split produced an empty partition")
    return train, test


#%% PCA and candidate mixture selection

def retained_pca_components(explained_ratio, target, minimum=2):
    explained_ratio = np.asarray(explained_ratio, dtype=float)
    if (
        explained_ratio.ndim != 1 or not len(explained_ratio)
        or not np.isfinite(explained_ratio).all()
        or np.any(explained_ratio < 0)
        or not 0 < target <= 1
    ):
        raise ValueError("invalid PCA explained-variance request")
    count = int(np.searchsorted(np.cumsum(explained_ratio), target) + 1)
    return min(len(explained_ratio), max(int(minimum), count))


def fit_pca(train_values, test_values, variance_target, maximum, seed):
    maximum = min(int(maximum), len(train_values) - 1, train_values.shape[1])
    if maximum < 2:
        raise ValueError("mode analysis requires at least two PCA dimensions")
    pca = PCA(
        n_components=maximum, svd_solver="randomized", random_state=seed
    )
    train_full = pca.fit_transform(train_values)
    test_full = pca.transform(test_values)
    retained = retained_pca_components(
        pca.explained_variance_ratio_, variance_target
    )
    return pca, retained, train_full[:, :retained], test_full[:, :retained]


def fit_mixture_candidates(
    train_scores, test_scores, maximum_components, covariance_type,
    regularization, initializations, seed,
):
    records, models = [], []
    for components in range(1, int(maximum_components) + 1):
        model = GaussianMixture(
            n_components=components,
            covariance_type=covariance_type,
            reg_covar=regularization,
            n_init=initializations,
            max_iter=500,
            random_state=seed + components,
        )
        model.fit(train_scores)
        test_log_density = model.score_samples(test_scores)
        records.append({
            "components": components,
            "bic": float(model.bic(train_scores)),
            "aic": float(model.aic(train_scores)),
            "train_log_likelihood": float(model.score(train_scores)),
            "test_log_likelihood": float(np.mean(test_log_density)),
            "test_log_likelihood_standard_error": float(
                np.std(test_log_density, ddof=1) / np.sqrt(len(test_log_density))
            ),
            "minimum_weight": float(np.min(model.weights_)),
            "converged": bool(model.converged_),
            "iterations": int(model.n_iter_),
        })
        models.append(model)
    return records, models


def select_candidate(records, minimum_component_fraction):
    eligible = [
        index for index, record in enumerate(records)
        if record["converged"]
        and record["minimum_weight"] >= minimum_component_fraction
    ]
    if not eligible:
        eligible = [
            index for index, record in enumerate(records)
            if record["converged"]
        ]
    if not eligible:
        raise RuntimeError("no Gaussian-mixture candidate converged")
    return min(eligible, key=lambda index: records[index]["bic"])


def assignment_stability(
    train_scores, evaluation_scores, reference_labels, components,
    covariance_type, regularization, repeats, seed,
):
    scores = []
    for repeat in range(repeats):
        model = GaussianMixture(
            n_components=components,
            covariance_type=covariance_type,
            reg_covar=regularization,
            n_init=1,
            max_iter=500,
            random_state=seed + 10_000 + repeat,
        )
        model.fit(train_scores)
        scores.append(adjusted_rand_score(
            reference_labels, model.predict(evaluation_scores)
        ))
    return np.asarray(scores, dtype=float)


#%% Decoder representatives and plots

def inverse_pca_centers(pca, retained, centers):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != retained:
        raise ValueError("mixture centers do not match retained PCA dimensions")
    return (
        pca.mean_[None]
        + centers @ pca.components_[:retained]
    ).astype(np.float32)


def representative_rows(responsibility, labels):
    rows = []
    for component in range(responsibility.shape[1]):
        candidates = np.flatnonzero(labels == component)
        if not len(candidates):
            candidates = np.arange(len(labels))
        rows.append(int(candidates[
            np.argmax(responsibility[candidates, component])
        ]))
    return np.asarray(rows, dtype=np.int64)


def decode(model, transform, latent, kp, device, support):
    with torch.no_grad():
        values = torch.from_numpy(np.asarray(latent, dtype=np.float32)).to(device)
        condition = torch.full(
            (len(values),), float(kp), dtype=torch.float32, device=device
        )
        physical = transform.inverse(
            model.decode(values, kp=condition).cpu().numpy()
        )
    return np.where(support[None], physical, np.nan)


def plot_model_selection(records, selected, output, kp):
    components = np.asarray([record["components"] for record in records])
    bic = np.asarray([record["bic"] for record in records])
    heldout = np.asarray([record["test_log_likelihood"] for record in records])
    error = np.asarray([
        record["test_log_likelihood_standard_error"] for record in records
    ])
    weights = np.asarray([record["minimum_weight"] for record in records])
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    axes[0].plot(components, bic, marker="o")
    axes[0].set(title="Training BIC", xlabel="mixture components", ylabel="BIC")
    axes[1].errorbar(components, heldout, yerr=error, marker="o")
    axes[1].set(
        title="Orbit-held-out density", xlabel="mixture components",
        ylabel="mean log density",
    )
    axes[2].plot(components, weights, marker="o")
    axes[2].set(
        title="Smallest component", xlabel="mixture components",
        ylabel="mixture weight", ylim=(0, 1),
    )
    for axis in axes:
        axis.axvline(selected, color="black", linestyle="--", linewidth=.8)
        axis.set_xticks(components)
    figure.suptitle(f"Kp category {kp:g}: statistical mode selection")
    figure.savefig(output / "model_selection.png", dpi=180)
    figure.savefig(output / "model_selection.pdf")
    plt.close(figure)


def plot_latent_projection(scores, labels, centers, output, kp):
    figure, axis = plt.subplots(figsize=(7, 5.5), layout="constrained")
    for component in range(centers.shape[0]):
        selected = labels == component
        axis.scatter(
            scores[selected, 0], scores[selected, 1], s=8, alpha=.35,
            label=f"mode {component + 1}",
        )
    axis.scatter(
        centers[:, 0], centers[:, 1], marker="X", s=130,
        edgecolor="black", linewidth=.8, color="white", label="GMM centers",
    )
    axis.set(
        title=f"Kp category {kp:g}: posterior means in PCA space",
        xlabel="PC1", ylabel="PC2",
    )
    axis.legend()
    figure.savefig(output / "latent_projection.png", dpi=180)
    figure.savefig(output / "latent_projection.pdf")
    plt.close(figure)


def plot_mode_gallery(
    center_fields, representative_fields, weights, support, mlat, mlt,
    output, kp, limits,
):
    components = len(weights)
    figure, axes = plt.subplots(
        4, components, figsize=(3.0 * components, 10),
        squeeze=False, layout="constrained",
    )
    rows = (
        (center_fields[:, 0], "WIC: GMM center", limits[0]),
        (representative_fields[:, 0], "WIC: real representative", limits[0]),
        (center_fields[:, 1], "SI12: GMM center", limits[1]),
        (representative_fields[:, 1], "SI12: real representative", limits[1]),
    )
    contours = []
    for row, (fields, label, maximum) in enumerate(rows):
        for component in range(components):
            contour = polar_contour(
                axes[row, component], mlat, mlt,
                np.where(support[row // 2], fields[component], np.nan),
                maximum,
                f"mode {component + 1}; weight={weights[component]:.2f}",
                coordinate_labels=False, title_fontsize=8,
            )
        axes[row, 0].set_ylabel(label)
        contours.append(contour)
    figure.colorbar(contours[0], ax=axes[:2], shrink=.7, label="WIC [counts]")
    figure.colorbar(contours[2], ax=axes[2:], shrink=.7, label="SI12 [counts]")
    figure.suptitle(
        f"Kp category {kp:g}: candidate statistical modes\n"
        "centers are synthetic; representatives are actual encoded frames"
    )
    figure.savefig(output / "decoded_mode_gallery.png", dpi=180)
    figure.savefig(output / "decoded_mode_gallery.pdf")
    plt.close(figure)


#%% One exact-Kp analysis

def analyze_kp(bank, selected_rows, model, transform, artifact, arguments, device):
    kp = float(arguments.current_kp)
    destination = arguments.output / f"kp_{kp:g}"
    destination.mkdir()
    values = bank["mu"][selected_rows]
    orbits = bank["orbit"][selected_rows]
    train, test = orbit_heldout_split(orbits, arguments.test_fraction, arguments.seed)
    pca, retained, train_scores, test_scores = fit_pca(
        values[train], values[test], arguments.pca_variance,
        arguments.pca_max_components, arguments.seed,
    )
    all_scores = pca.transform(values)[:, :retained]
    records, models = fit_mixture_candidates(
        train_scores, test_scores, arguments.max_components,
        arguments.covariance_type, arguments.regularization,
        arguments.initializations, arguments.seed,
    )
    selected_index = select_candidate(
        records, arguments.minimum_component_fraction
    )
    mixture = models[selected_index]
    labels = mixture.predict(all_scores)
    responsibility = mixture.predict_proba(all_scores)
    representatives = representative_rows(responsibility, labels)
    centers = inverse_pca_centers(pca, retained, mixture.means_)
    stability = assignment_stability(
        train_scores, all_scores, labels, mixture.n_components,
        arguments.covariance_type, arguments.regularization,
        arguments.stability_repeats, arguments.seed,
    )

    support = np.asarray(
        artifact["data_provenance"]["training_support"], dtype=bool
    )
    grid = artifact["data_provenance"]["grid"]
    mlat = np.asarray(grid["mlat"], dtype=float)
    mlt = np.asarray(grid["mlt"], dtype=float)
    center_fields = decode(model, transform, centers, kp, device, support)
    representative_fields = decode(
        model, transform, values[representatives], kp, device, support
    )

    plot_model_selection(
        records, mixture.n_components, destination, kp
    )
    plot_latent_projection(
        all_scores, labels, mixture.means_, destination, kp
    )
    plot_mode_gallery(
        center_fields, representative_fields, mixture.weights_, support,
        mlat, mlt, destination, kp,
        (arguments.wic_vmax, arguments.si12_vmax),
    )

    with (destination / "component_assignments.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "bank_row", "dataset_index", "orbit", "source_frame", "time",
            "kp", "component", "responsibility", "pc1", "pc2",
        ))
        for local, bank_row in enumerate(selected_rows):
            component = int(labels[local])
            writer.writerow((
                int(bank_row), int(bank["dataset_index"][bank_row]),
                int(bank["orbit"][bank_row]),
                int(bank["source_frame"][bank_row]), str(bank["time"][bank_row]),
                float(bank["kp"][bank_row]), component + 1,
                float(responsibility[local, component]),
                float(all_scores[local, 0]), float(all_scores[local, 1]),
            ))

    np.savez_compressed(
        destination / "mode_model.npz",
        selected_bank_rows=selected_rows,
        train_local_rows=train, test_local_rows=test,
        pca_mean=pca.mean_, pca_components=pca.components_[:retained],
        pca_explained_variance_ratio=pca.explained_variance_ratio_[:retained],
        scores=all_scores, labels=labels, responsibility=responsibility,
        mixture_weights=mixture.weights_, mixture_means=mixture.means_,
        mixture_covariances=mixture.covariances_,
        representative_local_rows=representatives,
        representative_bank_rows=selected_rows[representatives],
        decoded_centers=center_fields,
        decoded_representatives=representative_fields,
        stability_adjusted_rand=stability,
    )
    summary = {
        "exact_kp": kp,
        "kp_tolerance": arguments.kp_tolerance,
        "frames": int(len(selected_rows)),
        "orbits": int(len(np.unique(orbits))),
        "train_frames": int(len(train)),
        "test_frames": int(len(test)),
        "pca_components": int(retained),
        "pca_variance_captured": float(
            np.sum(pca.explained_variance_ratio_[:retained])
        ),
        "selection_policy": (
            "minimum training BIC among converged candidates whose smallest "
            "component meets the configured fraction; held-out likelihood is "
            "reported independently"
        ),
        "selected_components": int(mixture.n_components),
        "component_weights": mixture.weights_.tolist(),
        "representative_bank_rows": selected_rows[representatives].tolist(),
        "stability_adjusted_rand": {
            "minimum": float(np.min(stability)),
            "median": float(np.median(stability)),
            "maximum": float(np.max(stability)),
        },
        "candidate_models": records,
        "qualification": (
            "Mixture components are statistical candidates, not physical "
            "auroral modes, until morphology, stability, metadata and data "
            "quality checks support that interpretation."
        ),
    }
    (destination / "mode_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


#%% Run and CLI

def run(arguments):
    bank_path = arguments.bank.expanduser().resolve()
    bank = load_bank(bank_path)
    run_directory = arguments.run_directory.expanduser().resolve()
    arguments.output = arguments.output.expanduser().resolve()
    if arguments.output.exists() and (
        not arguments.output.is_dir() or any(arguments.output.iterdir())
    ):
        raise FileExistsError("choose an empty latent-mode output directory")
    arguments.output.mkdir(parents=True, exist_ok=True)
    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if arguments.device == "auto" else arguments.device
    device = torch.device(device_name)
    artifact_path = run_directory / "wic_cvae.pt"
    model, transform, artifact = load_run_artifact(artifact_path, device=device)
    if tuple(model.field_names) != FIELD_NAMES or model.latent_dim != bank["mu"].shape[1]:
        raise ValueError("latent bank and paired WIC/SI12 artifact do not align")
    sampling_summary = bank_path.parent / "sampling_summary.json"
    if sampling_summary.is_file():
        expected_sha = json.loads(sampling_summary.read_text()).get(
            "artifact_sha256"
        )
        actual_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if expected_sha != actual_sha:
            raise ValueError("latent bank and run artifact hashes do not match")

    summaries = []
    for kp in arguments.kp:
        selected = np.flatnonzero(
            np.abs(bank["kp"] - kp) <= arguments.kp_tolerance
        )
        if len(selected) < arguments.minimum_frames:
            raise ValueError(
                f"Kp category {kp:g} has only {len(selected)} bank frames; "
                "reduce --minimum-frames or rebuild a larger bank"
            )
        arguments.current_kp = kp
        summaries.append(analyze_kp(
            bank, selected, model, transform, artifact, arguments, device
        ))
        print(
            f"Kp {kp:g}: selected {summaries[-1]['selected_components']} "
            "candidate component(s)", flush=True,
        )
    overall = {
        "latent_bank": str(bank_path),
        "run_directory": str(run_directory),
        "analysis": summaries,
    }
    (arguments.output / "mode_analysis_summary.json").write_text(
        json.dumps(overall, indent=2) + "\n"
    )
    return overall


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--kp", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kp-tolerance", type=float, default=0.01)
    parser.add_argument("--minimum-frames", type=int, default=500)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--pca-max-components", type=int, default=40)
    parser.add_argument("--max-components", type=int, default=6)
    parser.add_argument("--covariance-type", choices=("diag", "full"), default="diag")
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--initializations", type=int, default=5)
    parser.add_argument("--minimum-component-fraction", type=float, default=0.03)
    parser.add_argument("--stability-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12104)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wic-vmax", type=float, default=1000)
    parser.add_argument("--si12-vmax", type=float, default=50)
    arguments = parser.parse_args(argv)
    if (
        not np.isfinite(arguments.kp).all()
        or any(not 0 <= value <= 9 for value in arguments.kp)
        or len(set(arguments.kp)) != len(arguments.kp)
        or not np.isfinite(arguments.kp_tolerance) or arguments.kp_tolerance < 0
        or arguments.minimum_frames < 2
        or not 0 < arguments.test_fraction < 1
        or not 0 < arguments.pca_variance <= 1
        or arguments.pca_max_components < 2
        or arguments.max_components < 1
        or not np.isfinite(arguments.regularization) or arguments.regularization <= 0
        or arguments.initializations < 1 or arguments.stability_repeats < 1
        or not 0 <= arguments.minimum_component_fraction < 1
        or arguments.wic_vmax <= 0 or arguments.si12_vmax <= 0
    ):
        parser.error("invalid Kp, split, PCA, mixture, stability or display setting")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
