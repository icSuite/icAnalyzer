"""Sample minute-resolved paired WIC/SI12 medians and optional medoids."""

#%% Imports

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import torch

from icanalyzer.cvae.artifacts import WIC_CVAE_ARTIFACT_TYPE, load_run_artifact
from icanalyzer.cvae.conditioning import (
    CONDITIONING_PROFILES,
    conditioning_time_features,
    derive_sml_history,
    read_supermag_file,
    resolve_conditioning_features,
)
from icanalyzer.cvae.ensemble import sample_decoded_ensemble
from icanalyzer.cvae.summaries import summarize_wic_samples


#%% Event conditions on an exact one-minute grid

def _interpolate_short_gaps(values, maximum_gap_minutes):
    values = np.asarray(values, dtype=np.float32).copy()
    filled = np.zeros(len(values), dtype=bool)
    missing = ~np.isfinite(values)
    starts = np.flatnonzero(missing & np.r_[True, ~missing[:-1]])
    stops = np.flatnonzero(missing & np.r_[~missing[1:], True])
    for start, stop in zip(starts, stops):
        length = stop - start + 1
        bounded = start > 0 and stop + 1 < len(values)
        if length > maximum_gap_minutes or not bounded:
            continue
        values[start:stop + 1] = np.interp(
            np.arange(start, stop + 1), (start - 1, stop + 1),
            (values[start - 1], values[stop + 1])
        )
        filled[start:stop + 1] = True
    return values, filled


def _event_profile(feature_names):
    names = resolve_conditioning_features(feature_names)
    supported = ("hybrid4", "hybrid6_sml_history")
    matches = [
        profile for profile in supported
        if tuple(CONDITIONING_PROFILES[profile]) == names
    ]
    if not matches:
        raise ValueError(
            "event sampling supports hybrid4 or hybrid6_sml_history conditioning"
        )
    return matches[0], names


def event_conditions(
    supermag, start, stop, maximum_imf_gap_minutes=0,
    feature_names="hybrid4"
):
    """Select every complete SuperMAG minute inside inclusive event bounds."""

    start = np.datetime64(start, "us")
    stop = np.datetime64(stop, "us")
    if np.isnat(start) or np.isnat(stop) or stop < start:
        raise ValueError("event stop must not precede event start")

    first = start.astype("datetime64[m]")
    if first.astype("datetime64[us]") < start:
        first += np.timedelta64(1, "m")
    last = stop.astype("datetime64[m]")
    expected = np.arange(first, last + np.timedelta64(1, "m"), np.timedelta64(1, "m"))
    if not len(expected):
        raise ValueError("event bounds contain no complete one-minute timestamps")

    profile, names = _event_profile(feature_names)
    if profile == "hybrid6_sml_history":
        supermag = derive_sml_history(supermag)
    source_time = np.asarray(supermag["time"], dtype="datetime64[m]")
    positions = np.searchsorted(source_time, expected)
    if (
        np.any(positions >= len(source_time))
        or not np.array_equal(source_time[positions], expected)
    ):
        available = source_time[
            (source_time >= expected[0]) & (source_time <= expected[-1])
        ]
        raise ValueError(
            f"SuperMAG source has {len(available)}/{len(expected)} required event minutes; "
            "do not interpolate missing conditions"
        )

    if maximum_imf_gap_minutes < 0:
        raise ValueError("maximum IMF gap must be nonnegative")
    selected_names = tuple(dict.fromkeys(
        name for name in names if name != "dipole_tilt"
    ))
    selected = {
        name: np.asarray(supermag[name], dtype=np.float32)[positions]
        for name in selected_names
    }
    interpolated = np.zeros(len(expected), dtype=bool)
    for name in ("by_gsm", "bz_gsm"):
        selected[name], filled = _interpolate_short_gaps(
            selected[name], maximum_imf_gap_minutes
        )
        interpolated |= filled
    if not all(np.isfinite(values).all() for values in selected.values()):
        missing = [
            name for name, values in selected.items()
            if not np.isfinite(values).all()
        ]
        raise ValueError(
            f"event contains nonfinite {profile} conditions: " + ", ".join(missing)
        )
    tilt = conditioning_time_features(expected.astype("datetime64[us]"))["dipole_tilt"]
    selected["dipole_tilt"] = tilt
    matrix = np.column_stack([selected[name] for name in names]).astype(np.float32)
    return expected.astype("datetime64[us]"), matrix, interpolated


#%% Resumable minute-by-minute sampling

def sha256(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(arguments):
    if arguments.draws < 2 or arguments.batch_size < 1 or arguments.threads < 1:
        raise ValueError("draws must be at least two; batch size and threads must be positive")
    if (
        arguments.wic_medoid_threshold < 0
        or arguments.si12_medoid_threshold < 0
    ):
        raise ValueError("medoid morphology thresholds must be nonnegative")
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if arguments.device == "auto" else arguments.device
    torch.set_num_threads(arguments.threads)

    run_directory = arguments.run_directory.expanduser().resolve()
    artifact_path = run_directory / "wic_cvae.pt"
    supermag_path = arguments.supermag.expanduser().resolve()
    if not supermag_path.is_file():
        raise FileNotFoundError(
            f"one-minute SuperMAG export not found: {supermag_path}; "
            "replace the example path with the real year_2000.txt"
        )
    model, transform, artifact = load_run_artifact(artifact_path, device=device)
    required_names = resolve_conditioning_features(
        model.conditioning["feature_names"] if model.conditioning else ()
    )
    profile, required_names = _event_profile(required_names)
    if (
        artifact.get("artifact_type") != WIC_CVAE_ARTIFACT_TYPE
        or tuple(model.field_names) != ("wic_corrected", "si12")
        or model.conditioning is None
    ):
        raise ValueError(
            "requires a paired corrected-WIC/SI12 hybrid4 or "
            "hybrid6_sml_history artifact"
        )

    supermag, supermag_provenance = read_supermag_file(supermag_path)
    times, conditions, interpolated = event_conditions(
        supermag, arguments.start, arguments.stop,
        arguments.maximum_imf_gap_minutes, feature_names=required_names
    )
    lower = np.asarray(model.conditioning["training_min"], dtype=np.float32)
    upper = np.asarray(model.conditioning["training_max"], dtype=np.float32)
    outside = np.any((conditions < lower) | (conditions > upper), axis=1)
    if outside.any():
        warnings.warn(
            f"{outside.sum()}/{len(outside)} event minutes extend outside the training range",
            RuntimeWarning, stacklevel=2
        )

    provenance = artifact["data_provenance"]
    support = np.asarray(provenance["training_support"], dtype=bool)
    expected_shape = (2, *model.image_shape)
    if support.shape != expected_shape or not np.all(support.reshape(2, -1).any(axis=1)):
        raise ValueError("artifact training support does not align with paired output fields")

    output = arguments.output.expanduser().resolve()
    configuration = {
        "format": (
            "hybrid4_event_medians_v1" if profile == "hybrid4"
            else "named_event_medians_v2"
        ),
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "supermag": str(supermag_path),
        "supermag_sha256": supermag_provenance["sha256"],
        "requested_start": str(np.datetime64(arguments.start, "us")),
        "requested_stop": str(np.datetime64(arguments.stop, "us")),
        "first_minute": str(times[0]),
        "last_minute": str(times[-1]),
        "minutes": len(times),
        "draws_per_minute": arguments.draws,
        "batch_size": arguments.batch_size,
        "latent_seed": arguments.seed,
        "field_names": list(model.field_names),
        "feature_names": list(required_names),
        "maximum_imf_gap_minutes": arguments.maximum_imf_gap_minutes,
        "interpolated_imf_minutes": int(interpolated.sum()),
        "common_latent_draws_across_minutes": True,
        "interpretation": "minute-wise conditional marginals; not a learned temporal trajectory",
    }
    if profile != "hybrid4":
        configuration["conditioning_profile"] = profile
    config_path = output / "event_configuration.json"
    median_path = output / "event_medians.npy"
    completed_path = output / "completed_minutes.npy"
    medoid_path = output / "event_medoids.npy"
    medoid_index_path = output / "event_medoid_indices.npy"
    medoid_completed_path = output / "completed_medoid_minutes.npy"
    medoid_config_path = output / "event_medoid_configuration.json"

    if arguments.resume:
        saved_configuration = json.loads(config_path.read_text())
        if saved_configuration != configuration:
            raise ValueError("resume configuration differs from the existing event run")
        medians = np.load(median_path, mmap_mode="r+")
        completed = np.load(completed_path, mmap_mode="r+")
        if medians.shape != (len(times), *expected_shape) or completed.shape != (len(times),):
            raise ValueError("resume arrays do not match the event configuration")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("choose an empty event output directory or use --resume")
        output.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(configuration, indent=2) + "\n")
        saved_conditions = {
            name: conditions[:, column]
            for column, name in enumerate(required_names)
        }
        saved_conditions["imf_by"] = saved_conditions.pop("by_gsm")
        saved_conditions["imf_bz"] = saved_conditions.pop("bz_gsm")
        np.savez_compressed(
            output / "event_conditions.npz", time=times,
            **saved_conditions, outside_training_range=outside,
            imf_interpolated=interpolated,
            support=support, mlat=np.asarray(provenance["grid"]["mlat"]),
            mlt=np.asarray(provenance["grid"]["mlt"]),
        )
        medians = np.lib.format.open_memmap(
            median_path, mode="w+", dtype=np.float32,
            shape=(len(times), *expected_shape)
        )
        medians[:] = np.nan
        completed = np.lib.format.open_memmap(
            completed_path, mode="w+", dtype=bool, shape=(len(times),)
        )
        completed[:] = False
        medians.flush()
        completed.flush()

    medoids = medoid_indices = medoid_completed = None
    if arguments.add_medoids:
        medoid_configuration = {
            "format": (
                "hybrid4_event_medoids_v1" if profile == "hybrid4"
                else "named_event_medoids_v2"
            ),
            "artifact_sha256": configuration["artifact_sha256"],
            "draws_per_minute": arguments.draws,
            "latent_seed": arguments.seed,
            "field_names": list(model.field_names),
            "thresholds": {
                "wic_corrected": arguments.wic_medoid_threshold,
                "si12": arguments.si12_medoid_threshold,
            },
            "method": (
                "separate actual-member medoid for each sensor, minimizing squared "
                "distance to the robust-scaled morphology-feature center"
            ),
        }
        if medoid_config_path.exists():
            if json.loads(medoid_config_path.read_text()) != medoid_configuration:
                raise ValueError("medoid configuration differs from the existing event run")
            medoids = np.load(medoid_path, mmap_mode="r+")
            medoid_indices = np.load(medoid_index_path, mmap_mode="r+")
            medoid_completed = np.load(medoid_completed_path, mmap_mode="r+")
        else:
            medoid_config_path.write_text(
                json.dumps(medoid_configuration, indent=2) + "\n"
            )
            medoids = np.lib.format.open_memmap(
                medoid_path, mode="w+", dtype=np.float32,
                shape=(len(times), *expected_shape)
            )
            medoids[:] = np.nan
            medoid_indices = np.lib.format.open_memmap(
                medoid_index_path, mode="w+", dtype=np.int32,
                shape=(len(times), 2)
            )
            medoid_indices[:] = -1
            medoid_completed = np.lib.format.open_memmap(
                medoid_completed_path, mode="w+", dtype=bool,
                shape=(len(times),)
            )
            medoid_completed[:] = False
            medoids.flush()
            medoid_indices.flush()
            medoid_completed.flush()
        if (
            medoids.shape != (len(times), *expected_shape)
            or medoid_indices.shape != (len(times), 2)
            or medoid_completed.shape != (len(times),)
        ):
            raise ValueError("saved medoid arrays do not match the event configuration")

    for index, (time, condition) in enumerate(zip(times, conditions)):
        need_median = not completed[index]
        need_medoid = arguments.add_medoids and not medoid_completed[index]
        if not need_median and not need_medoid:
            continue
        samples = sample_decoded_ensemble(
            model, transform, arguments.draws, batch_size=arguments.batch_size,
            device=device, seed=arguments.seed, support_mask=support,
            conditioning=condition
        )
        completed_labels = []
        if need_median:
            current_median = np.median(samples, axis=0).astype(np.float32)
            current_median[~support] = np.nan
            medians[index] = current_median
            medians.flush()
            completed[index] = True
            completed.flush()
            completed_labels.append("median")
        if need_medoid:
            thresholds = (
                arguments.wic_medoid_threshold,
                arguments.si12_medoid_threshold,
            )
            for channel, threshold in enumerate(thresholds):
                current_summary = summarize_wic_samples(
                    samples[:, channel], support[channel],
                    np.asarray(provenance["grid"]["mlat"]),
                    np.asarray(provenance["grid"]["mlt"]),
                    threshold=threshold
                )
                medoids[index, channel] = current_summary["medoid"]
                medoids[index, channel][~support[channel]] = np.nan
                medoid_indices[index, channel] = current_summary["medoid_index"]
            medoids.flush()
            medoid_indices.flush()
            medoid_completed[index] = True
            medoid_completed.flush()
            completed_labels.append("medoids")
        print(
            f"{index + 1}/{len(times)} {time}: {' and '.join(completed_labels)} complete",
            flush=True
        )

    event_summary = {
        **configuration,
        "completed_minutes": int(completed.sum()),
        "outside_training_range_minutes": int(outside.sum()),
        "output": str(median_path),
    }
    if medoid_completed is not None:
        event_summary.update({
            "completed_medoid_minutes": int(medoid_completed.sum()),
            "medoid_output": str(medoid_path),
            "medoid_index_output": str(medoid_index_path),
        })
    (output / "event_summary.json").write_text(
        json.dumps(event_summary, indent=2) + "\n"
    )
    return configuration


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("supermag", type=Path, help="original one-minute SuperMAG text export")
    parser.add_argument("--start", required=True, help="inclusive UTC event bound")
    parser.add_argument("--stop", required=True, help="inclusive UTC event bound")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--maximum-imf-gap-minutes", type=int, default=0,
        help="linearly interpolate bounded IMF gaps no longer than this; default 0 rejects gaps"
    )
    parser.add_argument("--seed", type=int, default=12104)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--add-medoids", action="store_true",
        help="save separate morphology-medoid WIC and SI12 members"
    )
    parser.add_argument("--wic-medoid-threshold", type=float, default=100.0)
    parser.add_argument("--si12-medoid-threshold", type=float, default=10.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
