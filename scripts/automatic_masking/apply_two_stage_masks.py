"""Export Model1 full releases with frozen Model2 masks on all other frames."""

#%% Saved scientific choices: no fitting or manual overrides during inference

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from icreader import load

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.automatic_masking.prepare_model1_training_data import CORE, sha, write_json
from scripts.automatic_masking.model1_cnn import FullFrameClassifier, normalized_core, route_two_stage

def predict_acceptance(model, inputs):
    result = np.empty(len(inputs), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(inputs), 128):
            batch = torch.from_numpy(inputs[start:start + 128])
            result[start:start + len(batch)] = model(batch).sigmoid().numpy()
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, required=True,
        help="completed Model-2 mask directory used as the pixel-mask fallback"
    )
    parser.add_argument(
        "--trial", type=Path, required=True,
        help="saved Model-1 acceptance trial containing its frozen checkpoint"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orbits", type=int, nargs="+")
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args(argv)
    if arguments.threads < 1:
        parser.error("--threads must be positive")
    return arguments


def run(args):
    baseline = args.baseline.expanduser().resolve()
    trial = args.trial.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("Choose a new combined-mask output directory")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    selection = json.loads((trial / "selection_before_test.json").read_text())
    verified = json.loads((trial / "verification.json").read_text())
    accept_file = trial / "seed12027/frame_classifier.pt"
    assert sha(accept_file) == verified["accept_checkpoint_sha256"]
    source_manifest = json.loads((baseline / "manifest.json").read_text())
    assert source_manifest["model_sha256"] == verified["model2_sha256"]
    model2_threshold = float(source_manifest["threshold"])
    accept_saved = torch.load(accept_file, map_location="cpu", weights_only=False)
    normal = accept_saved["normalization"]
    assert normal["fields"] == CORE
    accept_model = FullFrameClassifier(8).eval()
    accept_model.load_state_dict(accept_saved["state_dict"])
    accept_cutoff = selection["full_cutoff"]
    records = source_manifest["orbits"]
    if args.orbits is not None:
        requested = set(args.orbits)
        records = [record for record in records if int(record["orbit"][3:]) in requested]
        if {int(record["orbit"][3:]) for record in records} != requested:
            raise FileNotFoundError("Requested orbit absent from frozen Model2 manifest")
    policy = dict(method="Model1 whole-frame acceptance; Model2 pixel mask fallback",
        accept_checkpoint=str(accept_file), accept_sha256=sha(accept_file),
        model1_selection_sha256=sha(trial / "selection_before_test.json"), accept_cutoff=accept_cutoff,
        selection_field="Only full_cutoff is used; any historical veto fields in an older trial are ignored",
        model1_transfer="Frozen Model1 acceptance checkpoint applied without retraining to the Product-2 generation identified by the Model2 manifest",
        model2_directory=str(baseline), model2_manifest_sha256=sha(baseline / "manifest.json"),
        model2_sha256=source_manifest["model_sha256"], model2_threshold=model2_threshold,
        routing="Accept>=cutoff and nonempty finite WIC -> all intrinsic-valid pixels; every other frame retains Model2 mask exactly",
        annotation_policy="Model1 trained with all-observed-accepted manual Partial frames positive; no manual labels read at inference")
    output.mkdir(parents=True)
    write_json(output / "policy.json", policy)
    policy_sha = sha(output / "policy.json")
    rows, grid = [], None
    available_map = np.zeros((46, 46), dtype=np.int64)
    baseline_map, combined_map = available_map.copy(), available_map.copy()
    started = time.monotonic()

    #%% Stream an orbit at a time; preserve native frames, gaps and Model2 masks

    for number, record in enumerate(records, start=1):
        filename = Path(record["source"])
        stat = filename.stat()
        assert stat.st_size == record["source_bytes"] and stat.st_mtime_ns == record["source_mtime_ns"]
        with load(filename) as product:
            times = np.asarray(product.time, dtype="datetime64[us]")
            images = np.stack([np.asarray(product.read(field), dtype=np.float32) for field in CORE], axis=1)
            native_grid = dict(mlat=np.asarray(product.mlat), mlt=np.asarray(product.mlt))
        assert len(times) == record["frames"] and images.shape == (len(times), 4, 46, 46)
        assert not np.isnat(times).any() and np.all(np.diff(times) > np.timedelta64(0, "us"))
        if grid is None:
            grid = native_grid
        for name in grid:
            np.testing.assert_allclose(native_grid[name], grid[name], rtol=0, atol=1e-6)
        baseline_file = baseline / f"{filename.stem}.npz"
        with np.load(baseline_file) as saved:
            np.testing.assert_array_equal(saved["time"].astype("datetime64[us]"), times)
            current = saved["mask"]
        assert current.shape == (len(times), 46, 46) and current.dtype == bool
        observed = np.isfinite(images[:, 0])
        inputs = normalized_core(images, normal)
        accept_scores = predict_acceptance(accept_model, inputs)
        combined, released = route_two_stage(current, accept_scores, accept_cutoff, observed)
        assert not np.any(current & ~combined)
        np.testing.assert_array_equal(combined[~released], current[~released])
        assert np.all(combined[released]) and not np.any(released & ~observed.any(axis=(1, 2)))
        np.savez_compressed(output / f"{filename.stem}.npz", time=times, mask=combined,
            model1_accept_score=accept_scores, model1_released=released)
        available_map += observed.sum(axis=0)
        baseline_map += (current & observed).sum(axis=0)
        combined_map += (combined & observed).sum(axis=0)
        row = dict(record, model_kept_finite_pixels=int((combined & observed).sum()),
            unet_kept_finite_pixels=int((current & observed).sum()), newly_retained_finite_pixels=int((combined & ~current & observed).sum()),
            model1_released_frames=int(released.sum()),
            baseline_mask_sha256=sha(baseline_file), mask_sha256=sha(output / f"{filename.stem}.npz"))
        rows.append(row)
        if number % 50 == 0 or number == len(records):
            print(f"{number}/{len(records)} orbits; {sum(row['frames'] for row in rows):,} frames; "
                f"{sum(row['model1_released_frames'] for row in rows):,} released; {time.monotonic() - started:.1f}s", flush=True)

    #%% Distinct combined-policy identity and a completion marker for cache use

    np.savez_compressed(output / "wic_coverage.npz", available=available_map, current=baseline_map,
        combined=combined_map, mlat=grid["mlat"], mlt=grid["mlt"])
    regions = dict(dayside=(grid["mlt"] >= 6) & (grid["mlt"] < 18) & (grid["mlat"] >= 60),
        noon=(grid["mlt"] >= 10) & (grid["mlt"] < 14) & (grid["mlat"] >= 60) & (grid["mlat"] < 75))
    summary = dict(model=str(output / "policy.json"), model_sha256=policy_sha,
        source_directory=source_manifest["source_directory"], threshold=model2_threshold,
        model2_sha256=source_manifest["model_sha256"], frame_gate=policy,
        mask_meaning="True means keep; Model1 releases whole frame, otherwise frozen Model2 prediction. Intersect with intrinsic product validity; never fill missing observations.",
        alignment=source_manifest["alignment"], orbit_files=len(rows), frames=sum(row["frames"] for row in rows),
        frames_with_finite_wic=sum(row["frames_with_wic"] for row in rows),
        model1_released_frames=sum(row["model1_released_frames"] for row in rows),
        wic_regions={name:dict(available=int(available_map[pixels].sum()), current=int(baseline_map[pixels].sum()),
            combined=int(combined_map[pixels].sum())) for name, pixels in regions.items()},
        seconds=time.monotonic() - started, orbits=rows)
    assert sha(baseline / "manifest.json") == policy["model2_manifest_sha256"]
    write_json(output / "manifest.json", summary)
    print(json.dumps({key:summary[key] for key in ("orbit_files", "frames", "model1_released_frames", "wic_regions", "seconds")}, indent=2))
    return summary


def main():
    run(parse_arguments())


if __name__ == "__main__":
    main()
