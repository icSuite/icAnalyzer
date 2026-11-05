"""Apply the saved WIC U-Net to every orbit and frame in precipitation_cs.

Run from any directory with the tConductance environment:
    python scripts/automatic_masking/apply_model2_unet.py

Each orbit gets one NPZ with the original frame times and raw 46x46 binary
model masks. Manual annotations are never read or mixed into these outputs.
The source, trained checkpoint and fresh output directory are explicit.
"""

#%% Source, checkpoint and output paths

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from icreader import load

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.automatic_masking.model2_unet import SmallUNet

FIELDS = ["wic_corrected", "method_quality_weight", "sza", "dza"]

#%% Recover exactly the input scaling and threshold fitted during training

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    if arguments.threads < 1:
        parser.error("--threads must be positive")

    torch.set_num_threads(arguments.threads)
    device = torch.device("cuda" if arguments.device == "auto" and torch.cuda.is_available() else "cpu" if arguments.device == "auto" else arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model_file = arguments.checkpoint.expanduser().resolve()
    checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
    normalization = checkpoint["normalization"]
    if checkpoint["input_fields"] != FIELDS or normalization["fields"] != FIELDS:
        raise ValueError("This script expects the four-field single-frame WIC model")
    if checkpoint["topology_weight"] != 0.1:
        raise ValueError("The saved checkpoint is not the requested weight 0.1 model")
    model = SmallUNet(input_channels=8)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    threshold = checkpoint["threshold"]
    model_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()

    source = arguments.source.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    example_figure = output / "examples.png"


    def model_inputs(channels):
        """The same four transforms, normalization and finite flags as training."""
        values = channels.copy()
        values[:, 0] = np.arcsinh(values[:, 0] / 100)
        values[:, 2] /= 180
        values[:, 3] /= 90
        observed = np.isfinite(values)
        for channel in range(4):
            values[:, channel] = (values[:, channel] - normalization["centre"][channel]) / normalization["scale"][channel]
        values = np.nan_to_num(np.clip(values, -6, 6), nan=0)
        return np.concatenate((values, observed.astype(np.float32)), axis=1)


    files = sorted(source.glob("or_*.nc"))
    if not files:
        raise FileNotFoundError(f"No orbit files found in {source}")
    output.mkdir(parents=True, exist_ok=False)

    #%% Read each source orbit in small batches and save every raw predicted mask

    orbit_rows = []
    examples = []
    seen_nonempty = 0
    rng = np.random.default_rng(12026)
    for number, filename in enumerate(files, start=1):
        with load(filename) as product:
            if tuple(product.shape[1:]) != (46, 46):
                raise ValueError(f"Unexpected grid in {filename}: {product.shape}")
            times = np.asarray(product.time, dtype="datetime64[us]")
            masks = []
            nonempty = 0
            accepted = 0
            observed = 0
            for start in range(0, len(times), 128):
                stop = min(start + 128, len(times))
                channels = np.stack([np.asarray(product.read(field, slice(start, stop)), dtype=np.float32) for field in FIELDS], axis=1)
                finite_wic = np.isfinite(channels[:, 0])
                values = model_inputs(channels)
                with torch.inference_mode():
                    probability = model(torch.from_numpy(values).to(device)).sigmoid().cpu().numpy()
                raw_mask = probability >= threshold
                masks.append(raw_mask)
                nonempty += int(finite_wic.any(axis=(1, 2)).sum())
                accepted += int((raw_mask & finite_wic).sum())
                observed += int(finite_wic.sum())

                # A reproducible visual sample from the entire nonempty corpus.
                for local in np.flatnonzero(finite_wic.any(axis=(1, 2))):
                    seen_nonempty += 1
                    pick = seen_nonempty - 1 if len(examples) < 12 else int(rng.integers(seen_nonempty))
                    if pick < 12:
                        examples[pick:pick + 1] = [(filename.stem, start + int(local), channels[local, 0].copy(), raw_mask[local].copy())]

        masks = np.concatenate(masks) if masks else np.zeros((0, 46, 46), dtype=bool)
        if len(masks) != len(times):
            raise ValueError(f"Frame count mismatch for {filename}")
        np.savez_compressed(output / f"{filename.stem}.npz", time=times, mask=masks)
        orbit_rows.append({"source": str(filename), "orbit": filename.stem, "frames": len(times), "frames_with_wic": nonempty, "finite_wic_pixels": observed, "model_kept_finite_pixels": accepted, "source_bytes": filename.stat().st_size, "source_mtime_ns": filename.stat().st_mtime_ns})
        if number % 50 == 0 or number == len(files):
            print(f"{number}/{len(files)} orbits; {sum(row['frames'] for row in orbit_rows):,} frames", flush=True)

    #%% Mark the collection complete only after every file has an aligned output

    summary = {
        "model": str(model_file), "model_sha256": model_sha,
        "source_directory": str(source), "threshold": threshold,
        "mask_meaning": "Raw thresholded model prediction over the full 46x46 grid; true means keep. No hole filling, component filtering, or manual override. Combine with intrinsic product validity in the VAE.",
        "alignment": "One NPZ per source orbit, same stem and source frame order; time is datetime64[us] and mask has shape (frames, 46, 46)",
        "orbit_files": len(orbit_rows), "frames": sum(row["frames"] for row in orbit_rows),
        "frames_with_finite_wic": sum(row["frames_with_wic"] for row in orbit_rows),
        "orbits": orbit_rows
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")

    #%% Example output on source frames; this is visual inspection, not ground truth

    figure, axes = plt.subplots(12, 3, figsize=(9, 24))
    for axis_row, (orbit, frame, wic, raw_mask) in zip(axes, examples):
        finite = wic[np.isfinite(wic)]
        high = max(1, np.percentile(finite, 98))
        axis_row[0].imshow(wic, origin="lower", vmin=0, vmax=high)
        axis_row[1].imshow(raw_mask, origin="lower", vmin=0, vmax=1, cmap="RdYlGn")
        axis_row[2].imshow(np.where(raw_mask, wic, np.nan), origin="lower", vmin=0, vmax=high)
        axis_row[0].set_ylabel(f"{orbit}\nframe {frame}")
        for axis in axis_row:
            axis.set(xticks=[], yticks=[])
    for axis, title in zip(axes[0], ("Corrected WIC", "Raw mask", "WIC after mask")):
        axis.set_title(title)
    figure.suptitle("Automatic examples from all source orbits; no manual labels or shape cleanup")
    figure.tight_layout(rect=(0, 0, 1, .99))
    figure.savefig(example_figure, dpi=130)
    plt.close(figure)
    print(f"Saved {summary['frames']:,} raw masks in {output}")
    print(f"Example figure: {example_figure}")


if __name__ == "__main__":
    main()
