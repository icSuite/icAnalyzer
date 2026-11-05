"""Browse all held-out reference pairs, with IDs matching figures 4 and 5."""

#%% Imports

import argparse
import csv
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from icanalyzer.wic_si12.data import ChannelTransform
from icanalyzer.wic_si12.physics import conductance_from_corrected_wic
from scripts.IMAGE.vae_pipeline.plot_conductance_reconstruction import validate_conductance_propagation
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import _load_heldout_dataset


#%% One reference pair; original missing pixels stay blank

def make_figure(values, valid, field_names, units, identity, maxima):
    figure, axes = plt.subplots(1, 2, figsize=(7, 3.5), layout="constrained")
    limits = []
    for channel, axis in enumerate(axes):
        maximum = maxima[channel]
        if maximum is None:
            maximum = max(float(values[channel][valid].max()), np.finfo(float).eps) if valid.any() else 1.0
        limits.append(maximum)
        image = axis.imshow(np.where(valid, values[channel], np.nan), origin="lower", cmap="viridis", vmin=0, vmax=maximum, interpolation="nearest")
        axis.set_axis_off()
        axis.set_title(field_names[channel])
        figure.colorbar(image, ax=axis, shrink=0.8, label=units)
    figure.suptitle(identity)
    return figure, limits


#%% Load the saved selection once, then export every frame without inference

def run(arguments):
    if arguments.start < 0 or (arguments.stop is not None and arguments.stop <= arguments.start):
        raise ValueError("require 0 <= start < stop; stop is exclusive")
    if arguments.dpi <= 0:
        raise ValueError("DPI must be positive")
    maxima = (None, None) if arguments.maxima is None else arguments.maxima
    if any(value is not None and (not np.isfinite(value) or value <= 0) for value in maxima):
        raise ValueError("colour-scale maxima must be finite and positive")
    output = arguments.output.expanduser()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty output directory; existing plots are preserved")
    run_directory = arguments.run_directory.expanduser()
    native = (run_directory / "conductance_vae.pt").is_file()
    if arguments.fields == "conductance":
        native = validate_conductance_propagation(run_directory)
        field_names, units = ("Hall", "Pedersen"), "S"
        description = "Native conductance_cs retrieval, not independent conductance truth." if native else "Corrected grid-WIC converted at fixed 2 keV; not native conductance_cs or independent conductance truth."
    else:
        if native:
            raise ValueError("native conductance runs do not contain WIC/SI12; --fields fuv is not applicable")
        field_names, units = ("WIC", "SI12"), "counts"
        description = "Held-out corrected WIC and SI12 inputs, not latent samples or reconstructions."
    artifact = torch.load(run_directory / ("conductance_vae.pt" if native else "wic_si12_vae.pt"), map_location="cpu", weights_only=False)
    if native != (artifact.get("artifact_type") == "conductance_vae_run_v1"):
        raise ValueError("run summary and artifact disagree on native conductance versus FUV")
    dataset, split = _load_heldout_dataset(artifact, ChannelTransform.from_dict(artifact["transform"]), arguments.split)
    del artifact
    stop = len(dataset) if arguments.stop is None else arguments.stop
    if not arguments.start < stop <= len(dataset):
        raise ValueError(f"range must lie inside the {len(dataset)} held-out {split} frames")
    output.mkdir(parents=True, exist_ok=True)
    rows, panels = [], []

    for frame_index in range(arguments.start, stop):
        if arguments.fields == "conductance" and not native:
            converted = conductance_from_corrected_wic(dataset.values[frame_index, 0], include_intermediates=False)
            values = np.stack((converted["H"], converted["P"]))
        else:
            values = dataset.values[frame_index]
        valid = dataset.common_valid[frame_index, 0] & np.all(np.isfinite(values), axis=0)
        orbit = int(dataset.frame_orbit[frame_index])
        time = str(dataset.frame_time[frame_index])
        identity = f"{split} frame {frame_index} | orbit {orbit} | {time}"
        filename = f"{split}_frame_{frame_index:06d}.png"
        figure, limits = make_figure(values, valid, field_names, units, identity, maxima)
        figure.savefig(output / filename, dpi=arguments.dpi)
        plt.close(figure)
        rows.append((frame_index, orbit, time, int(valid.sum()), *limits, filename))
        panels.append(f'<a href="{filename}"><img src="{filename}" loading="lazy" alt="{escape(identity, quote=True)}"><span>{escape(identity)}</span></a>')
        count = frame_index - arguments.start + 1
        if count % 50 == 0 or frame_index == stop - 1:
            print(f"Saved {count}/{stop - arguments.start} {split} pairs", flush=True)

    # CSV preserves the full-split IDs, also when exporting only a subrange.
    with (output / "frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame_index", "orbit", "time", "valid_cells", field_names[0].lower() + "_scale_max", field_names[1].lower() + "_scale_max", "filename"))
        writer.writerows(rows)
    scales = "Independent per-frame colour maxima; use --maxima for fixed scales." if arguments.maxima is None else f"Fixed colour maxima: {maxima[0]:g}, {maxima[1]:g} {units}."
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Held-out samples</title><style>body{font:16px sans-serif;margin:20px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}a{color:inherit;text-decoration:none}img{width:100%;display:block}span{display:block;text-align:center}</style>'
    mask_description = "Common Hall/Pedersen retrieval validity" if native else "Common WIC/SI12 validity"
    page += f"<body><h1>{split}: {field_names[0]} / {field_names[1]}</h1><p>{escape(str(run_directory.resolve()))}</p><p>{escape(description)} {mask_description}; gaps are blank.</p><p>{escape(scales)} Click a pair to view its full image.</p><main>"
    page += "\n".join(panels) + "</main></body></html>\n"
    (output / "index.html").write_text(page, encoding="utf-8")
    print(f"Browse {output / 'index.html'}; {stop - arguments.start} pairs, IDs {arguments.start}--{stop - 1}.")
    return {"split": split, "count": stop - arguments.start, "total_frames": len(dataset), "fields": arguments.fields}


#%% All test frames by default; optional subrange does not renumber them

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="empty directory for paired PNGs, frames.csv and index.html")
    parser.add_argument("--fields", choices=("conductance", "fuv"), default="conductance", help="Hall/Pedersen by default; fuv plots corrected WIC/SI12")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--start", type=int, default=0, help="first full-split frame ID, inclusive")
    parser.add_argument("--stop", type=int, help="last full-split frame ID, exclusive; default all")
    parser.add_argument("--maxima", type=float, nargs=2, metavar=("FIRST", "SECOND"), help="fixed linear colour maxima for Hall/Pedersen (S) or WIC/SI12 (counts); default per-frame")
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
