"""Compare IMAGE WIC/SI12 with minute-resolved CVAE medians and medoids."""

#%% Imports

import argparse
import csv
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from icreader import load
from polplot import Polarplot


#%% Timestamp matching and polar panels

def nearest_event_minutes(frame_time, event_time, maximum_offset_seconds=30.0):
    frame_time = np.asarray(frame_time, dtype="datetime64[us]")
    event_time = np.asarray(event_time, dtype="datetime64[us]")
    if frame_time.ndim != 1 or event_time.ndim != 1 or not len(event_time):
        raise ValueError("frame and event times must be nonempty one-dimensional arrays")
    right = np.searchsorted(event_time, frame_time).clip(0, len(event_time) - 1)
    left = np.maximum(right - 1, 0)
    use_left = np.abs(frame_time - event_time[left]) <= np.abs(event_time[right] - frame_time)
    nearest = np.where(use_left, left, right)
    offset = (frame_time - event_time[nearest]) / np.timedelta64(1, "s")
    if np.any(np.abs(offset) > maximum_offset_seconds):
        bad = int(np.flatnonzero(np.abs(offset) > maximum_offset_seconds)[0])
        raise ValueError(
            f"frame {bad} is {offset[bad]:g} seconds from its nearest CVAE minute"
        )
    return nearest, offset.astype(float)


def polar_panel(axis, mlat, mlt, values, maximum, title):
    polar = Polarplot(axis, minlat=50)
    finite = np.isfinite(values)
    contour = None
    if finite.any():
        contour = polar.contourf(
            mlat, mlt, values, levels=np.linspace(0, maximum, 17),
            vmin=0, vmax=maximum, cmap="viridis", extend="max"
        )
    else:
        axis.text(.5, .5, "No finite IMAGE data", ha="center", va="center", transform=axis.transAxes)
    polar.writeLTlabels(fontsize=7)
    polar.writeLATlabels(fontsize=6)
    axis.set_title(title, fontsize=10)
    return contour


#%% Export every selected IMAGE frame

def run(arguments):
    event = arguments.event_directory.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty comparison output directory")
    if arguments.frame_start < 0 or arguments.frame_stop < arguments.frame_start:
        raise ValueError("require 0 <= frame-start <= frame-stop")
    if arguments.dpi < 1 or arguments.wic_vmax <= 0 or arguments.si12_vmax <= 0:
        raise ValueError("DPI and colour maxima must be positive")

    summary = json.loads((event / "event_summary.json").read_text())
    completed = np.load(event / "completed_minutes.npy")
    medians = np.load(event / "event_medians.npy", mmap_mode="r")
    medoid_completed = np.load(event / "completed_medoid_minutes.npy")
    medoids = np.load(event / "event_medoids.npy", mmap_mode="r")
    medoid_indices = np.load(event / "event_medoid_indices.npy")
    with np.load(event / "event_conditions.npz") as saved:
        event_data = {name: saved[name] for name in saved.files}
    event_time = event_data["time"].astype("datetime64[us]")
    if (
        summary.get("format") not in (
            "hybrid4_event_medians_v1", "named_event_medians_v2"
        )
        or summary.get("field_names") != ["wic_corrected", "si12"]
        or medians.shape != (len(event_time), 2, 46, 46)
        or completed.shape != (len(event_time),) or not completed.all()
        or medoids.shape != medians.shape
        or medoid_indices.shape != (len(event_time), 2)
        or medoid_completed.shape != (len(event_time),) or not medoid_completed.all()
    ):
        raise ValueError("event output is incomplete or has an unsupported format")

    product = load(arguments.precipitation_file.expanduser().resolve())
    if product.product_type != "precipitation_cs" or tuple(product.shape[1:]) != (46, 46):
        raise ValueError("requires a 46x46 precipitation_cs orbit file")
    stop = arguments.frame_stop + 1
    if stop > product.shape[0]:
        raise ValueError(f"frame-stop must be below {product.shape[0]}")
    frame_index = np.arange(arguments.frame_start, stop)
    frame_time = np.asarray(product.time, dtype="datetime64[us]")[frame_index]
    event_index, offset = nearest_event_minutes(
        frame_time, event_time, arguments.maximum_offset_seconds
    )

    mlat = np.asarray(product.grid.lat, dtype=float)
    mlt = np.mod(np.asarray(product.grid.lon, dtype=float) / 15.0, 24.0)
    if (
        not np.allclose(mlat, event_data["mlat"], atol=1e-5)
        or not np.allclose(mlt, event_data["mlt"], atol=1e-5)
    ):
        raise ValueError("IMAGE and CVAE grids differ")
    wic = np.asarray(product.read("wic_corrected"), dtype=np.float32)[frame_index]
    si12 = np.asarray(product.read("si12"), dtype=np.float32)[frame_index]
    kp = np.asarray(product.read("Kp"), dtype=float)[frame_index]
    output.mkdir(parents=True, exist_ok=True)
    rows, panels = [], []

    for number, source_frame in enumerate(frame_index):
        model_index = event_index[number]
        image_fields = (
            np.where(np.isfinite(wic[number]), np.maximum(wic[number], 0), np.nan),
            np.where(np.isfinite(si12[number]), np.maximum(si12[number], 0), np.nan),
        )
        median_fields = medians[model_index]
        medoid_fields = medoids[model_index]
        figure, axes = plt.subplots(2, 3, figsize=(12.6, 7.8), layout="constrained")
        polar_panel(axes[0, 0], mlat, mlt, image_fields[0], arguments.wic_vmax, "IMAGE")
        wic_contour = polar_panel(axes[0, 1], mlat, mlt, median_fields[0], arguments.wic_vmax, "CVAE median")
        polar_panel(axes[0, 2], mlat, mlt, medoid_fields[0], arguments.wic_vmax, "CVAE medoid")
        polar_panel(axes[1, 0], mlat, mlt, image_fields[1], arguments.si12_vmax, "IMAGE")
        si12_contour = polar_panel(axes[1, 1], mlat, mlt, median_fields[1], arguments.si12_vmax, "CVAE median")
        polar_panel(axes[1, 2], mlat, mlt, medoid_fields[1], arguments.si12_vmax, "CVAE medoid")
        axes[0, 0].set_ylabel("Corrected WIC", fontsize=10)
        axes[1, 0].set_ylabel("SI12", fontsize=10)
        figure.colorbar(wic_contour, ax=axes[0], shrink=.72, label="corrected WIC [counts]")
        figure.colorbar(si12_contour, ax=axes[1], shrink=.72, label="SI12 [counts]")
        condition = (
            f"SML {event_data['sml'][model_index]:g} nT | By {event_data['imf_by'][model_index]:g} nT | "
            f"Bz {event_data['imf_bz'][model_index]:g} nT | tilt {event_data['dipole_tilt'][model_index]:.1f}°"
        )
        identity = (
            f"orbit 261 frame {source_frame} | IMAGE {frame_time[number]} UTC | "
            f"CVAE {event_time[model_index]} UTC | Kp {kp[number]:g}"
        )
        figure.suptitle(identity + "\n" + condition, fontsize=10)
        filename = f"orbit_0261_frame_{source_frame:03d}.png"
        figure.savefig(output / filename, dpi=arguments.dpi)
        plt.close(figure)
        rows.append((
            source_frame, str(frame_time[number]), str(event_time[model_index]), offset[number],
            kp[number], event_data["sml"][model_index], event_data["imf_by"][model_index],
            event_data["imf_bz"][model_index], event_data["dipole_tilt"][model_index],
            bool(event_data["imf_interpolated"][model_index]),
            medoid_indices[model_index, 0], medoid_indices[model_index, 1], filename
        ))
        caption = escape(identity)
        panels.append(f'<a href="{filename}"><img src="{filename}" loading="lazy"><span>{caption}</span></a>')
        if (number + 1) % 25 == 0 or number + 1 == len(frame_index):
            print(f"Saved {number + 1}/{len(frame_index)} comparisons", flush=True)

    with (output / "frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "source_frame", "image_time", "cvae_time", "time_offset_seconds", "kp", "sml",
            "imf_by", "imf_bz", "dipole_tilt", "imf_interpolated",
            "wic_medoid_index", "si12_medoid_index", "filename"
        ))
        writer.writerows(rows)
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Orbit 261 IMAGE/CVAE comparison</title>'
        '<style>body{font:15px sans-serif;margin:20px}main{display:grid;grid-template-columns:'
        'repeat(auto-fill,minmax(440px,1fr));gap:16px}a{color:inherit;text-decoration:none}'
        'img{width:100%;display:block}span{display:block;text-align:center}</style><body>'
        '<h1>Orbit 261: IMAGE observations and hybrid4 CVAE medians/medoids</h1><main>'
        + "\n".join(panels) + '</main></body></html>\n'
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    print(f"Browse {output / 'index.html'}")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_directory", type=Path)
    parser.add_argument("precipitation_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=42)
    parser.add_argument("--frame-stop", type=int, default=185, help="inclusive; default 185")
    parser.add_argument("--maximum-offset-seconds", type=float, default=30.0)
    parser.add_argument("--wic-vmax", type=float, default=1500.0)
    parser.add_argument("--si12-vmax", type=float, default=50.0)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
