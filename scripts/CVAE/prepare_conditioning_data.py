"""Align SuperMAG parameters to IMAGE Product-2 frame times.

The command writes a compact table and coverage manifest to a new external
directory. It does not train a model or modify an existing CVAE cache.
"""

#%% Imports and defaults

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from icreader import load

from icanalyzer.cvae.conditioning import (
    SUPERMAG_SOURCE_COLUMNS,
    SUPERMAG_UNITS,
    align_conditioning,
    conditioning_coverage,
    read_supermag_files,
)


CONDITIONING_FORMAT = "image_supermag_conditioning_v1"
GRID_ID = "image_apex_130km_46x46_v1"
DEFAULT_SOURCE = Path.home() / "Dropbox/work/data/IMAGE_FUV_new/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic1_si121_si131"


#%% Input resolution and Product-2 timestamps

def supermag_files(directory, years=None):
    directory = Path(directory).expanduser().resolve()
    if years is None:
        files = sorted(directory.glob("year_*.txt"))
    else:
        files = [directory / f"year_{year}.txt" for year in sorted(set(years))]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing SuperMAG files: " + ", ".join(missing))
    if not files:
        raise FileNotFoundError(f"no year_*.txt SuperMAG files in {directory}")
    return files


def load_image_frame_times(source, start, stop, maximum_offset_seconds, orbits=None, product_loader=load):
    source = Path(source).expanduser().resolve()
    files = sorted(source.glob("or_*.nc"))
    if orbits is not None:
        requested = set(orbits)
        files = [path for path in files if int(path.stem[3:]) in requested]
        found = {int(path.stem[3:]) for path in files}
        if found != requested:
            raise FileNotFoundError(f"missing Product-2 orbits: {sorted(requested - found)}")
    if not files:
        raise FileNotFoundError(f"no Product-2 orbit files in {source}")

    margin = np.timedelta64(int(np.ceil(maximum_offset_seconds * 1e6)), "us")
    orbit_values, frame_values, wic_values, time_values = [], [], [], []
    input_frames = 0
    product_evidence = set()
    progress = tqdm(files, desc="Read IMAGE frame times", unit="orbit", dynamic_ncols=True)
    for filename in progress:
        with product_loader(filename) as product:
            if product.product_type != "precipitation_cs" or product.grid_id != GRID_ID:
                raise ValueError(f"{filename}: expected precipitation_cs on {GRID_ID}")
            times = np.asarray(product.time, dtype="datetime64[us]")
            source_wic = np.asarray(product.source_indices["wic"], dtype=np.int32)
            if times.ndim != 1 or source_wic.shape != times.shape or np.isnat(times).any():
                raise ValueError(f"{filename}: invalid Product-2 frame times or WIC source indices")
            product_evidence.add((int(product.schema_version), str(product.software_version)))

        input_frames += len(times)
        selected = (times >= start - margin) & (times <= stop + margin)
        count = int(selected.sum())
        if not count:
            continue
        orbit = int(filename.stem[3:])
        orbit_values.append(np.full(count, orbit, dtype=np.int32))
        frame_values.append(np.flatnonzero(selected).astype(np.int32))
        wic_values.append(source_wic[selected])
        time_values.append(times[selected])

    if not time_values:
        raise ValueError("no IMAGE frames overlap the supplied SuperMAG time span")
    return {
        "orbit": np.concatenate(orbit_values),
        "source_frame": np.concatenate(frame_values),
        "wic_source_index": np.concatenate(wic_values),
        "time": np.concatenate(time_values),
    }, {
        "source_directory": str(source), "orbit_files_scanned": len(files),
        "input_frames": input_frames, "overlapping_frames": int(sum(map(len, time_values))),
        "product_generations": [
            {"schema_version": schema, "software_version": software}
            for schema, software in sorted(product_evidence)
        ],
    }


#%% Preparation

def prepare(arguments, product_loader=load):
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("choose a new conditioning-data directory; existing outputs are never overwritten")

    files = supermag_files(arguments.supermag_dir, arguments.years)
    print(f"Reading {len(files)} SuperMAG yearly file(s)...", flush=True)
    supermag, supermag_provenance = read_supermag_files(files)
    images, image_provenance = load_image_frame_times(
        arguments.source, supermag["time"][0], supermag["time"][-1],
        arguments.maximum_time_offset_seconds, arguments.orbits,
        product_loader=product_loader
    )
    aligned = align_conditioning(
        supermag, images["time"], arguments.maximum_time_offset_seconds
    )
    coverage = conditioning_coverage(aligned)

    output.mkdir(parents=True)
    np.savez_compressed(output / "conditioning.npz", **images, **aligned)
    metadata = {
        "format": CONDITIONING_FORMAT,
        "alignment": {
            "method": "nearest SuperMAG minute",
            "maximum_absolute_offset_seconds": arguments.maximum_time_offset_seconds,
            "long_gap_policy": "unmatched; no interpolation or fill",
        },
        "columns": {
            canonical: {"source": source, "units": SUPERMAG_UNITS[canonical]}
            for source, canonical in SUPERMAG_SOURCE_COLUMNS.items()
        },
        "source_header_anomalies": [
            "SuperMAG labels VxGSE, VyGSE and VzGSM with nT in this export; stored values are treated as velocity in km/s without conversion"
        ],
        "supermag": supermag_provenance,
        "image": image_provenance,
        "coverage": coverage,
        "array_file": "conditioning.npz",
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Prepared {coverage['frames']} IMAGE frames; "
        f"time matched {coverage['time_matched_fraction']:.1%}; "
        f"core joint finite {coverage['core_joint_finite_fraction']:.1%}. "
        f"Output: {output}", flush=True
    )
    return metadata


#%% CLI

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supermag-dir", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orbits", type=int, nargs="+")
    parser.add_argument("--maximum-time-offset-seconds", type=float, default=30.0)
    arguments = parser.parse_args(argv)
    if (not np.isfinite(arguments.maximum_time_offset_seconds) or
            arguments.maximum_time_offset_seconds < 0):
        parser.error("--maximum-time-offset-seconds must be finite and nonnegative")
    return arguments


if __name__ == "__main__":
    prepare(parse_arguments())
