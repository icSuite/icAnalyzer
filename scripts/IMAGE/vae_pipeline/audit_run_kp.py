"""Recover Kp for the exact retained images of a saved native conductance VAE."""

#%% Imports and exact frame-to-source matching

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from icreader import load
from icanalyzer.wic_si12.data import _orbit_from_path
from icanalyzer.wic_si12.workflow import load_run_artifact
from scripts.IMAGE.vae_pipeline.validate_conductance_prior import input_evidence
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import _load_heldout_dataset


def matched_positions(source_times, retained_times):
    source_times = np.asarray(source_times, dtype="datetime64[ns]")
    retained_times = np.asarray(retained_times, dtype="datetime64[ns]")
    if np.isnat(source_times).any() or np.any(source_times[1:] <= source_times[:-1]):
        raise ValueError("source frame times must be finite, unique and increasing")
    positions = np.searchsorted(source_times, retained_times)
    if np.any(positions >= len(source_times)) or not np.array_equal(source_times[positions], retained_times):
        raise ValueError("retained image times do not exactly match source times")
    return positions


#%% Reload saved selections; summarize their Kp and temporal coverage

def run(arguments):
    run_directory, output = arguments.run_directory.expanduser(), arguments.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    _, transform, artifact = load_run_artifact(run_directory / "conductance_vae.pt", device="cpu")
    if artifact["artifact_type"] != "conductance_vae_run_v1":
        raise ValueError("requires a native conductance VAE run")
    before = input_evidence(run_directory, artifact)
    source_stats = {path: [Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
                    for split in ("train", "validation", "test") for path in artifact["split_files"][split]}
    rows, summaries, provenance = [], {}, {}
    for split in ("train", "validation", "test"):
        dataset, _ = _load_heldout_dataset(artifact, transform, requested_split=split)
        paths = {_orbit_from_path(Path(path)): path for path in artifact["split_files"][split]}
        for orbit in np.unique(dataset.frame_orbit):
            selected = np.flatnonzero(dataset.frame_orbit == orbit)
            times = np.asarray(dataset.frame_time[selected], dtype="datetime64[ns]")
            path = paths[int(orbit)]
            with load(path) as source:
                positions = matched_positions(source.time, times)
                kp = np.asarray(source.Kp, dtype=float)[positions]
                intervals = np.asarray(source.Kp_interval_start, dtype="datetime64[ns]")[positions]
                provenance[path] = {name: value for name, value in source.attrs.items() if name.startswith("kp_")}
            if np.isnat(intervals).any() or np.any(times < intervals) or np.any(times >= intervals + np.timedelta64(3, "h")):
                raise ValueError("source Kp interval does not contain its retained image time")
            for index, time, value, interval, source_row in zip(selected, times, kp, intervals, positions):
                rows.append({"split": split, "frame_index": int(index), "orbit": int(orbit), "time_UTC": str(time),
                             "Kp": float(value), "Kp_interval_start_UTC": str(interval),
                             "source_row": int(source_row), "source_file": path})
        table = pd.DataFrame([row for row in rows if row["split"] == split])
        finite = table["Kp"].to_numpy()[np.isfinite(table["Kp"])]
        if np.any((finite < 0) | (finite > 9)):
            raise ValueError("source Kp outside physical index range 0–9")
        bins = np.arange(11)
        count, _ = np.histogram(finite, bins=bins)
        # Count unique three-hour intervals separately from correlated IMAGE frames.
        if (table.groupby("Kp_interval_start_UTC")["Kp"].nunique() > 1).any():
            raise ValueError("source files disagree on Kp within a three-hour interval")
        interval_table = table.drop_duplicates("Kp_interval_start_UTC")
        interval_kp = interval_table["Kp"].to_numpy()
        summaries[split] = {
            "retained_frames": len(table), "contributing_orbits": int(table["orbit"].nunique()),
            "missing_Kp_frames": int(len(table) - len(finite)), "range": [float(finite.min()), float(finite.max())],
            "frame_quantiles_p10_p50_p90": np.quantile(finite, [0.1, 0.5, 0.9]).tolist(),
            "integer_bin_counts": {f"{left} <= Kp < {left+1}": int(value) for left, value in enumerate(count)},
            "frame_fraction_Kp_ge3": float(np.mean(finite >= 3)), "frame_fraction_Kp_ge5": float(np.mean(finite >= 5)),
            "unique_three_hour_intervals": len(interval_table),
            "interval_quantiles_p10_p50_p90": np.nanquantile(interval_kp, [0.1, 0.5, 0.9]).tolist(),
            "time_range_UTC": [str(table["time_UTC"].min()), str(table["time_UTC"].max())]
        }
        print(split, json.dumps(summaries[split]), flush=True)
        del dataset
    if input_evidence(run_directory, artifact) != before:
        raise RuntimeError("run/source evidence changed during audit")
    if any([Path(path).stat().st_size, Path(path).stat().st_mtime_ns] != stats for path, stats in source_stats.items()):
        raise RuntimeError("source files in a retained split changed during audit")
    pd.DataFrame(rows).to_csv(output / "retained_frame_kp.csv", index=False)
    report = {"run_directory": str(run_directory), "splits": summaries, "source_kp_provenance": provenance,
              "evidence": before, "source_file_stats_all_splits": source_stats, "interpretation": [
                  "No Kp labels or Kp-range summary are stored in the VAE checkpoint; Kp is recovered from exact saved selections and original source files.",
                  "Each row matches an original IMAGE frame time exactly; no interpolation or nearest-time lookup is introduced.",
                  "Frame counts are correlated in time, not independent observations; unique three-hour intervals are also summarized.",
                  "VAE is unconditional on Kp. Generated members have no assigned Kp; conductance quantiles are not Kp quantiles.",
                  "Comparing with Hardy at chosen Kp values requires activity-population and spatial/statistic matching."
              ]}
    (output / "kp_coverage.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
