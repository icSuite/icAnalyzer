"""Read SuperMAG conditioning data and align it to IMAGE frame times."""

#%% Imports and selected SuperMAG contract

import hashlib
from pathlib import Path
import re

import numpy as np


SUPERMAG_SOURCE_COLUMNS = {
    "SME (nT)": "sme",
    "SML (nT)": "sml",
    "MLAT SML (deg.)": "sml_mlat",
    "MLT SML (hrs)": "sml_mlt",
    "SMU (nT)": "smu",
    "MLAT SMU (deg.)": "smu_mlat",
    "MLT SMU (hrs)": "smu_mlt",
    "No. of Stations SME": "sme_station_count",
    "SMR (nT)": "smr",
    "SMR00 (nT)": "smr_00",
    "SMR06 (nT)": "smr_06",
    "SMR12 (nT)": "smr_12",
    "SMR18 (nT)": "smr_18",
    "No. of stations SMR": "smr_station_count",
    "No. of stations SMR00": "smr_00_station_count",
    "No. of stations SMR06": "smr_06_station_count",
    "No. of stations SMR12": "smr_12_station_count",
    "No. of stations SMR18": "smr_18_station_count",
    "BxGSM (nT)": "bx_gsm",
    "ByGSM (nT)": "by_gsm",
    "BzGSM (nT)": "bz_gsm",
    "VxGSE (nT)": "vx_gse",
    "VyGSE (nT)": "vy_gse",
    "VzGSM (nT)": "vz_gsm",
    "Clock Angle GSM (deg.)": "clock_angle_gsm",
    "Density (#/cm^3)": "density",
    "Dyn. Pres. (nPa)": "dynamic_pressure",
    "Epsilon (GW)": "epsilon",
    "Newell CF (Wb/s)": "newell",
}

SUPERMAG_UNITS = {
    "sme": "nT", "sml": "nT", "sml_mlat": "degrees",
    "sml_mlt": "hours", "smu": "nT", "smu_mlat": "degrees",
    "smu_mlt": "hours", "smr": "nT", "smr_00": "nT",
    "smr_06": "nT", "smr_12": "nT", "smr_18": "nT",
    "bx_gsm": "nT", "by_gsm": "nT", "bz_gsm": "nT",
    "vx_gse": "km/s", "vy_gse": "km/s", "vz_gsm": "km/s",
    "clock_angle_gsm": "degrees", "density": "1/cm^3",
    "dynamic_pressure": "nPa", "epsilon": "GW", "newell": "Wb/s",
}
for _name in SUPERMAG_SOURCE_COLUMNS.values():
    if _name.endswith("station_count"):
        SUPERMAG_UNITS[_name] = "count"

CORE_CONDITIONING_FIELDS = ("by_gsm", "bz_gsm", "sml", "smr", "newell")
TIME_SOURCE_COLUMNS = ("year", "month", "day", "hour", "min", "sec")


#%% File parsing

def _sha256(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _header(filename):
    lines = []
    with Path(filename).open(errors="replace") as stream:
        for number, line in enumerate(stream):
            if line.startswith("<year>\t"):
                names = [value.strip().strip("<>") for value in line.rstrip().split("\t")]
                return number, names, "".join(lines)
            lines.append(line)
    raise ValueError(f"{filename}: SuperMAG column header was not found")


def read_supermag_file(filename):
    """Read selected fields from one SuperMAG yearly text export."""

    import pandas as pd

    filename = Path(filename).expanduser().resolve()
    header_row, available, preamble = _header(filename)
    required = list(TIME_SOURCE_COLUMNS) + list(SUPERMAG_SOURCE_COLUMNS)
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"{filename}: missing SuperMAG columns: {', '.join(missing)}")

    table = pd.read_csv(
        filename, sep="\t", skiprows=header_row,
        usecols=[f"<{name}>" for name in required],
        na_values=(999999, 999999.0), low_memory=False
    )
    table.columns = [name.strip("<>") for name in table.columns]
    numeric_time = table[list(TIME_SOURCE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_time_rows = numeric_time.notna().all(axis=1).to_numpy()
    ignored_footer_rows = 0
    if not np.all(valid_time_rows):
        first_invalid = int(np.flatnonzero(~valid_time_rows)[0])
        if np.any(valid_time_rows[first_invalid:]):
            raise ValueError(f"{filename}: malformed row inside the SuperMAG time series")
        ignored_footer_rows = int(len(table) - first_invalid)
        table = table.iloc[:first_invalid].copy()
        numeric_time = numeric_time.iloc[:first_invalid]
    if table.empty:
        raise ValueError(f"{filename}: SuperMAG export contains no data rows")

    time_columns = numeric_time.rename(
        columns={"min": "minute", "sec": "second"}
    )
    time = pd.to_datetime(time_columns, errors="raise").to_numpy(dtype="datetime64[us]")
    values = {
        canonical: pd.to_numeric(table[source], errors="raise").to_numpy(dtype=np.float32)
        for source, canonical in SUPERMAG_SOURCE_COLUMNS.items()
    }
    if np.isnat(time).any() or np.any(np.diff(time) <= np.timedelta64(0, "us")):
        raise ValueError(f"{filename}: SuperMAG times must be finite, unique and increasing")

    revision = re.search(r"Data revision\s+(\d+)", preamble)
    downloaded = re.search(r"downloaded .* on (.+)\.\s*$", preamble, flags=re.MULTILINE)
    provenance = {
        "filename": str(filename), "sha256": _sha256(filename),
        "bytes": filename.stat().st_size,
        "data_revision": None if revision is None else int(revision.group(1)),
        "downloaded_at": None if downloaded is None else downloaded.group(1),
        "source_declares_data_not_final": "THESE ARE NOT FINAL DATA" in preamble,
        "source_declares_indices_not_iaga_official": "not authorized by IAGA" in preamble,
        "ignored_trailing_nondata_rows": ignored_footer_rows,
        "rows": int(len(time)), "start": str(time[0]), "stop": str(time[-1]),
    }
    return {"time": time, **values}, provenance


def read_supermag_files(filenames):
    """Combine non-overlapping yearly exports into one chronological table."""

    tables, provenance = [], []
    for filename in sorted(Path(name).expanduser().resolve() for name in filenames):
        table, evidence = read_supermag_file(filename)
        tables.append(table)
        provenance.append(evidence)
    if not tables:
        raise ValueError("at least one SuperMAG file is required")
    combined = {
        name: np.concatenate([table[name] for table in tables])
        for name in tables[0]
    }
    order = np.argsort(combined["time"])
    combined = {name: values[order] for name, values in combined.items()}
    if np.any(np.diff(combined["time"]) <= np.timedelta64(0, "us")):
        raise ValueError("SuperMAG yearly exports overlap or contain duplicate times")
    return combined, provenance


#%% IMAGE-time alignment and quantitative coverage

def align_conditioning(supermag, image_times, maximum_offset_seconds=30.0):
    """Nearest-minute alignment with an explicit maximum time separation."""

    reference = np.asarray(supermag["time"], dtype="datetime64[us]")
    query = np.asarray(image_times, dtype="datetime64[us]")
    if reference.ndim != 1 or query.ndim != 1 or len(reference) == 0:
        raise ValueError("SuperMAG and IMAGE times must be nonempty one-dimensional arrays")
    if not np.isfinite(maximum_offset_seconds) or maximum_offset_seconds < 0:
        raise ValueError("maximum time offset must be finite and nonnegative")

    positions = np.searchsorted(reference, query)
    right = np.clip(positions, 0, len(reference) - 1)
    left = np.clip(positions - 1, 0, len(reference) - 1)
    choose_left = np.abs(query - reference[left]) <= np.abs(reference[right] - query)
    nearest = np.where(choose_left, left, right)
    offsets = (query - reference[nearest]) / np.timedelta64(1, "s")
    matched = ~np.isnat(query) & (np.abs(offsets) <= maximum_offset_seconds)

    result = {
        "supermag_time": np.where(matched, reference[nearest], np.datetime64("NaT", "us")),
        "time_offset_seconds": np.where(matched, offsets, np.nan).astype(np.float32),
        "time_matched": matched,
    }
    for name, values in supermag.items():
        if name == "time":
            continue
        aligned = np.full(len(query), np.nan, dtype=np.float32)
        aligned[matched] = values[nearest[matched]]
        result[name] = aligned
    return result


def conditioning_coverage(aligned):
    matched = np.asarray(aligned["time_matched"], dtype=bool)
    denominator = int(matched.sum())
    fields = {}
    for name in SUPERMAG_SOURCE_COLUMNS.values():
        finite = matched & np.isfinite(aligned[name])
        fields[name] = {
            "finite_frames": int(finite.sum()),
            "finite_fraction_of_time_matched": float(finite.sum() / denominator) if denominator else 0.0,
        }
    joint = matched.copy()
    for name in CORE_CONDITIONING_FIELDS:
        joint &= np.isfinite(aligned[name])
    return {
        "frames": int(len(matched)), "time_matched_frames": denominator,
        "time_matched_fraction": float(denominator / len(matched)) if len(matched) else 0.0,
        "core_fields": list(CORE_CONDITIONING_FIELDS),
        "core_joint_finite_frames": int(joint.sum()),
        "core_joint_finite_fraction": float(joint.sum() / denominator) if denominator else 0.0,
        "fields": fields,
    }
