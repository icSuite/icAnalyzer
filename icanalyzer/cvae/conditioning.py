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
CONDITIONING_FORMAT = "image_supermag_conditioning_v1"
CONDITIONING_SOURCE_FIELDS = (
    "sml", "smr", "by_gsm", "bz_gsm", "vx_gse",
    "clock_angle_gsm", "dynamic_pressure", "newell",
)
CONDITIONING_FEATURE_NAMES = (
    "sml", "smr", "by_gsm", "bz_gsm", "vx_gse", "dynamic_pressure",
    "newell", "kp", "day_of_year_sin", "day_of_year_cos", "dipole_tilt",
    "clock_angle_sin", "clock_angle_cos",
)
CONCEPTUAL_CONDITIONING_FEATURES = {
    "sml": ("sml",),
    "smr": ("smr",),
    "imf_by": ("by_gsm",),
    "imf_bz": ("bz_gsm",),
    "vx": ("vx_gse",),
    "clock_angle": ("clock_angle_sin", "clock_angle_cos"),
    "dynamic_pressure": ("dynamic_pressure",),
    "newell": ("newell",),
    "kp": ("kp",),
    "day_of_year": ("day_of_year_sin", "day_of_year_cos"),
    "dipole_tilt": ("dipole_tilt",),
}


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


#%% Exact CVAE feature contract

def conditioning_time_features(time):
    """Return cyclic day-of-year and centered-dipole tilt at epoch 2001.5."""

    time = np.asarray(time, dtype="datetime64[us]")
    year_start = time.astype("datetime64[Y]")
    next_year = year_start + np.timedelta64(1, "Y")
    day_start = time.astype("datetime64[D]")
    year = year_start.astype(int) + 1970
    day_of_year = (
        day_start - year_start.astype("datetime64[D]")
    ).astype(int) + 1
    seconds = (time - day_start) / np.timedelta64(1, "s")
    year_fraction = (
        (time - year_start.astype("datetime64[us]"))
        / (next_year.astype("datetime64[us]") - year_start.astype("datetime64[us]"))
    )
    day_phase = 2 * np.pi * year_fraction

    if np.min(year) <= 1600 or np.max(year) >= 2100:
        raise ValueError("dipole tilt is implemented only for years 1601--2099")
    year_from_2000 = year - 2000
    leap_years = np.floor((year - 1601) / 4) - 99
    centuries = 3 - np.floor((year - 1601) / 100)
    leap_years = np.where(year <= 1900, leap_years + centuries, leap_years)
    longitude_0 = (
        -79.549 - .238699 * (year_from_2000 - 4 * leap_years)
        + .0308514 * leap_years
    )
    anomaly_0 = (
        -2.472 - .2558905 * (year_from_2000 - 4 * leap_years)
        - .0379617 * leap_years
    )
    days = seconds / 86400 - 1.5 + day_of_year
    mean_longitude = longitude_0 + .9856474 * days
    mean_anomaly = np.deg2rad(anomaly_0 + .9856003 * days)
    ecliptic_longitude = np.deg2rad(
        mean_longitude + 1.915 * np.sin(mean_anomaly)
        + .020 * np.sin(2 * mean_anomaly)
    )
    days_from_2000 = days + 365 * year_from_2000 + leap_years
    obliquity = np.deg2rad(23.439 - 4e-7 * days_from_2000)
    right_ascension = np.arctan2(
        np.cos(obliquity) * np.sin(ecliptic_longitude),
        np.cos(ecliptic_longitude)
    )
    declination = np.arcsin(
        np.sin(obliquity) * np.sin(ecliptic_longitude)
    )
    equation_of_time = mean_longitude - np.rad2deg(right_ascension)
    equation_of_time -= 360 * np.round(equation_of_time / 360)
    subsolar_longitude = np.deg2rad(
        180 - (seconds / 240 + equation_of_time)
    )
    sun = np.column_stack((
        np.cos(declination) * np.cos(subsolar_longitude),
        np.cos(declination) * np.sin(subsolar_longitude),
        np.sin(declination),
    ))
    dipole_axis_2001_5 = np.asarray([.05683683, -.17125208, .98358645])
    return {
        "day_of_year_sin": np.sin(day_phase),
        "day_of_year_cos": np.cos(day_phase),
        "dipole_tilt": np.rad2deg(np.arcsin(sun @ dipole_axis_2001_5)),
    }


def conditioning_feature_matrix(condition, positions, time, kp):
    """Build the exact 13-column representation of 11 conceptual features."""

    features = {
        name: condition[name][positions] for name in CONDITIONING_SOURCE_FIELDS
    }
    features.update(conditioning_time_features(time))
    features.update({
        "kp": np.asarray(kp),
        "clock_angle_sin": np.sin(np.deg2rad(features["clock_angle_gsm"])),
        "clock_angle_cos": np.cos(np.deg2rad(features["clock_angle_gsm"])),
    })
    return np.column_stack([
        features[name] for name in CONDITIONING_FEATURE_NAMES
    ]).astype(np.float32)


def load_prepared_conditioning(directory):
    """Load a completed conditioning table without weakening its contract."""

    import json

    directory = Path(directory).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != CONDITIONING_FORMAT:
        raise ValueError(f"expected completed {CONDITIONING_FORMAT} data")
    with np.load(directory / manifest["array_file"], allow_pickle=False) as saved:
        condition = {name: saved[name].copy() for name in saved.files}
    required = {
        "orbit", "source_frame", "time", *CONDITIONING_SOURCE_FIELDS
    }
    missing = sorted(required - set(condition))
    if missing:
        raise ValueError("conditioning table lacks fields: " + ", ".join(missing))
    return condition, manifest, _sha256(manifest_path)


def align_conditioning_features(
    conditioning_directory, orbit, source_frame, time, kp
):
    """Exact orbit/frame/time join from one cache split to named conditions."""

    condition, manifest, manifest_sha256 = load_prepared_conditioning(
        conditioning_directory
    )
    condition_key = (
        condition["orbit"].astype(np.int64) * 1_000_000
        + condition["source_frame"].astype(np.int64)
    )
    order = np.argsort(condition_key)
    sorted_key = condition_key[order]
    if np.any(np.diff(sorted_key) == 0):
        raise ValueError("conditioning table contains duplicate orbit/frame keys")

    orbit = np.asarray(orbit, dtype=np.int64)
    source_frame = np.asarray(source_frame, dtype=np.int64)
    time = np.asarray(time, dtype="datetime64[us]")
    kp = np.asarray(kp, dtype=np.float32)
    key = orbit * 1_000_000 + source_frame
    positions = np.searchsorted(sorted_key, key)
    bounded = np.minimum(positions, len(sorted_key) - 1)
    if np.any(positions == len(sorted_key)) or np.any(sorted_key[bounded] != key):
        raise ValueError("some cache frames are absent from conditioning data")
    positions = order[positions]
    if not np.array_equal(
        condition["time"][positions].astype("datetime64[us]"), time
    ):
        raise ValueError("conditioning and cache timestamps differ")
    matrix = conditioning_feature_matrix(condition, positions, time, kp)
    return matrix, manifest, manifest_sha256


def fit_conditioning_normalization(values, embedding_dim=32):
    """Fit named feature standardization on finite training frames only."""

    values = np.asarray(values, dtype=np.float64)
    embedding_dim = int(embedding_dim)
    if values.ndim != 2 or values.shape[1] != len(CONDITIONING_FEATURE_NAMES):
        raise ValueError("conditioning matrix does not match the named feature contract")
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("conditioning normalization requires finite training rows")
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0) or embedding_dim < 1:
        raise ValueError("conditioning features need positive spread and embedding size")
    return {
        "kind": "named_standardization_v1",
        "feature_names": list(CONDITIONING_FEATURE_NAMES),
        "conceptual_features": {
            name: list(columns)
            for name, columns in CONCEPTUAL_CONDITIONING_FEATURES.items()
        },
        "center": center.tolist(), "scale": scale.tolist(),
        "training_min": values.min(axis=0).tolist(),
        "training_max": values.max(axis=0).tolist(),
        "embedding_dim": embedding_dim,
    }


def conditioning_vector_from_record(record):
    """Convert 11 named physical conditions to the trained 13-column layout."""

    required = (
        "sml", "smr", "imf_by", "imf_bz", "vx", "clock_angle",
        "dynamic_pressure", "newell", "kp", "day_of_year", "dipole_tilt",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError("condition record lacks: " + ", ".join(missing))
    numbers = {name: float(record[name]) for name in required}
    values = np.asarray([numbers[name] for name in required], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("all requested conditions must be finite")
    if not 0 <= numbers["kp"] <= 9:
        raise ValueError("requested Kp must lie within 0--9")
    if not 1 <= numbers["day_of_year"] <= 366:
        raise ValueError("requested day of year must lie within 1--366")
    day_phase = 2 * np.pi * (numbers["day_of_year"] - 1) / 365.25
    clock_phase = np.deg2rad(numbers["clock_angle"])
    columns = {
        "sml": numbers["sml"], "smr": numbers["smr"],
        "by_gsm": numbers["imf_by"], "bz_gsm": numbers["imf_bz"],
        "vx_gse": numbers["vx"],
        "dynamic_pressure": numbers["dynamic_pressure"],
        "newell": numbers["newell"], "kp": numbers["kp"],
        "day_of_year_sin": np.sin(day_phase),
        "day_of_year_cos": np.cos(day_phase),
        "dipole_tilt": numbers["dipole_tilt"],
        "clock_angle_sin": np.sin(clock_phase),
        "clock_angle_cos": np.cos(clock_phase),
    }
    return np.asarray(
        [columns[name] for name in CONDITIONING_FEATURE_NAMES],
        dtype=np.float32
    )
