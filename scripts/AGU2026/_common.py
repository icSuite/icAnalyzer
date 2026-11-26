"""Small shared utilities for the chronological AGU 2026 poster scripts."""

#%% Imports

from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np


#%% Saved ensemble access

def ensemble_archive(path):
    """Resolve either an ensemble directory or its NPZ archive."""

    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "ensemble_products.npz"
    if not path.is_file():
        raise FileNotFoundError(f"saved ensemble archive not found: {path}")
    return path


@contextmanager
def mapped_npz_array(archive_path, array_name, workspace=None):
    """Extract one NPZ member and expose it as a read-only memory map."""

    archive_path = ensemble_archive(archive_path)
    temporary = None
    if workspace is None:
        temporary = tempfile.TemporaryDirectory(prefix="icanalyzer_agu2026_")
        workspace = Path(temporary.name)
    else:
        workspace = Path(workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / f"{array_name}.npy"
    if target.exists():
        raise FileExistsError(f"temporary array already exists: {target}")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            member = f"{array_name}.npy"
            if member not in archive.namelist():
                raise KeyError(f"{archive_path} has no {array_name!r} array")
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024**2)
        array = np.load(target, mmap_mode="r", allow_pickle=False)
        yield array
        del array
    finally:
        target.unlink(missing_ok=True)
        if temporary is not None:
            temporary.cleanup()


def load_ensemble_grid(archive_path):
    """Load the small joint-conductance arrays and validate their shapes."""

    archive_path = ensemble_archive(archive_path)
    with np.load(archive_path, allow_pickle=False) as saved:
        required = {"conductance_mean", "conductance_supported", "mlat", "mlt"}
        missing = required.difference(saved.files)
        if missing:
            raise KeyError("saved ensemble lacks: " + ", ".join(sorted(missing)))
        mean = np.asarray(saved["conductance_mean"], dtype=np.float64)
        support = np.asarray(saved["conductance_supported"], dtype=bool)
        mlat = np.asarray(saved["mlat"], dtype=np.float64)
        mlt = np.asarray(saved["mlt"], dtype=np.float64)
        sample_kp = (
            np.asarray(saved["conductance_sample_kp"], dtype=np.float32)
            if "conductance_sample_kp" in saved else np.empty(0, dtype=np.float32)
        )
    if mean.ndim != 3 or mean.shape[0] != 2:
        raise ValueError("conductance mean must have shape (2, rows, columns)")
    if support.shape != mean.shape[1:] or mlat.shape != support.shape or mlt.shape != support.shape:
        raise ValueError("conductance support and magnetic grid must match the mean")
    if not support.any() or not np.isfinite(mean[:, support]).all():
        raise ValueError("supported conductance mean must be nonempty and finite")
    return {
        "archive": archive_path,
        "mean": mean,
        "support": support,
        "mlat": mlat,
        "mlt": mlt,
        "sample_kp": sample_kp,
    }


#%% Plot conventions

def masked(field, support):
    return np.where(support, field, np.nan)


def save_figure(figure, output, stem, dpi=300):
    output = Path(output)
    figure.savefig(output / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
