"""Plot five selected saved prior draws: WIC above SI12, without axes or titles."""

#%% Imports

import argparse
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


#%% Recover only the requested ensemble members

def load_selected_samples(run_directory, indices):
    """Stream the saved anomaly factor, rather than loading the full ensemble."""

    indices = np.asarray(indices, dtype=int)
    if indices.shape != (5,):
        raise ValueError("supply exactly five zero-based sample indices")
    filename = Path(run_directory).expanduser() / "ensemble_products.npz"
    with np.load(filename, allow_pickle=False) as archive:
        mean = archive["fuv_mean"]
    if mean.ndim != 3 or mean.shape[0] != 2:
        raise ValueError("saved FUV mean must have shape (2, rows, columns)")

    with zipfile.ZipFile(filename) as archive:
        with archive.open("fuv_anomaly_factor.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"unsupported anomaly-array format: {version}")
            if len(shape) != 2 or shape[0] != mean.size or shape[1] < 2:
                raise ValueError("saved anomaly factor does not match the FUV mean")
            if dtype.kind != "f":
                raise ValueError("saved anomaly factor must contain floating-point values")
            sample_count = shape[1]
            if np.any(indices < 0) or np.any(indices >= sample_count):
                raise ValueError(f"sample indices must lie between 0 and {sample_count - 1}")

            selected = np.empty((mean.size, 5), dtype=dtype)
            # Read contiguous slabs, retaining only the requested columns.
            # Normal runner archives are C-order; support F-order as well.
            slab_width = shape[0] if fortran_order else shape[1]
            slab_count = shape[1] if fortran_order else shape[0]
            slabs_per_chunk = max(1, (8 * 1024 * 1024) // (slab_width * dtype.itemsize))
            for start in range(0, slab_count, slabs_per_chunk):
                stop = min(start + slabs_per_chunk, slab_count)
                byte_count = (stop - start) * slab_width * dtype.itemsize
                raw = stream.read(byte_count)
                if len(raw) != byte_count:
                    raise ValueError("saved anomaly array is truncated")
                block = np.frombuffer(raw, dtype=dtype).reshape(stop - start, slab_width)
                if fortran_order:
                    positions = np.flatnonzero((indices >= start) & (indices < stop))
                    selected[:, positions] = block[indices[positions] - start].T
                else:
                    selected[start:stop] = block[:, indices]

    # A = (sample - mean) / sqrt(N - 1), with channels ordered WIC, SI12.
    samples = mean.reshape(1, -1) + np.sqrt(sample_count - 1) * selected.T
    samples = samples.reshape(5, *mean.shape)
    if not np.isfinite(samples).all():
        raise ValueError("selected saved samples contain nonfinite values")
    return samples, sample_count


#%% Presentation layout

def make_figure(samples, cmap="viridis", wic_max=None, si12_max=None):
    """Use one linear counts scale per channel across all five samples."""

    figure, axes = plt.subplots(2, 5, figsize=(12.5, 5))
    for channel, requested_max in enumerate((wic_max, si12_max)):
        maximum = max(float(samples[:, channel].max()), np.finfo(float).eps) if requested_max is None else requested_max
        if not np.isfinite(maximum) or maximum <= 0:
            plt.close(figure)
            raise ValueError("colour-scale maxima must be finite and positive")
        for column in range(5):
            axes[channel, column].imshow(samples[column, channel], origin="lower", cmap=cmap, vmin=0, vmax=maximum, interpolation="nearest")
            axes[channel, column].set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0.02)
    return figure


def run(arguments):
    samples, sample_count = load_selected_samples(arguments.run_directory, arguments.indices)
    figure = make_figure(samples, arguments.cmap, arguments.wic_max, arguments.si12_max)
    output = arguments.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=arguments.dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    print(f"Saved {output}: indices {arguments.indices} from {sample_count} stored draws; WIC top, SI12 bottom.")


#%% Command-line settings

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--indices", nargs=5, type=int, required=True, metavar="INDEX", help="five zero-based saved sample indices, in display order")
    parser.add_argument("--output", type=Path, required=True, help="figure path, e.g. figures/presentation/figure_1.png")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--wic-max", type=float, help="optional common WIC colour-scale maximum in counts")
    parser.add_argument("--si12-max", type=float, help="optional common SI12 colour-scale maximum in counts")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
