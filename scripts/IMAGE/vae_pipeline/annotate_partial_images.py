"""Interactive, resumable annotation of partially useful IMAGE frames.

The tool keeps the existing one-interval-per-orbit workflow, but permits the
usable interval to extend into partially valid frames. A lasso drawn around
the valid pixels of a sparse set of keyframes is propagated through time.

Run this module as a script; importing it never opens a GUI.
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from icreader import ConductanceImage
from matplotlib.path import Path as MatplotlibPath
from matplotlib.widgets import Button, LassoSelector

try:
    from .annotation_store import (
        clear_boundary,
        delete_keyframe,
        ensure_orbit_record,
        exact_keyframe_mask,
        frame_class,
        keyframe_frames,
        load_document,
        load_good_segment_seeds,
        mark_complete,
        mark_no_usable,
        orbit_key,
        progress_counts,
        propagated_mask,
        restore_record,
        save_document_atomic,
        set_boundary,
        set_keyframe,
    )
except ImportError:
    # Support direct execution: python scripts/IMAGE/vae_pipeline/annotate...
    from annotation_store import (  # type: ignore[no-redef]
        clear_boundary,
        delete_keyframe,
        ensure_orbit_record,
        exact_keyframe_mask,
        frame_class,
        keyframe_frames,
        load_document,
        load_good_segment_seeds,
        mark_complete,
        mark_no_usable,
        orbit_key,
        progress_counts,
        propagated_mask,
        restore_record,
        save_document_atomic,
        set_boundary,
        set_keyframe,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIRECTORY = Path(
    "/home/bing/Dropbox/work/data/IMAGE_FUV/conductance"
)
DEFAULT_OVERVIEW_DIRECTORY = Path(
    "/home/bing/Dropbox/work/projects/icAnalyzer/ncso"
)
DEFAULT_ANNOTATION_FILE = (
    REPOSITORY_ROOT / "data" / "partial_image_annotations.json"
)
DEFAULT_SEGMENT_FILE = REPOSITORY_ROOT / "data" / "good_orbit_segments.csv"

PAGE_SIZE = 36
MASK_SHAPE = (36, 36)
CHANNEL_NAMES = ("WIC", "Hall", "Pedersen", "weight")
CLASS_COLORS = {
    "unreviewed": "#777777",
    "bad": "#d62728",
    "partial": "#ff9f1c",
    "full": "#2ca02c",
}


def discover_orbits(
    overview_directory: Path,
    data_directory: Path,
) -> list[int]:
    """Discover orbit numbers without loading every NetCDF file."""
    overview_pattern = re.compile(r"orbit_(\d+)_\d+\.png$")
    overview_orbits = {
        int(match.group(1))
        for path in overview_directory.glob("orbit_*.png")
        if (match := overview_pattern.match(path.name))
    }
    data_pattern = re.compile(r"or_(\d+)\.nc$")
    data_orbits = {
        int(match.group(1))
        for path in data_directory.glob("or_*.nc")
        if (match := data_pattern.match(path.name))
    }
    orbits = overview_orbits & data_orbits if overview_orbits else data_orbits
    if not orbits:
        raise FileNotFoundError(
            "No loadable orbits were found. The overview directory was "
            f"{overview_directory} and the data directory was {data_directory}"
        )
    return sorted(orbits)


def finite_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.nanpercentile(finite, [2.0, 98.0])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


class MaskEditor:
    """Enlarged four-channel editor for one mask keyframe."""

    def __init__(self, parent: "PartialImageAnnotator", frame: int) -> None:
        self.parent = parent
        self.frame = int(frame)
        self.record = parent.record
        payload = self.record.get("keyframes", {}).get(str(self.frame), {})
        stored_vertices = payload.get("vertices")
        self.vertices: list[list[float]] | None = copy.deepcopy(stored_vertices)
        self.geometry: dict[str, object] | None = copy.deepcopy(
            payload.get("geometry")
        )
        if self.geometry is None and self.vertices is not None:
            self.geometry = {
                "kind": "lasso",
                "vertices": copy.deepcopy(self.vertices),
            }
        elif self.geometry is not None and self.geometry.get("kind") == "line":
            self.geometry = {
                "kind": "line_set",
                "lines": [
                    {
                        "points": copy.deepcopy(self.geometry.get("points")),
                        "valid_side": int(
                            self.geometry.get("valid_side", 1)
                        ),
                    }
                ],
            }
        self.mode = "line"
        self.line_start: tuple[float, float] | None = None
        self.line_artists: list[object] = []

        exact = exact_keyframe_mask(self.record, self.frame)
        if exact is not None:
            self.mask = exact
        elif keyframe_frames(self.record):
            self.mask = propagated_mask(self.record, self.frame)
        else:
            # Starting fully valid makes the undrawn state easy to understand.
            # The first lasso replaces this with the selected valid region.
            self.mask = np.ones(MASK_SHAPE, dtype=bool)

        self.figure, axes = plt.subplots(2, 2, figsize=(11, 10.5))
        self.axes = list(axes.flat)
        self.figure.subplots_adjust(bottom=0.37, wspace=0.08, hspace=0.14)
        self.figure.canvas.manager.set_window_title(
            f"Mask editor — orbit {parent.orbit:04d}, frame {self.frame}"
        )

        self.images = []
        self.overlays = []
        channels = parent.channels
        for axis, channel, name in zip(
            self.axes, channels, CHANNEL_NAMES, strict=True
        ):
            limits = finite_limits(channel[self.frame])
            image = axis.imshow(
                channel[self.frame],
                origin="lower",
                cmap="viridis",
                vmin=limits[0],
                vmax=limits[1],
            )
            overlay = axis.imshow(
                np.ma.masked_where(self.mask, ~self.mask),
                origin="lower",
                cmap="Reds",
                vmin=0,
                vmax=1,
                alpha=0.48,
                interpolation="nearest",
            )
            axis.set_title(name)
            axis.set_xticks([])
            axis.set_yticks([])
            self.images.append(image)
            self.overlays.append(overlay)

        self.lasso = LassoSelector(
            self.axes[0],
            onselect=self._lasso_selected,
            button=1,
            useblit=True,
        )
        self.status = self.figure.text(
            0.02,
            0.315,
            "",
            va="bottom",
            fontsize=9,
        )
        controls = (
            ("Draw next line", 0.04, 0.255, self.use_line_mode),
            ("Lasso valid region", 0.36, 0.255, self.use_lasso_mode),
            ("Flip last line", 0.68, 0.255, self.flip_last_line),
            ("Undo last line", 0.04, 0.195, self.undo_last_line),
            ("Save keyframe", 0.36, 0.195, self.save),
            ("Use active mask", 0.68, 0.195, self.use_propagated),
            ("All valid", 0.04, 0.135, self.all_valid),
            ("All invalid", 0.36, 0.135, self.all_invalid),
            ("Delete keyframe", 0.68, 0.135, self.delete),
            ("Clear line cuts", 0.20, 0.075, self.clear_line_cuts),
            ("Close", 0.52, 0.075, self.close),
        )
        for label, left, bottom, callback in controls:
            self._add_button(label, left, bottom, callback)
        self.figure.canvas.mpl_connect("key_press_event", self._key_pressed)
        self.figure.canvas.mpl_connect(
            "button_press_event", self._line_pressed
        )
        self.figure.canvas.mpl_connect(
            "motion_notify_event", self._line_moved
        )
        self.figure.canvas.mpl_connect(
            "button_release_event", self._line_released
        )
        if self.geometry is not None and self.geometry.get("kind") == "lasso":
            self.use_lasso_mode(
                message="Loaded the saved lasso boundary."
            )
        else:
            self.use_line_mode(
                message=(
                    "Draw the first dividing line on any panel. Additional "
                    "lines further trim the valid region."
                )
            )
        if plt.get_backend().lower() != "agg":
            self.figure.show()
            self._raise_window()

    def _raise_window(self) -> None:
        """Bring a Qt/Tk mask editor in front of the overview window."""
        window = getattr(self.figure.canvas.manager, "window", None)
        if window is None:
            return
        for method_name in ("showNormal", "deiconify", "lift", "raise_"):
            method = getattr(window, method_name, None)
            if callable(method):
                method()
        activate = getattr(window, "activateWindow", None)
        if callable(activate):
            activate()
        focus = getattr(window, "focus_force", None)
        if callable(focus):
            focus()

    def _add_button(
        self,
        label: str,
        left: float,
        bottom: float,
        callback: Callable[[object], None],
    ) -> None:
        axis = self.figure.add_axes((left, bottom, 0.28, 0.042))
        button = Button(axis, label)
        button.on_clicked(callback)
        # Widgets need a live reference or matplotlib may stop dispatching.
        if not hasattr(self, "_buttons"):
            self._buttons: list[Button] = []
        self._buttons.append(button)

    def _lasso_selected(self, vertices: list[tuple[float, float]]) -> None:
        rows, columns = MASK_SHAPE
        xx, yy = np.meshgrid(np.arange(columns), np.arange(rows))
        points = np.column_stack([xx.ravel(), yy.ravel()])
        path = MatplotlibPath(vertices, closed=True)
        self.mask = path.contains_points(points, radius=0.15).reshape(MASK_SHAPE)
        self.vertices = [[float(x), float(y)] for x, y in vertices]
        self.geometry = {
            "kind": "lasso",
            "vertices": copy.deepcopy(self.vertices),
        }
        self._clear_line_artists()
        self._refresh_overlay(
            f"Draft valid area: {100.0 * self.mask.mean():.1f}%. "
            "Press Save keyframe (or S) to keep it."
        )

    def _refresh_overlay(self, message: str) -> None:
        bad_pixels = ~self.mask
        overlay = np.ma.masked_where(~bad_pixels, bad_pixels.astype(float))
        for artist in self.overlays:
            artist.set_data(overlay)
        self.status.set_text(message)
        self.figure.canvas.draw_idle()

    def use_lasso_mode(
        self,
        _event: object = None,
        message: str | None = None,
    ) -> None:
        self.mode = "lasso"
        self.line_start = None
        self.lasso.set_active(True)
        self.axes[0].set_title("WIC — lasso the VALID region")
        self._refresh_overlay(
            message
            or "Lasso mode: draw around the valid pixels on the WIC panel."
        )

    def use_line_mode(
        self,
        _event: object = None,
        message: str | None = None,
    ) -> None:
        self.mode = "line"
        self.line_start = None
        self.lasso.set_active(False)
        self.axes[0].set_title("WIC — draw divider on any panel")
        if (
            self.geometry is not None
            and self.geometry.get("kind") == "line_set"
        ):
            self._apply_line_set()
        self._refresh_overlay(
            message
            or "Draw the next line by dragging on any panel or clicking two "
            "endpoints. It adds another valid-side cut."
        )

    def _line_pressed(self, event: object) -> None:
        if (
            self.mode != "line"
            or getattr(event, "inaxes", None) not in self.axes
            or getattr(event, "button", None) != 1
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return
        current = (float(event.xdata), float(event.ydata))
        if self.line_start is None:
            self.line_start = current
        self._redraw_lines(preview=[self.line_start, current])

    def _line_moved(self, event: object) -> None:
        if (
            self.mode != "line"
            or self.line_start is None
            or getattr(event, "inaxes", None) not in self.axes
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return
        self._redraw_lines(
            preview=[
                self.line_start,
                (float(event.xdata), float(event.ydata)),
            ]
        )

    def _line_released(self, event: object) -> None:
        if self.mode != "line" or self.line_start is None:
            return
        if (
            getattr(event, "inaxes", None) not in self.axes
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            self.line_start = None
            self._redraw_lines()
            self._refresh_overlay(
                "Line cancelled; release it inside an image panel."
            )
            return

        end = (float(event.xdata), float(event.ydata))
        start = self.line_start
        if np.hypot(end[0] - start[0], end[1] - start[1]) < 1.0:
            self._redraw_lines(preview=[start, start])
            self._refresh_overlay(
                "First endpoint set. Click the second endpoint, or drag from "
                "the first point."
            )
            return

        self.line_start = None
        self.vertices = None
        if self.geometry is None or self.geometry.get("kind") != "line_set":
            self.geometry = {"kind": "line_set", "lines": []}
        lines = self.geometry["lines"]
        if not isinstance(lines, list):
            raise ValueError("Saved line-set geometry is invalid")
        lines.append(
            {
                "points": [
                    [float(start[0]), float(start[1])],
                    [float(end[0]), float(end[1])],
                ],
                "valid_side": 1,
            }
        )
        self._apply_line_set()
        self._refresh_overlay(
            f"Draft valid area: {100.0 * self.mask.mean():.1f}%. "
            f"{len(lines)} line cut(s). Flip the last line if needed, or "
            "draw another line."
        )

    def _line_mask(self, line: object) -> np.ndarray:
        if not isinstance(line, dict):
            raise ValueError("Saved line entry is invalid")
        points = line.get("points")
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError("Saved line does not contain two points")
        (x_start, y_start), (x_stop, y_stop) = points
        direction_x = float(x_stop) - float(x_start)
        direction_y = float(y_stop) - float(y_start)
        rows, columns = MASK_SHAPE
        xx, yy = np.meshgrid(np.arange(columns), np.arange(rows))
        signed_side = (
            direction_x * (yy - float(y_start))
            - direction_y * (xx - float(x_start))
        )
        return int(line.get("valid_side", 1)) * signed_side >= 0.0

    def _line_entries(self) -> list[dict[str, object]]:
        if self.geometry is None or self.geometry.get("kind") != "line_set":
            return []
        lines = self.geometry.get("lines")
        if not isinstance(lines, list):
            raise ValueError("Saved line-set geometry is invalid")
        if not all(isinstance(line, dict) for line in lines):
            raise ValueError("Saved line-set contains an invalid line")
        return lines

    def _apply_line_set(self) -> None:
        if self.geometry is None or self.geometry.get("kind") != "line_set":
            return
        self.mask = np.ones(MASK_SHAPE, dtype=bool)
        for line in self._line_entries():
            self.mask &= self._line_mask(line)
        self._redraw_lines()

    def _redraw_lines(
        self,
        preview: list[tuple[float, float]] | None = None,
    ) -> None:
        self._clear_line_artists()
        lines = self._line_entries()
        for index, line in enumerate(lines):
            points = line["points"]
            color = "#00ffff" if index == len(lines) - 1 else "white"
            self._plot_line(points, color=color, linewidth=1.8)
        if preview is not None:
            self._plot_line(preview, color="#ffdd00", linewidth=2.4)
        self.figure.canvas.draw_idle()

    def _plot_line(
        self,
        points: object,
        color: str,
        linewidth: float,
    ) -> None:
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError("Line display requires two points")
        x_coordinates = [float(point[0]) for point in points]
        y_coordinates = [float(point[1]) for point in points]
        for axis in self.axes:
            (artist,) = axis.plot(
                x_coordinates,
                y_coordinates,
                color=color,
                linewidth=linewidth,
                linestyle="--",
                marker="o",
                markersize=4,
            )
            self.line_artists.append(artist)

    def _clear_line_artists(self) -> None:
        for artist in self.line_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self.line_artists.clear()

    def flip_last_line(self, _event: object = None) -> None:
        lines = self._line_entries()
        if not lines:
            self._refresh_overlay("There is no dividing line to flip.")
            return
        current_side = int(lines[-1].get("valid_side", 1))
        lines[-1]["valid_side"] = -current_side
        self._apply_line_set()
        self._refresh_overlay(
            f"Flipped the last of {len(lines)} line cut(s): "
            f"{100.0 * self.mask.mean():.1f}% is valid."
        )

    def undo_last_line(self, _event: object = None) -> None:
        lines = self._line_entries()
        if not lines:
            self._refresh_overlay("There is no dividing line to undo.")
            return
        lines.pop()
        self._apply_line_set()
        self._refresh_overlay(
            f"Removed the last line; {len(lines)} line cut(s) remain."
        )

    def clear_line_cuts(self, _event: object = None) -> None:
        self.mode = "line"
        self.line_start = None
        self.lasso.set_active(False)
        self.axes[0].set_title("WIC — draw divider on any panel")
        self.vertices = None
        self.geometry = {"kind": "line_set", "lines": []}
        self._apply_line_set()
        self._refresh_overlay("Cleared all line cuts; every pixel is valid.")

    def save(self, _event: object = None) -> None:
        snapshot = copy.deepcopy(self.record)
        try:
            set_keyframe(
                self.record,
                self.frame,
                self.mask,
                vertices=self.vertices,
                geometry=self.geometry,
            )
            self.parent.push_undo(snapshot)
            self.parent.save()
            self.parent.refresh(
                f"Saved mask keyframe at frame {self.frame} "
                f"({100.0 * self.mask.mean():.1f}% valid)."
            )
            self._refresh_overlay("Keyframe saved. You may refine it or close.")
        except Exception as error:
            self._refresh_overlay(f"Could not save: {error}")

    def use_propagated(self, _event: object = None) -> None:
        if not keyframe_frames(self.record):
            self._refresh_overlay("There are no other keyframes to propagate.")
            return
        self.mask = propagated_mask(self.record, self.frame)
        self.vertices = None
        self.geometry = None
        self._clear_line_artists()
        self._refresh_overlay("Loaded the currently propagated mask as a draft.")

    def all_valid(self, _event: object = None) -> None:
        self.mask = np.ones(MASK_SHAPE, dtype=bool)
        self.vertices = None
        self.geometry = None
        self._clear_line_artists()
        self._refresh_overlay("All pixels are valid in the draft.")

    def all_invalid(self, _event: object = None) -> None:
        self.mask = np.zeros(MASK_SHAPE, dtype=bool)
        self.vertices = None
        self.geometry = None
        self._clear_line_artists()
        self._refresh_overlay("All pixels are invalid in the draft.")

    def delete(self, _event: object = None) -> None:
        snapshot = copy.deepcopy(self.record)
        if not delete_keyframe(self.record, self.frame):
            self._refresh_overlay("This frame is not an exact keyframe.")
            return
        self.parent.push_undo(snapshot)
        self.parent.save()
        self.parent.refresh(f"Deleted mask keyframe at frame {self.frame}.")
        self._refresh_overlay("Keyframe deleted; the draft remains visible.")

    def close(self, _event: object = None) -> None:
        plt.close(self.figure)

    def _key_pressed(self, event: object) -> None:
        key = getattr(event, "key", "")
        if key == "s":
            self.save()
        elif key in {"q", "escape"}:
            self.close()
        elif key == "d":
            self.delete()
        elif key == "a":
            self.all_valid()
        elif key == "l":
            self.use_lasso_mode()
        elif key == "x":
            self.use_line_mode()
        elif key == "f":
            self.flip_last_line()
        elif key == "z":
            self.undo_last_line()


class PartialImageAnnotator:
    """Main 6x6 orbit viewer with persistent orbit-level annotation state."""

    def __init__(
        self,
        data_directory: Path,
        overview_directory: Path,
        annotation_file: Path,
        segment_file: Path,
        start_orbit: int | None = None,
        start_frame: int | None = None,
    ) -> None:
        self.data_directory = Path(data_directory)
        self.overview_directory = Path(overview_directory)
        self.annotation_file = Path(annotation_file)
        self.document = load_document(self.annotation_file)
        self.seeds = load_good_segment_seeds(segment_file)
        self.orbits = discover_orbits(
            self.overview_directory,
            self.data_directory,
        )
        self.orbit_index = self._initial_orbit_index(start_orbit)
        self.page_start = 0
        self.selected_frame = 0
        self.channel_index = 0
        self.undo_stack: list[dict[str, object]] = []
        self.no_usable_armed = False
        self._editors: list[MaskEditor] = []

        cursor = self.document.get("cursor", {})
        if start_orbit is None and cursor.get("orbit") == self.orbit:
            self.page_start = max(0, int(cursor.get("page_start", 0)))
            self.selected_frame = max(0, int(cursor.get("frame", 0)))
        if start_frame is not None:
            self.selected_frame = max(0, int(start_frame))
            self.page_start = (self.selected_frame // PAGE_SIZE) * PAGE_SIZE

        self._load_orbit()
        self.figure, axes = plt.subplots(6, 6, figsize=(16, 10))
        self.axes = list(axes.flat)
        self.figure.subplots_adjust(
            left=0.025,
            right=0.77,
            bottom=0.04,
            top=0.95,
            wspace=0.08,
            hspace=0.25,
        )
        self.figure.canvas.manager.set_window_title(
            "IMAGE partial-frame annotator"
        )
        self.status = self.figure.text(
            0.785,
            0.325,
            "",
            va="top",
            fontsize=9,
            wrap=True,
        )
        self._add_controls()
        self.figure.canvas.mpl_connect("button_press_event", self._panel_clicked)
        self.figure.canvas.mpl_connect("key_press_event", self._key_pressed)
        self.figure.canvas.mpl_connect("close_event", self._window_closed)
        self.refresh(self._workflow_hint())

    @property
    def orbit(self) -> int:
        return self.orbits[self.orbit_index]

    def _initial_orbit_index(self, start_orbit: int | None) -> int:
        requested = start_orbit
        if requested is None:
            requested = self.document.get("cursor", {}).get("orbit")
        if requested is not None:
            if int(requested) not in self.orbits:
                raise ValueError(f"Orbit {requested} was not found")
            return self.orbits.index(int(requested))

        for index, orbit in enumerate(self.orbits):
            record = self.document["orbits"].get(orbit_key(orbit))
            if record is None or record.get("review_status") not in {
                "complete",
                "no_usable",
            }:
                return index
        return 0

    def _load_orbit(self) -> None:
        source = self.data_directory / f"or_{self.orbit:04d}.nc"
        if not source.exists():
            raise FileNotFoundError(source)
        image = ConductanceImage(str(source))
        wic = np.asarray(image.wic_avg)
        hall = np.asarray(image.H)
        pedersen = np.asarray(image.P)
        weight = np.asarray(image.w)
        if not (
            wic.shape == hall.shape == pedersen.shape == weight.shape
            and wic.ndim == 3
            and wic.shape[1:] == MASK_SHAPE
        ):
            raise ValueError(
                f"Unexpected arrays for orbit {self.orbit}: "
                f"WIC={wic.shape}, H={hall.shape}, P={pedersen.shape}, "
                f"w={weight.shape}"
            )
        self.channels = (wic, hall, pedersen, weight)
        self.n_frames = int(wic.shape[0])
        seed_start, seed_stop = self.seeds.get(self.orbit, (None, None))
        if seed_start is not None and seed_stop is not None:
            seed_start = int(seed_start)
            seed_stop = min(int(seed_stop), self.n_frames - 1)
            if not 0 <= seed_start <= seed_stop:
                seed_start, seed_stop = None, None
        self.record = ensure_orbit_record(
            self.document,
            self.orbit,
            self.n_frames,
            seed_good_start=seed_start,
            seed_good_stop=seed_stop,
        )
        self.selected_frame = min(max(0, self.selected_frame), self.n_frames - 1)
        largest_page = ((self.n_frames - 1) // PAGE_SIZE) * PAGE_SIZE
        self.page_start = min(max(0, self.page_start), largest_page)
        if not self.page_start <= self.selected_frame < self.page_start + PAGE_SIZE:
            self.page_start = (self.selected_frame // PAGE_SIZE) * PAGE_SIZE
        self.undo_stack.clear()
        self.no_usable_armed = False

    def _add_controls(self) -> None:
        controls: list[tuple[str, Callable[[object], None]]] = [
            ("Previous orbit", self.previous_orbit),
            ("Next orbit", self.next_orbit),
            ("Previous page", self.previous_page),
            ("Next page", self.next_page),
            ("Channel: WIC", self.cycle_channel),
            ("Set FIRST usable", self.set_usable_start),
            ("Set FIRST fully good", self.set_full_start),
            ("Set LAST fully good", self.set_full_stop),
            ("Set LAST usable", self.set_usable_stop),
            ("Clear full-good core", self.clear_full_core),
            ("Edit selected mask", self.edit_selected),
            ("Mark orbit complete", self.complete_orbit),
            ("No usable data", self.no_usable),
            ("Undo last edit", self.undo),
            ("Stop and save", self.stop),
        ]
        self.buttons: list[Button] = []
        self.button_by_label: dict[str, Button] = {}
        top = 0.94
        for index, (label, callback) in enumerate(controls):
            axis = self.figure.add_axes(
                (0.79, top - index * 0.043, 0.19, 0.032)
            )
            button = Button(axis, label)
            button.on_clicked(callback)
            self.buttons.append(button)
            self.button_by_label[label] = button
        self.channel_button = self.buttons[4]

    def _workflow_hint(self) -> str:
        full_start = self.record.get("full_good_start")
        full_stop = self.record.get("full_good_stop")
        usable_start = self.record.get("usable_start")
        usable_stop = self.record.get("usable_stop")
        if (
            full_start is not None
            and full_stop is not None
            and usable_start == full_start
            and usable_stop == full_stop
        ):
            return (
                f"Existing fully-good seed: {full_start}–{full_stop}. "
                "For a later partial section, select its LAST useful frame "
                "and click “Set LAST usable”. Do not change FIRST usable."
            )
        return (
            "FIRST/LAST usable are the outer limits of the entire retained "
            "orbit; the fully-good limits sit inside them."
        )

    def push_undo(self, snapshot: dict[str, object]) -> None:
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > 30:
            del self.undo_stack[0]

    def _mutate(
        self,
        operation: Callable[[], None],
        success_message: str,
    ) -> None:
        snapshot = copy.deepcopy(self.record)
        try:
            operation()
        except Exception as error:
            self.refresh(str(error))
            return
        self.push_undo(snapshot)
        self.no_usable_armed = False
        self.save()
        self.refresh(success_message)

    def save(self) -> None:
        self.document["cursor"] = {
            "orbit": self.orbit,
            "page_start": self.page_start,
            "frame": self.selected_frame,
        }
        save_document_atomic(self.document, self.annotation_file)

    def refresh(self, message: str = "") -> None:
        channel = self.channels[self.channel_index]
        page_stop = min(self.page_start + PAGE_SIZE, self.n_frames)
        low, high = finite_limits(channel[self.page_start:page_stop])
        keyframes = set(keyframe_frames(self.record))

        for offset, axis in enumerate(self.axes):
            axis.clear()
            frame = self.page_start + offset
            if frame >= self.n_frames:
                axis.set_axis_off()
                continue

            axis.imshow(
                channel[frame],
                origin="lower",
                cmap="viridis",
                vmin=low,
                vmax=high,
                interpolation="nearest",
            )
            classification = frame_class(self.record, frame)
            if classification == "partial" and keyframes:
                mask = propagated_mask(self.record, frame)
                bad = ~mask
                axis.imshow(
                    np.ma.masked_where(~bad, bad.astype(float)),
                    origin="lower",
                    cmap="Reds",
                    vmin=0,
                    vmax=1,
                    alpha=0.38,
                    interpolation="nearest",
                )
            color = CLASS_COLORS[classification]
            width = 3.0 if frame == self.selected_frame else 1.8
            if frame == self.selected_frame:
                color = "#1677ff"
            for spine in axis.spines.values():
                spine.set_color(color)
                spine.set_linewidth(width)
            marker = " ★" if frame in keyframes else ""
            axis.set_title(f"{frame}{marker}", fontsize=8, pad=1)
            axis.set_xticks([])
            axis.set_yticks([])

        boundaries = ", ".join(
            f"{name.replace('_', ' ')}={self.record.get(name)}"
            for name in (
                "usable_start",
                "full_good_start",
                "full_good_stop",
                "usable_stop",
            )
        )
        counts = progress_counts(self.document, self.orbits)
        progress = (
            f"{counts.get('complete', 0)} complete, "
            f"{counts.get('no_usable', 0)} no-usable, "
            f"{counts.get('in_progress', 0)} in progress, "
            f"{counts.get('unreviewed', 0)} untouched"
        )
        self.status.set_text(
            f"Orbit {self.orbit:04d} ({self.orbit_index + 1}/"
            f"{len(self.orbits)})\n"
            f"Frame {self.selected_frame}/{self.n_frames - 1}; "
            f"page {self.page_start}–{page_stop - 1}\n"
            f"Channel: {CHANNEL_NAMES[self.channel_index]}\n"
            f"Status: {self.record['review_status']}\n"
            f"{boundaries}\n"
            f"Mask keyframes: {keyframe_frames(self.record) or 'none'}\n\n"
            f"All orbits: {progress}\n\n"
            f"{message}\n\n"
            f"Autosaved: {self.annotation_file.name}\n"
            "Double-click an orange frame to edit its valid mask.\n"
            "For partial data after a good core, change only LAST usable."
        )
        self.figure.canvas.draw_idle()

    def _panel_clicked(self, event: object) -> None:
        if getattr(event, "inaxes", None) not in self.axes:
            return
        offset = self.axes.index(event.inaxes)
        frame = self.page_start + offset
        if frame >= self.n_frames:
            return
        self.selected_frame = frame
        self.save()
        self.refresh(f"Selected frame {frame}.")
        if getattr(event, "dblclick", False):
            self.edit_selected()

    def _move_orbit(self, change: int) -> None:
        target = self.orbit_index + change
        if not 0 <= target < len(self.orbits):
            self.refresh("There is no orbit in that direction.")
            return
        self._close_editors()
        self.save()
        self.orbit_index = target
        self.page_start = 0
        self.selected_frame = 0
        try:
            self._load_orbit()
        except Exception as error:
            # Restore the prior orbit if its neighbor cannot be loaded.
            self.orbit_index -= change
            self._load_orbit()
            self.refresh(f"Could not load neighboring orbit: {error}")
            return
        self.save()
        self.refresh(self._workflow_hint())

    def previous_orbit(self, _event: object = None) -> None:
        self._move_orbit(-1)

    def next_orbit(self, _event: object = None) -> None:
        self._move_orbit(1)

    def _move_page(self, change: int) -> None:
        new_start = self.page_start + change * PAGE_SIZE
        largest_page = ((self.n_frames - 1) // PAGE_SIZE) * PAGE_SIZE
        new_start = min(max(0, new_start), largest_page)
        if new_start == self.page_start:
            self.refresh("There is no page in that direction.")
            return
        self.page_start = new_start
        self.selected_frame = new_start
        self.save()
        self.refresh("Changed page.")

    def previous_page(self, _event: object = None) -> None:
        self._move_page(-1)

    def next_page(self, _event: object = None) -> None:
        self._move_page(1)

    def cycle_channel(self, _event: object = None) -> None:
        self.channel_index = (self.channel_index + 1) % len(self.channels)
        self.channel_button.label.set_text(
            f"Channel: {CHANNEL_NAMES[self.channel_index]}"
        )
        self.refresh("Changed display channel.")

    def set_usable_start(self, _event: object = None) -> None:
        full_start = self.record.get("full_good_start")
        if full_start is not None and self.selected_frame > full_start:
            self.refresh(
                f"FIRST usable cannot be {self.selected_frame} because the "
                f"fully-good data already begins at {full_start}. To retain "
                "partial frames after that good section, select the final "
                "partial frame and click “Set LAST usable”."
            )
            return
        self._mutate(
            lambda: set_boundary(
                self.record, "usable_start", self.selected_frame
            ),
            f"The entire usable envelope starts at {self.selected_frame}.",
        )

    def set_full_start(self, _event: object = None) -> None:
        self._mutate(
            lambda: set_boundary(
                self.record, "full_good_start", self.selected_frame
            ),
            f"Fully good interval starts at {self.selected_frame}.",
        )

    def set_full_stop(self, _event: object = None) -> None:
        self._mutate(
            lambda: set_boundary(
                self.record, "full_good_stop", self.selected_frame
            ),
            f"Fully good interval stops at {self.selected_frame}.",
        )

    def set_usable_stop(self, _event: object = None) -> None:
        full_stop = self.record.get("full_good_stop")
        if full_stop is not None and self.selected_frame < full_stop:
            self.refresh(
                f"LAST usable cannot be {self.selected_frame} because the "
                f"fully-good data continues through {full_stop}."
            )
            return
        self._mutate(
            lambda: set_boundary(
                self.record, "usable_stop", self.selected_frame
            ),
            (
                f"The entire usable envelope now ends at "
                f"{self.selected_frame}; intervening frames are partial."
            ),
        )

    def clear_full_core(self, _event: object = None) -> None:
        def clear() -> None:
            clear_boundary(self.record, "full_good_start")
            clear_boundary(self.record, "full_good_stop")

        self._mutate(clear, "Cleared the fully good core.")

    def edit_selected(self, _event: object = None) -> None:
        classification = frame_class(self.record, self.selected_frame)
        if classification != "partial":
            self.refresh(
                "Mask keyframes belong in an orange partial interval. "
                "Set the four interval boundaries first."
            )
            return
        try:
            editor = MaskEditor(self, self.selected_frame)
        except Exception as error:
            self.refresh(
                f"Could not open mask editor: {type(error).__name__}: {error}"
            )
            return
        self._editors.append(editor)
        self.refresh(
            f"Opened mask editor for frame {self.selected_frame}. "
            "It should appear as a separate foreground window."
        )

    def complete_orbit(self, _event: object = None) -> None:
        def complete() -> None:
            mark_complete(self.record)

        snapshot = copy.deepcopy(self.record)
        try:
            complete()
        except Exception as error:
            self.refresh(str(error))
            return
        self.push_undo(snapshot)
        self.save()
        self.refresh("Orbit marked complete. Moving to the next orbit.")
        if self.orbit_index + 1 < len(self.orbits):
            self._move_orbit(1)

    def no_usable(self, _event: object = None) -> None:
        if not self.no_usable_armed:
            self.no_usable_armed = True
            self.refresh(
                "Click “No usable data” again to clear this orbit's "
                "boundaries and masks."
            )
            return
        self._mutate(
            lambda: mark_no_usable(self.record),
            "Orbit marked as containing no usable data.",
        )

    def undo(self, _event: object = None) -> None:
        if not self.undo_stack:
            self.refresh("There is nothing to undo for this orbit.")
            return
        restore_record(self.record, self.undo_stack.pop())
        self.save()
        self.refresh("Restored the previous annotation state.")

    def stop(self, _event: object = None) -> None:
        self._close_editors()
        self.save()
        plt.close(self.figure)

    def _close_editors(self) -> None:
        for editor in self._editors:
            plt.close(editor.figure)
        self._editors.clear()

    def _window_closed(self, _event: object) -> None:
        self.save()

    def _key_pressed(self, event: object) -> None:
        key = getattr(event, "key", "")
        if key == "left":
            self.previous_page()
        elif key == "right":
            self.next_page()
        elif key == "up":
            self.previous_orbit()
        elif key == "down":
            self.next_orbit()
        elif key == "e":
            self.edit_selected()
        elif key == "s":
            self.save()
            self.refresh("Saved.")
        elif key in {"q", "escape"}:
            self.stop()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate IMAGE orbits with a usable interval and sparse spatial "
            "mask keyframes. Work is saved automatically and resumed later."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
        help=f"Directory containing or_####.nc (default: {DEFAULT_DATA_DIRECTORY})",
    )
    parser.add_argument(
        "--overview-dir",
        type=Path,
        default=DEFAULT_OVERVIEW_DIRECTORY,
        help=(
            "Directory containing orbit_####_###.png, used to discover orbit "
            f"numbers (default: {DEFAULT_OVERVIEW_DIRECTORY})"
        ),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATION_FILE,
        help=f"Persistent annotation JSON (default: {DEFAULT_ANNOTATION_FILE})",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=DEFAULT_SEGMENT_FILE,
        help=(
            "Existing good-interval CSV used only as seeds "
            f"(default: {DEFAULT_SEGMENT_FILE})"
        ),
    )
    parser.add_argument(
        "--start-orbit",
        type=int,
        help="Open this orbit instead of the saved cursor",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        help="Open this frame within the selected or resumed orbit",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    annotator = PartialImageAnnotator(
        data_directory=arguments.data_dir,
        overview_directory=arguments.overview_dir,
        annotation_file=arguments.annotations,
        segment_file=arguments.segments,
        start_orbit=arguments.start_orbit,
        start_frame=arguments.start_frame,
    )
    try:
        plt.show()
    finally:
        annotator.save()


if __name__ == "__main__":
    main()
