"""Shared disk-backed cache access for Hall/Pedersen and WIC CVAEs."""

#%% Imports

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import ChannelTransform


#%% Completed-cache identity and evidence

AUTOMATIC_CONDUCTANCE_CACHE_FORMAT = "automatic_conductance_cache_v1"
WIC_SI12_CACHE_FORMAT = "wic_si12_cvae_cache_v1"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cache(cache_directory, expected_format=AUTOMATIC_CONDUCTANCE_CACHE_FORMAT):
    manifest = Path(cache_directory) / "manifest.json"
    contents = manifest.read_bytes()
    metadata = json.loads(contents)
    if metadata.get("cache_format") != expected_format:
        raise ValueError(f"expected a completed {expected_format} cache")
    metadata["manifest_sha256"] = hashlib.sha256(contents).hexdigest()
    return metadata


#%% Memory-mapped batches with bounded open-orbit mappings

class AutomaticFieldDataset(Dataset):
    """Read eligible field frames from bounded orbit-level memory maps."""

    def __init__(
        self, cache_directory, split, cache_mib=512, max_open_orbits=8,
        include_uncertainty=True,
        expected_cache_format=AUTOMATIC_CONDUCTANCE_CACHE_FORMAT
    ):
        self.directory = Path(cache_directory).expanduser().resolve()
        self.metadata = read_cache(self.directory, expected_cache_format)
        self.transform = ChannelTransform.from_dict(self.metadata["transform"])
        field_names = self.metadata.get("field_names")
        if field_names is None and expected_cache_format == AUTOMATIC_CONDUCTANCE_CACHE_FORMAT:
            field_names = ["Hall", "Pedersen"]
        self.field_count = len(field_names or [])
        if self.field_count < 1 or self.transform.channel_count != self.field_count:
            raise ValueError("cache field names and transform channels do not align")
        self.quality_channel_count = int(
            self.metadata.get("quality_channel_count", 1)
        )
        self.valid_channel_count = int(
            self.metadata.get("valid_channel_count", 1)
        )
        allowed_counts = (1, self.field_count)
        if self.quality_channel_count not in allowed_counts:
            raise ValueError(
                "cache quality channels must be one or match the field count"
            )
        if self.valid_channel_count not in allowed_counts:
            raise ValueError(
                "cache validity channels must be one or match the field count"
            )
        self.physical_channels = (
            2 * self.field_count + self.quality_channel_count
        )
        if not np.isfinite(cache_mib) or cache_mib <= 0 or max_open_orbits < 1:
            raise ValueError("cache MiB and maximum open orbits must be positive")
        self.cache_limit = int(cache_mib * 1024**2)
        self.max_open_orbits = int(max_open_orbits)
        self.include_uncertainty = bool(include_uncertainty)
        self.open_shards = OrderedDict()
        self.array_layouts = {}
        self.mapped_bytes = 0
        self.rows = [
            row for row in self.metadata["orbits"]
            if row["split"] == split and row["shard"]
        ]
        shard_ids, frame_ids, kp = [], [], []
        selected, source_frames, times, orbits = [], [], [], []
        offset = 0
        for number, row in enumerate(self.rows):
            with np.load(
                self.directory / row["shard"] / "frames.npz",
                allow_pickle=False
            ) as frames:
                indices = np.flatnonzero(frames["eligible"])
                shard_ids.append(np.full(len(indices), number, dtype=np.int32))
                frame_ids.append(indices)
                kp.append(frames["kp"][indices])
                selected.append(offset + indices)
                source_frames.append(frames["source_frame"][indices])
                times.append(frames["time"][indices])
                orbits.append(np.full(len(indices), row["orbit"], dtype=np.int32))
            offset += row["prepared_frames"]
        if not frame_ids or sum(len(ids) for ids in frame_ids) == 0:
            raise ValueError(f"{split} split has no cached eligible images")
        self.shard_ids = np.concatenate(shard_ids)
        self.frame_ids = np.concatenate(frame_ids)
        self.kp = np.concatenate(kp)
        self.selected_indices = np.concatenate(selected)
        self.source_frames = np.concatenate(source_frames)
        self.frame_time = np.concatenate(times)
        self.frame_orbit = np.concatenate(orbits)
        self.conditioning = None

    def __len__(self):
        return len(self.frame_ids)

    def close(self):
        for physical, valid in self.open_shards.values():
            physical._mmap.close()
            valid._mmap.close()
        self.open_shards.clear()
        self.mapped_bytes = 0

    def attach_conditioning(self, values):
        """Attach finite named conditions and remove unavailable frames."""

        if self.conditioning is not None:
            raise ValueError("conditioning is already attached")
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(self):
            raise ValueError("conditioning must have one row per cached frame")
        finite = np.all(np.isfinite(values), axis=1)
        if not finite.any():
            raise ValueError("no cached frames have complete conditioning")
        dropped = int((~finite).sum())
        for name in (
            "shard_ids", "frame_ids", "kp", "selected_indices",
            "source_frames", "frame_time", "frame_orbit"
        ):
            setattr(self, name, getattr(self, name)[finite])
        self.conditioning = values[finite]
        return dropped

    def _shard(self, number):
        if number in self.open_shards:
            self.open_shards.move_to_end(number)
            return self.open_shards[number]
        path = self.directory / self.rows[number]["shard"]
        count = self.rows[number]["prepared_frames"]
        bytes_per_frame = 46 * 46 * (
            self.physical_channels * np.dtype(np.float32).itemsize
            + self.valid_channel_count * np.dtype(bool).itemsize
        )
        size = count * bytes_per_frame
        if size > self.cache_limit:
            raise ValueError(
                f"{path}: orbit mapping exceeds --cache-mib; increase that setting"
            )
        while self.open_shards and (
            self.mapped_bytes + size > self.cache_limit
            or len(self.open_shards) >= self.max_open_orbits
        ):
            _, (old_physical, old_valid) = self.open_shards.popitem(last=False)
            self.mapped_bytes -= old_physical.nbytes + old_valid.nbytes
            old_physical._mmap.close()
            old_valid._mmap.close()
        if number not in self.array_layouts:
            physical = np.load(
                path / "physical.npy", mmap_mode="r", allow_pickle=False
            )
            valid = np.load(path / "valid.npy", mmap_mode="r", allow_pickle=False)
            expected = (count, self.physical_channels, 46, 46)
            valid_expected = (
                (count, 46, 46)
                if self.valid_channel_count == 1
                else (count, self.valid_channel_count, 46, 46)
            )
            if (
                physical.dtype != np.float32 or physical.shape != expected
                or valid.dtype != bool or valid.shape != valid_expected
            ):
                physical._mmap.close()
                valid._mmap.close()
                raise ValueError(f"{path}: cached arrays do not match their manifest")
            self.array_layouts[number] = [
                (array.offset, "F" if array.flags.f_contiguous else "C")
                for array in (physical, valid)
            ]
        else:
            physical_layout, valid_layout = self.array_layouts[number]
            physical = np.memmap(
                path / "physical.npy", mode="r", dtype=np.float32,
                shape=(count, self.physical_channels, 46, 46),
                offset=physical_layout[0], order=physical_layout[1]
            )
            valid = np.memmap(
                path / "valid.npy", mode="r", dtype=bool,
                shape=(
                    (count, 46, 46)
                    if self.valid_channel_count == 1
                    else (count, self.valid_channel_count, 46, 46)
                ), offset=valid_layout[0],
                order=valid_layout[1]
            )
        self.open_shards[number] = (physical, valid)
        self.mapped_bytes += size
        return physical, valid

    def __getitem__(self, index):
        physical, valid = self._shard(int(self.shard_ids[index]))
        frame = int(self.frame_ids[index])
        values = physical[frame, :self.field_count].copy()
        mask = valid[frame].copy()
        if self.valid_channel_count == 1:
            mask = mask[None]
        weight = physical[
            frame,
            2 * self.field_count:
            2 * self.field_count + self.quality_channel_count
        ].copy()
        target = self.transform.transform(values[None])[0].astype(np.float32)
        target_mask = np.broadcast_to(mask, target.shape)
        target = np.where(target_mask, target, 0.0)
        batch = {
            "encoder_input": torch.from_numpy(
                np.concatenate((target, weight), axis=0)
            ),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "weight": torch.from_numpy(weight),
            "kp": torch.tensor(self.kp[index], dtype=torch.float32),
            "index": int(self.selected_indices[index])
        }
        if self.conditioning is not None:
            batch["conditioning"] = torch.from_numpy(
                self.conditioning[index].copy()
            )
        if self.include_uncertainty:
            errors = physical[
                frame, self.field_count:2 * self.field_count
            ].copy()
            variance = self.transform.variance(
                values[None], errors[None]
            )[0].astype(np.float32)
            batch.update({
                "variance": torch.from_numpy(variance),
                "physical_target": torch.from_numpy(
                    np.where(target_mask, values, 0.0)
                ),
                "physical_standard_deviation": torch.from_numpy(
                    np.where(target_mask, errors, 0.0)
                )
            })
        return batch


AutomaticConductanceDataset = AutomaticFieldDataset
