"""Freeze reviewed WIC images with full, rejected and propagated partial masks."""

#%% Allowed image inputs; annotation masks are targets, never input channels

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from icreader import load

from scripts.manual_masking.annotated_data import annotation_times
from scripts.manual_masking.annotation_store import decode_mask, frame_class, propagated_mask

FIELDS = ("wic_corrected", "method_quality_weight", "wic_coverage", "sza", "dza")
SOURCE = Path("/home/bing/Dropbox/work/data/IMAGE_FUV/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2")


def human_keyframe(record, frame):
    payload = record.get("keyframes", {}).get(str(frame))
    if payload is None or not payload.get("updated_at"):
        return False
    return payload.get("human_reviewed", True)


def active_keyframe(record, frame):
    start = record["usable_start"]
    if record.get("full_good_stop") is not None and frame > record["full_good_stop"]:
        start = record["full_good_stop"] + 1
    earlier = [int(f) for f in record["keyframes"] if start <= int(f) <= frame]
    return max(earlier) if earlier else None


def frame_segments(record, classes):
    """Constant full/rejected intervals and held-forward partial masks."""
    segments = []
    interval_start = 0
    for frame, label in enumerate(classes):
        if frame and label != classes[frame - 1]:
            interval_start = frame
        boundary = active_keyframe(record, frame) if label == "partial" else interval_start
        segments.append(f"{record['orbit']}:{label}:{boundary}")
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/partial_image_annotations_46x46.json"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    contents = args.annotations.read_bytes()
    document = json.loads(contents)
    (args.output / "annotations_snapshot.json").write_bytes(contents)
    rows, images, masks, domains, sources, empty_rows = [], [], [], [], [], []
    excluded = dict(missing_files=[], unmatched_frames=0, no_observed_pixels=0, no_active_keyframe=0)

    #%% Full = keep all; rejected = keep none; partial = active keyframe

    for key, record in sorted(document["orbits"].items()):
        if record["review_status"] not in ("complete", "no_usable"):
            continue
        filename = args.source / f"or_{int(key):04d}.nc"
        if not filename.exists():
            excluded["missing_files"].append(int(key))
            continue
        review_times = annotation_times(record)
        classes = [frame_class(record, f) for f in range(record["n_frames"])]
        if any(label not in ("full", "bad", "partial") for label in classes):
            raise ValueError(f"Completed orbit{key} has unreviewed frames")
        segments = frame_segments(record, classes)
        with load(filename) as product:
            times = np.asarray(product.time, dtype="datetime64[us]")
            channels = np.stack([np.asarray(product.read(field), dtype=np.float32) for field in FIELDS], axis=1)
        distance = np.abs(times[:, None] - review_times[None, :])
        nearest = distance.argmin(axis=1)
        matched = distance[np.arange(len(times)), nearest] <= np.timedelta64(2, "s")
        excluded["unmatched_frames"] += int(np.sum(~matched))
        selected = np.flatnonzero(matched)
        if len(set(nearest[selected])) != len(selected):
            raise ValueError(f"Nonunique source/review exposure matches in orbit{key}")
        footprint = decode_mask(record["target_annotated_runs"], shape=(46, 46))
        human_frames = {f for f in map(int, record["keyframes"]) if human_keyframe(record, f)}
        kept = 0
        for frame in selected:
            reviewed_frame = int(nearest[frame])
            label = classes[reviewed_frame]
            keyframe = active_keyframe(record, reviewed_frame) if label == "partial" else None
            if label == "partial" and keyframe is None:
                excluded["no_active_keyframe"] += 1
                continue
            row = dict(orbit=int(key), frame=int(frame), annotation_frame=reviewed_frame, time=str(times[frame]), day=str(review_times[0].astype("datetime64[D]")), label=label, segment=segments[reviewed_frame], active_keyframe=keyframe, human_keyframe=label == "partial" and reviewed_frame in human_frames, restricted_footprint=not footprint.all())
            domain = np.isfinite(channels[frame, 0]) & footprint
            if not domain.any():
                excluded["no_observed_pixels"] += 1
                empty_rows.append(row)
                continue
            target = propagated_mask(record, reviewed_frame, shape=(46, 46)) & footprint
            if np.any(target & ~footprint):
                raise ValueError("Target outside reviewed footprint")
            images.append(channels[frame])
            masks.append(target)
            domains.append(domain)
            kept += 1
            rows.append(dict(row, supervised_pixels=int(domain.sum()), kept_pixels=int((target & domain).sum()), rejected_pixels=int((~target & domain).sum())))
        stat = filename.stat()
        sources.append(dict(orbit=int(key), file=str(filename), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, supervised_frames=kept))
        print(f"orbit{key}: {kept} reviewed nonempty frames", flush=True)

    #%% Loss domains and labels remain separate from the WIC input channels

    np.savez_compressed(args.output / "dataset.npz", images=np.asarray(images), targets=np.asarray(masks), domains=np.asarray(domains))
    with (args.output / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "empty_frames.json").write_text(json.dumps(empty_rows, indent=2) + "\n")
    classes = ("full", "bad", "partial")
    counts = {label: sum(r["label"] == label for r in rows) for label in classes}
    empty_counts = {label: sum(r["label"] == label for r in empty_rows) for label in classes}
    metadata = dict(annotation_sha256=hashlib.sha256(contents).hexdigest(), fields=FIELDS, samples=len(rows), class_counts=counts, empty_class_counts=empty_counts, human_keyframe_samples=sum(r["human_keyframe"] for r in rows), propagated_partial_samples=sum(r["label"] == "partial" and not r["human_keyframe"] for r in rows), orbits=len({r["orbit"] for r in rows}), segments=len({r["segment"] for r in rows}), sources=sources, excluded=excluded, supervision="finite corrected WIC intersect reviewed footprint; full all-keep, bad all-reject, partial held-forward keyframe", caveat="CS input support inherits upstream retrieval validity; empty frames archived separately because no pixels can supervise a mask")
    if all(not row["restricted_footprint"] for row in rows):
        metadata["supervision"] = "all finite corrected WIC pixels across the entire native 46x46 grid; full all-keep, bad all-reject, partial held-forward drawn mask; no old-grid restriction"
    (args.output / "dataset_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("sources", "fields")}), flush=True)


if __name__ == "__main__":
    main()
