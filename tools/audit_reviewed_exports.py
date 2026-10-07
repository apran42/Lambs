"""Audit reviewed YOLO exports before combining team datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SPLITS = ("train", "val", "test")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def audit_folder(folder: Path) -> dict:
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    export_dirs = sorted((folder / "exports").glob("export_*"))
    if not export_dirs:
        return {"folder": str(folder), "errors": ["no export"]}
    export = export_dirs[-1]
    report = json.loads((export / "quality_report.json").read_text(encoding="utf-8"))
    review = rows(folder / "frame_review.csv")
    timeseries = rows(export / "reviewed_timeseries.csv")
    errors: list[str] = []
    image_names: set[str] = set()
    box_counts: dict[str, int] = {}
    total_boxes = 0
    counts: dict[str, int] = {}
    for split in SPLITS:
        images = list((export / "images" / split).glob("*.jpg"))
        labels = list((export / "labels" / split).glob("*.txt"))
        counts[split] = len(images)
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            errors.append(f"{split}: unmatched images/labels")
        if image_names & image_stems:
            errors.append(f"{split}: duplicate sample IDs across splits")
        image_names |= image_stems
        for label in labels:
            box_counts[label.stem] = 0
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label.name}:{line_number}: expected 5 YOLO columns")
                    continue
                try:
                    klass, x, y, width, height = map(float, parts)
                    if klass != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                        raise ValueError("invalid class or box range")
                    total_boxes += 1
                    box_counts[label.stem] += 1
                    if x - width / 2 < -1e-5 or x + width / 2 > 1 + 1e-5 or y - height / 2 < -1e-5 or y + height / 2 > 1 + 1e-5:
                        raise ValueError("box extends beyond image")
                except ValueError as exc:
                    errors.append(f"{label.name}:{line_number}: {exc}")
    if sum(counts.values()) != report["accepted_frames"]:
        errors.append("frame count differs from quality report")
    if total_boxes != report["person_annotations"]:
        errors.append("box count differs from quality report")
    if len(timeseries) != report["accepted_frames"]:
        errors.append("timeseries count differs from quality report")
    if {row["sample_id"] for row in timeseries} != image_names:
        errors.append("timeseries IDs differ from exported image IDs")
    status_by_id = {row["sample_id"]: row for row in review}
    if len(status_by_id) != len(review):
        errors.append("duplicate sample IDs in frame_review.csv")
    for sample_id in image_names:
        item = status_by_id.get(sample_id)
        if item is None or item["manual_review_status"] not in {"approved", "corrected"}:
            errors.append(f"{sample_id}: not approved/corrected")
        elif item["manual_box_review_status"] != "completed":
            errors.append(f"{sample_id}: boxes not completed")
        elif int(item["manual_ground_truth_count"]) != box_counts[sample_id]:
            errors.append(f"{sample_id}: manual count differs from label boxes")
    for item in timeseries:
        if int(item["ground_truth_count"]) != box_counts[item["sample_id"]]:
            errors.append(f"{item['sample_id']}: timeseries count differs from label boxes")
    return {
        "folder": str(folder), "video": Path(metadata["source_video"]).name,
        "export": str(export), "all_exports": len(export_dirs),
        "frames": counts, "boxes": total_boxes,
        "reviewed": len(review), "errors": errors[:30], "error_count": len(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/training"))
    args = parser.parse_args()
    folders = sorted(path.parent for path in args.root.rglob("metadata.json"))
    results = [audit_folder(folder) for folder in folders]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if any(item["error_count"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
