from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test")


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine exported Lambs YOLO datasets safely")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Legacy mode: name=export_directory (keeps each export's internal split)",
    )
    for split in SPLITS:
        parser.add_argument(
            f"--{split}-source",
            action="append",
            default=[],
            help=f"Grouped mode: name=export_directory (puts the entire video in {split})",
        )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_source(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"Source must use name=path syntax: {item}")
    name, raw_path = item.split("=", 1)
    source = Path(raw_path).resolve()
    if not name or not source.is_dir():
        raise ValueError(f"Invalid source: {item}")
    return name, source


def copy_export(source_name: str, source: Path, output: Path, target_split: str) -> int:
    copied = 0
    for original_split in SPLITS:
        image_dir = source / "images" / original_split
        label_dir = source / "labels" / original_split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing images/labels split in {source}: {original_split}")
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}")
            destination_stem = f"{source_name}__{image_path.stem}"
            link_or_copy(image_path, output / "images" / target_split / f"{destination_stem}.jpg")
            link_or_copy(label_path, output / "labels" / target_split / f"{destination_stem}.txt")
            copied += 1
    return copied


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    grouped_items = {
        split: getattr(args, f"{split}_source")
        for split in SPLITS
    }
    grouped_mode = any(grouped_items.values())
    if args.source and grouped_mode:
        raise ValueError("Do not mix --source with --train-source/--val-source/--test-source")
    if not args.source and not grouped_mode:
        raise ValueError("At least one source is required")
    if grouped_mode and any(not grouped_items[split] for split in SPLITS):
        raise ValueError("Grouped mode requires at least one train, val, and test source")

    source_names: set[str] = set()
    source_manifest: list[dict[str, str]] = []

    def register_source(item: str, target_split: str) -> tuple[str, Path]:
        name, source = parse_source(item)
        if name in source_names:
            raise ValueError(f"Duplicate source name: {name}")
        source_names.add(name)
        source_manifest.append({"name": name, "path": str(source), "split": target_split})
        return name, source

    counts = {split: 0 for split in SPLITS}
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    if grouped_mode:
        for target_split in SPLITS:
            for item in grouped_items[target_split]:
                source_name, source = register_source(item, target_split)
                counts[target_split] += copy_export(source_name, source, output, target_split)
        note = "Video-grouped split: every frame from a source video belongs to exactly one split."
    else:
        for item in args.source:
            source_name, source = register_source(item, "internal")
            for split in SPLITS:
                image_dir = source / "images" / split
                label_dir = source / "labels" / split
                for image_path in sorted(image_dir.glob("*.jpg")):
                    label_path = label_dir / f"{image_path.stem}.txt"
                    if not label_path.is_file():
                        raise FileNotFoundError(f"Missing label for {image_path}")
                    destination_stem = f"{source_name}__{image_path.stem}"
                    link_or_copy(image_path, output / "images" / split / f"{destination_stem}.jpg")
                    link_or_copy(label_path, output / "labels" / split / f"{destination_stem}.txt")
                    counts[split] += 1
        note = "Legacy split: preserves each source export's internal chronological split."

    (output / "data.yaml").write_text(
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: person\n",
        encoding="utf-8",
    )
    manifest = {
        "sources": source_manifest,
        "split_counts": counts,
        "note": note,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
