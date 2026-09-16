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
    parser.add_argument("--source", action="append", required=True, help="name=export_directory")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    sources: list[tuple[str, Path]] = []
    for item in args.source:
        if "=" not in item:
            raise ValueError(f"Source must use name=path syntax: {item}")
        name, raw_path = item.split("=", 1)
        source = Path(raw_path).resolve()
        if not name or not source.is_dir():
            raise ValueError(f"Invalid source: {item}")
        sources.append((name, source))

    counts = {split: 0 for split in SPLITS}
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
        for source_name, source in sources:
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
        "sources": [{"name": name, "path": str(path)} for name, path in sources],
        "split_counts": counts,
        "note": "Pilot split preserves each source export's chronological train/val/test split.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
