"""DERM-10 image-decode check: every image in the frozen split must exist and decode.

Usage (Kaggle or local):
    python -m perception.check_images --images /kaggle/input --splits splits/splits.csv

The Kaggle copy of HAM10000 stores each image twice (two folder layouts), so images are
indexed by file name (image_id) and each image_id is checked once.

Exit code 0 when every image decodes, 1 otherwise. Failures are written to --report.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image


def index_images(root: Path) -> dict[str, Path]:
    """Map image_id (file stem) -> first path found under root."""
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*.jpg")):
        index.setdefault(path.stem, path)
    return index


def check_image(path: Path) -> str | None:
    """Return None if the image decodes to a non-empty RGB array, else an error message."""
    try:
        with Image.open(path) as img:
            img.load()  # forces a full decode; truncated files fail here
            rgb = img.convert("RGB")
            if rgb.width == 0 or rgb.height == 0:
                return "empty image"
    except Exception as exc:  # PIL raises several exception types for bad files
        return f"{type(exc).__name__}: {exc}"
    return None


def check_all(image_ids: list[str], index: dict[str, Path]) -> list[tuple[str, str]]:
    """Return (image_id, problem) for every missing or undecodable image."""
    failures = []
    for image_id in image_ids:
        path = index.get(image_id)
        if path is None:
            failures.append((image_id, "missing"))
            continue
        error = check_image(path)
        if error:
            failures.append((image_id, error))
    return failures


def read_split_ids(splits_csv: Path) -> list[str]:
    with splits_csv.open(newline="") as f:
        return [row["image_id"] for row in csv.DictReader(f)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", required=True, type=Path, help="folder searched recursively for *.jpg")
    parser.add_argument("--splits", default=Path("splits/splits.csv"), type=Path)
    parser.add_argument("--report", default=Path("results/image_check_failures.csv"), type=Path)
    args = parser.parse_args(argv)

    ids = read_split_ids(args.splits)
    index = index_images(args.images)
    failures = check_all(ids, index)

    print(f"checked {len(ids)} images from {args.splits}; {len(failures)} problems")
    if failures:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "problem"])
            writer.writerows(failures)
        print(f"failures written to {args.report}")
        return 1
    print("all images decode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
