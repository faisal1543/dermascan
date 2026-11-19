"""DERM-10: image-decode check.

Unit tests run everywhere (CI uses small synthetic images). The full-dataset test runs only
where HAM10000 is available: set HAM10000_IMAGES to the image folder (e.g. /kaggle/input).
"""
import os
from pathlib import Path

import pytest
from PIL import Image

from perception.check_images import check_all, check_image, index_images, main

SPLITS = Path(__file__).resolve().parents[1] / "splits" / "splits.csv"


def _jpeg(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (180, 120, 90)).save(path, "JPEG")
    return path


def test_valid_image_passes(tmp_path):
    assert check_image(_jpeg(tmp_path / "ISIC_0000001.jpg")) is None


def test_truncated_image_fails(tmp_path):
    good = _jpeg(tmp_path / "good.jpg").read_bytes()
    bad = tmp_path / "ISIC_0000002.jpg"
    bad.write_bytes(good[: len(good) // 3])
    assert check_image(bad) is not None


def test_non_image_file_fails(tmp_path):
    fake = tmp_path / "ISIC_0000003.jpg"
    fake.write_text("not an image")
    assert check_image(fake) is not None


def test_duplicate_copies_indexed_once_and_missing_reported(tmp_path):
    # Kaggle layout: the same image_id appears in two folders.
    _jpeg(tmp_path / "part_1" / "ISIC_0000004.jpg")
    _jpeg(tmp_path / "copy" / "part_1" / "ISIC_0000004.jpg")
    index = index_images(tmp_path)
    assert list(index) == ["ISIC_0000004"]
    failures = check_all(["ISIC_0000004", "ISIC_9999999"], index)
    assert failures == [("ISIC_9999999", "missing")]


def test_cli_exit_codes(tmp_path):
    _jpeg(tmp_path / "imgs" / "ISIC_0000005.jpg")
    splits = tmp_path / "splits.csv"
    splits.write_text("image_id,lesion_id,dx,split\nISIC_0000005,HAM_1,nv,train\n")
    assert main(["--images", str(tmp_path / "imgs"), "--splits", str(splits)]) == 0
    splits.write_text(splits.read_text() + "ISIC_0000006,HAM_2,nv,val\n")
    report = tmp_path / "fail.csv"
    assert main(["--images", str(tmp_path / "imgs"), "--splits", str(splits), "--report", str(report)]) == 1
    assert "ISIC_0000006" in report.read_text()


@pytest.mark.skipif(not os.environ.get("HAM10000_IMAGES"), reason="HAM10000 images not available here")
def test_every_split_image_decodes():
    from perception.check_images import read_split_ids

    index = index_images(Path(os.environ["HAM10000_IMAGES"]))
    failures = check_all(read_split_ids(SPLITS), index)
    assert not failures, f"{len(failures)} images missing or corrupt, first: {failures[:5]}"
