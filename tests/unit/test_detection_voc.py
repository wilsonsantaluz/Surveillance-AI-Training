import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from surveillance_ml.detection.voc import import_zenodo_pascal_voc


def _image_bytes(seed: int) -> bytes:
    image = np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    output = BytesIO()
    Image.fromarray(image).save(output, format="JPEG", quality=95)
    return output.getvalue()


def _annotation(stem: str, class_name: str) -> bytes:
    return f"""<annotation>
  <filename>{stem}.jpg</filename>
  <source><database>COCO2017</database><annotation>COCO2017</annotation>
    <image>flickr</image><flickrid>NULL</flickrid></source>
  <size><width>32</width><height>32</height><depth>3</depth></size>
  <object><name>{class_name}</name><difficult>0</difficult>
    <bndbox><xmin>0</xmin><ymin>2</ymin><xmax>20</xmax><ymax>30</ymax></bndbox>
  </object>
</annotation>""".encode()


def _write_archive(path: Path) -> None:
    train: list[str] = []
    test: list[str] = []
    records: dict[str, tuple[bytes, str]] = {}
    for index in range(20):
        stem = f"train-{index:02d}"
        class_name = "Gun" if index % 2 == 0 else "Knife"
        records[stem] = (_image_bytes(index), class_name)
        train.append(stem)
    for index in range(8):
        stem = f"test-{index:02d}"
        class_name = "Rifle" if index % 2 == 0 else "Machete"
        records[stem] = (_image_bytes(100 + index), class_name)
        test.append(stem)

    records["train-duplicate"] = (records["test-00"][0], "Rifle")
    train.append("train-duplicate")
    records["train-conflicting-duplicate"] = (records["test-02"][0], "Knife")
    train.append("train-conflicting-duplicate")
    records["train-bat"] = (_image_bytes(300), "Baseball Bat")
    train.append("train-bat")

    with ZipFile(path, "w") as archive:
        for stem, (image_bytes, class_name) in records.items():
            archive.writestr(f"release/images/{stem}.jpg", image_bytes)
            archive.writestr(
                f"release/annotations/{stem}.xml",
                _annotation(stem, class_name),
            )
        archive.writestr("release/datasplit/train.txt", "\n".join(train) + "\n")
        archive.writestr("release/datasplit/test.txt", "\n".join(test) + "\n")


def test_voc_import_maps_classes_excludes_unknown_objects_and_keeps_groups_together(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "source.zip"
    output_root = tmp_path / "yolo"
    _write_archive(archive_path)

    result = import_zenodo_pascal_voc(
        archive_path,
        output_root,
        seed=17,
        validation_fraction=0.25,
        perceptual_distance=0,
    )

    assert result["source_image_count"] == 31
    assert result["converted_split_counts"]["test"]["images"] == 7
    assert result["excluded_reason_counts"] == {
        "duplicate_annotation_conflict": 2,
        "exact_duplicate": 1,
        "unsupported_baseball_bat": 1,
    }
    assert result["validation"]["image_count"] == 27
    assert result["validation"]["object_counts"] == {"firearm": 13, "knife": 14}
    assert (output_root / "dataset.yaml").is_file()
    manifest = json.loads((output_root / "provenance.json").read_text(encoding="utf-8"))
    assert manifest["conversion"]["class_mapping"] == {
        "Gun": "firearm",
        "Rifle": "firearm",
        "Knife": "knife",
        "Machete": "knife",
    }
    assert all(
        item["split"] == "test"
        for item in manifest["converted_records"]
        if item["image_id"] == "test-00"
    )
    train_labels = list((output_root / "labels" / "train").rglob("*.txt"))
    assert any(path.read_text(encoding="utf-8").startswith("0 ") for path in train_labels)
    assert any(path.read_text(encoding="utf-8").startswith("1 ") for path in train_labels)
