from __future__ import annotations

import json

from PIL import Image

from vision_trace.cub_pilot import CUB_PILOT_SELECTION_RULE, generate_cub_pilot_manifest, load_cub_images, select_cub_pilot
from vision_trace.manifest import load_image_manifest, sha256_file, verified_image


def _cub_fixture(tmp_path, *, class_count: int = 20):
    root = tmp_path / "CUB_200_2011"
    images_root = root / "images"
    image_lines, label_lines, class_lines = [], [], []
    for identifier in range(1, class_count + 1):
        category = f"{identifier:03d}.species_{identifier:03d}"
        relative = f"{category}/image_{identifier:04d}.jpg"
        path = images_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (identifier + 10, identifier + 20), color=(identifier, 2, 3)).save(path)
        image_lines.append(f"{identifier} {relative}")
        label_lines.append(f"{identifier} {identifier}")
        class_lines.append(f"{identifier} {category}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "images.txt").write_text("\n".join(image_lines) + "\n")
    (root / "image_class_labels.txt").write_text("\n".join(label_lines) + "\n")
    (root / "classes.txt").write_text("\n".join(class_lines) + "\n")
    return root


def test_cub_pilot_selection_is_deterministic_and_spans_classes(tmp_path) -> None:
    root = _cub_fixture(tmp_path)
    images = load_cub_images(root)
    first = select_cub_pilot(images, count=10)
    second = select_cub_pilot(images, count=10)
    assert first == second
    assert [image.class_id for image in first] == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    assert len({image.class_id for image in first}) == 10


def test_cub_manifest_records_hashes_provenance_and_validates_images(tmp_path) -> None:
    root = _cub_fixture(tmp_path)
    manifest_path, provenance_path = generate_cub_pilot_manifest(root, tmp_path / "pilot", count=10)
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    entries = load_image_manifest(manifest_path)
    assert len(records) == len(entries) == 10
    assert records[0]["selection_rule"] == CUB_PILOT_SELECTION_RULE
    assert records[0]["selection_seed"] is None
    assert records[0]["sha256"] == sha256_file(root / "images" / "001.species_001/image_0001.jpg")
    assert verified_image(entries[0], manifest_directory=manifest_path.parent).size == (11, 21)
    provenance = json.loads(provenance_path.read_text())
    assert provenance["selection"]["count"] == 10
    assert provenance["dataset"]["official_archive_md5"] == "97eceeb196236b17998738112f37df78"
    assert provenance["selected_images"] == records
