"""Narrow CUB-200-2011 manifest preparation for the real-image pilot.

This module is intentionally limited to selecting a tiny, reproducible set of
photographs and making the existing explicit-manifest extractor able to read
them.  It is not a general CUB dataset API.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .manifest import sha256_file


CUB_DATASET_NAME = "CUB-200-2011"
CUB_DATASET_VERSION = "1.0"
CUB_OFFICIAL_RECORD_URL = "https://data.caltech.edu/records/65de6-vp158"
CUB_OFFICIAL_ARCHIVE_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
CUB_OFFICIAL_ARCHIVE_MD5 = "97eceeb196236b17998738112f37df78"
CUB_ROOT_NAME = "CUB_200_2011"
CUB_PILOT_SELECTION_RULE = "evenly_spaced_class_ids_then_lowest_numeric_image_id"


@dataclass(frozen=True)
class CUBImage:
    image_id: int
    relative_path: str
    class_id: int
    class_name: str


def _read_index(path: Path, expected_columns: int) -> dict[int, tuple[str, ...]]:
    values: dict[int, tuple[str, ...]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split(maxsplit=expected_columns - 1)
        if len(fields) != expected_columns:
            raise ValueError(f"invalid CUB metadata at {path}:{line_number}")
        identifier = int(fields[0])
        if identifier in values:
            raise ValueError(f"duplicate CUB identifier {identifier} in {path}")
        values[identifier] = tuple(fields[1:])
    if not values:
        raise ValueError(f"empty CUB metadata file {path}")
    return values


def load_cub_images(cub_root: str | Path) -> list[CUBImage]:
    """Read only the three metadata files needed by the small pilot."""
    root = Path(cub_root)
    images = _read_index(root / "images.txt", 2)
    labels = _read_index(root / "image_class_labels.txt", 2)
    classes = _read_index(root / "classes.txt", 2)
    if set(images) != set(labels):
        raise ValueError("CUB image and class-label identifiers differ")
    result: list[CUBImage] = []
    for image_id in sorted(images):
        class_id = int(labels[image_id][0])
        if class_id not in classes:
            raise ValueError(f"CUB image {image_id} references missing class {class_id}")
        relative_path = images[image_id][0]
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError(f"unsafe CUB image path for image {image_id}")
        result.append(CUBImage(image_id, relative_path, class_id, classes[class_id][0]))
    return result


def select_cub_pilot(images: list[CUBImage], *, count: int = 16) -> list[CUBImage]:
    """Choose one image from evenly spaced classes, with no random sampling."""
    if not 10 <= count <= 20:
        raise ValueError("CUB pilot count must be between 10 and 20")
    by_class: dict[int, list[CUBImage]] = {}
    for image in images:
        by_class.setdefault(image.class_id, []).append(image)
    class_ids = sorted(by_class)
    if len(class_ids) < count:
        raise ValueError("CUB metadata has fewer classes than requested pilot images")
    selected_classes = [class_ids[index * len(class_ids) // count] for index in range(count)]
    return [min(by_class[class_id], key=lambda image: image.image_id) for class_id in selected_classes]


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324: verified against the publisher's published archive checksum.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_official_cub_archive(path: str | Path) -> Path:
    archive = Path(path)
    observed = _md5_file(archive)
    if observed != CUB_OFFICIAL_ARCHIVE_MD5:
        raise ValueError(f"official CUB archive MD5 mismatch: {observed}")
    return archive


def download_official_cub_archive(destination: str | Path) -> Path:
    """Download and verify the official archive; intended for Kaggle only."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(CUB_OFFICIAL_ARCHIVE_URL) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)
    try:
        return verify_official_cub_archive(target)
    except ValueError:
        target.unlink(missing_ok=True)
        raise


def extract_cub_pilot_from_archive(archive_path: str | Path, destination: str | Path, *, count: int = 16) -> Path:
    """Extract just CUB metadata and selected images, never the full image tree."""
    archive, output = Path(archive_path), Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    metadata_names = [f"{CUB_ROOT_NAME}/{name}" for name in ("images.txt", "image_class_labels.txt", "classes.txt")]
    with tarfile.open(archive, "r:gz") as tar:
        members = {member.name: member for member in tar.getmembers() if member.isfile()}
        if any(name not in members for name in metadata_names):
            raise ValueError("archive does not contain required CUB metadata")
        for name in metadata_names:
            _extract_member(tar, members[name], output)
    selected = select_cub_pilot(load_cub_images(output / CUB_ROOT_NAME), count=count)
    with tarfile.open(archive, "r:gz") as tar:
        members = {member.name: member for member in tar.getmembers() if member.isfile()}
        for image in selected:
            name = f"{CUB_ROOT_NAME}/images/{image.relative_path}"
            if name not in members:
                raise ValueError(f"archive lacks selected CUB image {image.image_id}")
            _extract_member(tar, members[name], output)
    return output / CUB_ROOT_NAME


def _extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    target = destination / member.name
    if target.resolve().is_relative_to(destination.resolve()) is False:
        raise ValueError(f"unsafe archive member {member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source = tar.extractfile(member)
    if source is None:
        raise ValueError(f"cannot extract archive member {member.name}")
    with source, target.open("wb") as output:
        shutil.copyfileobj(source, output)


def generate_cub_pilot_manifest(
    cub_root: str | Path, output_directory: str | Path, *, count: int = 16,
    source_location: str = CUB_OFFICIAL_RECORD_URL, dataset_version: str = CUB_DATASET_VERSION,
) -> tuple[Path, Path]:
    """Write the extractor-compatible JSONL and a complete pilot provenance sidecar."""
    root, output = Path(cub_root), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    selected = select_cub_pilot(load_cub_images(root), count=count)
    manifest_path = output / "pilot_images.jsonl"
    records: list[dict[str, object]] = []
    for image in selected:
        image_path = root / "images" / image.relative_path
        with Image.open(image_path) as opened:
            width, height = opened.size
        record = {
            "image_id": f"cub-{image.image_id:05d}", "source": os.path.relpath(image_path, output),
            "sha256": sha256_file(image_path), "original_width": width, "original_height": height,
            "source_dataset": CUB_DATASET_NAME, "source_dataset_version": dataset_version,
            "source_location": source_location, "source_image_id": image.image_id,
            "category_id": image.class_id, "category": image.class_name,
            "selection_rule": CUB_PILOT_SELECTION_RULE, "selection_seed": None,
        }
        records.append(record)
    manifest_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    provenance_path = output / "pilot_source_provenance.json"
    provenance_path.write_text(json.dumps({
        "dataset": {
            "name": CUB_DATASET_NAME, "version": dataset_version, "source_location": source_location,
            "official_archive_url": CUB_OFFICIAL_ARCHIVE_URL, "official_archive_md5": CUB_OFFICIAL_ARCHIVE_MD5,
        },
        "selection": {"count": count, "rule": CUB_PILOT_SELECTION_RULE, "seed": None},
        "selected_images": records,
    }, indent=2, sort_keys=True) + "\n")
    return manifest_path, provenance_path
