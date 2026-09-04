"""Explicit, validated JSONL manifests for small real-image pilots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class ImageManifestEntry:
    image_id: str
    source: str
    sha256: str
    original_width: int
    original_height: int

    def metadata(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image_manifest(path: str | Path) -> list[ImageManifestEntry]:
    """Load an explicit JSONL manifest without discovering files implicitly."""
    manifest_path = Path(path)
    entries: list[ImageManifestEntry] = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            entry = ImageManifestEntry(
                image_id=str(payload["image_id"]), source=str(payload["source"]), sha256=str(payload["sha256"]),
                original_width=int(payload["original_width"]), original_height=int(payload["original_height"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid manifest entry at line {line_number}") from exc
        if not entry.image_id or entry.image_id in identifiers:
            raise ValueError(f"duplicate or empty image_id at line {line_number}")
        if len(entry.sha256) != 64 or any(char not in "0123456789abcdef" for char in entry.sha256.lower()):
            raise ValueError(f"invalid SHA-256 for {entry.image_id}")
        if entry.original_width <= 0 or entry.original_height <= 0:
            raise ValueError(f"invalid original dimensions for {entry.image_id}")
        identifiers.add(entry.image_id)
        entries.append(entry)
    if not entries:
        raise ValueError("image manifest contains no entries")
    return entries


def verified_image(entry: ImageManifestEntry, *, manifest_directory: str | Path) -> Image.Image:
    """Load an image only after verifying its declared bytes and dimensions."""
    source_path = Path(manifest_directory) / entry.source
    if sha256_file(source_path) != entry.sha256:
        raise ValueError(f"source hash mismatch for {entry.image_id}")
    with Image.open(source_path) as opened:
        if opened.width != entry.original_width or opened.height != entry.original_height:
            raise ValueError(f"source dimensions mismatch for {entry.image_id}")
        return opened.convert("RGB").copy()
