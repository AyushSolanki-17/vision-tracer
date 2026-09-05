"""Small explicit torch cache format, readable with CPU-only dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


LEGACY_CACHE_SCHEMA_VERSION = 1
PILOT_CACHE_SCHEMA_VERSION = 2
# Kept for callers that create the completed synthetic evidence format.
CACHE_SCHEMA_VERSION = LEGACY_CACHE_SCHEMA_VERSION


def tensor_descriptor(tensor: torch.Tensor) -> dict[str, object]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def make_cache_record(
    *, metadata: dict[str, Any], representations: dict[str, torch.Tensor], attentions: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Detach all evidence to compact CPU tensors and attach descriptors."""
    cpu_representations = {name: value.detach().to("cpu").contiguous().clone() for name, value in representations.items()}
    cpu_attentions = {name: value.detach().to("cpu").contiguous().clone() for name, value in attentions.items()}
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "metadata": metadata,
        "representations": cpu_representations,
        "attentions": cpu_attentions,
        "tensor_descriptors": {
            "representations": {name: tensor_descriptor(value) for name, value in cpu_representations.items()},
            "attentions": {name: tensor_descriptor(value) for name, value in cpu_attentions.items()},
        },
    }


def make_pilot_cache_record(
    *, identifiers: dict[str, Any], provenance: dict[str, Any], input_metadata: dict[str, Any],
    extraction: dict[str, Any], attention: dict[str, Any], tensors: dict[str, torch.Tensor],
    tensor_descriptors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Create immutable schema-2 raw evidence for one real-image pilot item."""
    cpu_tensors = {name: value.detach().to("cpu").contiguous().clone() for name, value in tensors.items()}
    return {
        "schema_version": PILOT_CACHE_SCHEMA_VERSION,
        "record_kind": "sample_evidence",
        "identifiers": identifiers,
        "provenance": provenance,
        "input": input_metadata,
        "extraction": extraction,
        "attention": attention,
        "tensors": cpu_tensors,
        "tensor_descriptors": tensor_descriptors,
    }


def validate_cache_record(record: dict[str, Any]) -> None:
    version = record.get("schema_version")
    if version == LEGACY_CACHE_SCHEMA_VERSION:
        _validate_legacy_cache_record(record)
        return
    if version == PILOT_CACHE_SCHEMA_VERSION:
        _validate_pilot_cache_record(record)
        return
    raise ValueError(f"unsupported cache schema: {version}")


def _validate_legacy_cache_record(record: dict[str, Any]) -> None:
    for section in ("representations", "attentions"):
        tensors = record.get(section)
        descriptors = record.get("tensor_descriptors", {}).get(section, {})
        if not isinstance(tensors, dict) or not tensors:
            raise ValueError(f"cache has no {section}")
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                raise ValueError(f"{section}.{name} is not a CPU tensor")
            if descriptors.get(name) != tensor_descriptor(tensor):
                raise ValueError(f"descriptor mismatch for {section}.{name}")
    metadata = record.get("metadata", {})
    for key in ("experiment_identifier", "image_identifier", "model", "geometry", "extraction_configuration"):
        if key not in metadata:
            raise ValueError(f"cache metadata is missing {key}")


def _validate_pilot_cache_record(record: dict[str, Any]) -> None:
    if record.get("record_kind") != "sample_evidence":
        raise ValueError("schema-2 cache has an invalid record_kind")
    for key in ("identifiers", "provenance", "input", "extraction", "attention", "tensors", "tensor_descriptors"):
        if key not in record or not isinstance(record[key], dict):
            raise ValueError(f"schema-2 cache is missing {key}")
    identifiers = record["identifiers"]
    for key in ("experiment_id", "run_id", "sample_id", "image_id", "manifest_id"):
        if not identifiers.get(key):
            raise ValueError(f"schema-2 cache identifiers missing {key}")
    source_image = record["provenance"].get("source_image", {})
    for key in ("source", "sha256", "original_width", "original_height"):
        if key not in source_image:
            raise ValueError(f"schema-2 cache source image missing {key}")
    if len(str(source_image["sha256"])) != 64:
        raise ValueError("schema-2 cache source image has invalid SHA-256")
    for key in ("repository_commit", "extraction_timestamp_utc", "code_package_version", "environment"):
        if not record["provenance"].get(key):
            raise ValueError(f"schema-2 cache provenance missing {key}")
    for kind in ("qwen", "dino"):
        model = record["provenance"].get("model", {}).get(kind, {})
        processor = record["provenance"].get("processor", {}).get(kind, {})
        if not model.get("identifier") or not model.get("revision"):
            raise ValueError(f"schema-2 cache provenance missing {kind} model identity")
        if not processor.get("identifier") or not processor.get("revision") or not processor.get("config_sha256"):
            raise ValueError(f"schema-2 cache provenance missing {kind} processor identity")
    input_metadata = record["input"]
    for key in ("prompt_template_id", "resolved_width", "resolved_height", "image_grid_thw", "patch_grid_thw", "merged_grid_thw", "image_token_sequence_positions", "token_coordinates"):
        if key not in input_metadata:
            raise ValueError(f"schema-2 cache input missing {key}")
    merged = input_metadata["merged_grid_thw"]
    expected = int(merged[0]) * int(merged[1]) * int(merged[2])
    if len(input_metadata["image_token_sequence_positions"]) != expected:
        raise ValueError("schema-2 image token positions do not match merged grid")
    coordinates = input_metadata["token_coordinates"]
    if len(coordinates.get("patch", [])) != int(input_metadata["patch_grid_thw"][0]) * int(input_metadata["patch_grid_thw"][1]) * int(input_metadata["patch_grid_thw"][2]):
        raise ValueError("schema-2 patch coordinate count does not match patch grid")
    if len(coordinates.get("merged", [])) != expected:
        raise ValueError("schema-2 merged coordinate count does not match merged grid")
    tensors = record["tensors"]
    descriptors = record["tensor_descriptors"]
    if not tensors or set(tensors) != set(descriptors):
        raise ValueError("schema-2 tensors and descriptors do not agree")
    for name, tensor in tensors.items():
        descriptor = descriptors[name]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError(f"schema-2 tensor {name} is not a CPU tensor")
        if descriptor.get("shape") != list(tensor.shape) or descriptor.get("dtype") != str(tensor.dtype):
            raise ValueError(f"schema-2 tensor descriptor mismatch for {name}")
        for key in ("semantic_role", "layer_id", "token_space", "coordinate_space", "module_path"):
            if key not in descriptor:
                raise ValueError(f"schema-2 tensor descriptor missing {key} for {name}")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_integrity_manifest(path: str | Path, records: list[dict[str, Any]]) -> Path:
    """Write a small JSON manifest tying source images to finalized caches."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": 1, "records": records}, indent=2, sort_keys=True) + "\n")
    return output


def validate_integrity_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("records"), list):
        raise ValueError("invalid integrity manifest")
    seen: set[str] = set()
    for entry in payload["records"]:
        for key in ("image_id", "source_sha256", "cache_path", "cache_sha256", "schema_version", "repository_commit", "extraction_configuration_sha256"):
            if key not in entry:
                raise ValueError(f"integrity manifest record missing {key}")
        if entry["image_id"] in seen:
            raise ValueError("duplicate image_id in integrity manifest")
        seen.add(entry["image_id"])
        cache_path = manifest_path.parent / entry["cache_path"]
        if sha256_path(cache_path) != entry["cache_sha256"]:
            raise ValueError(f"cache hash mismatch for {entry['image_id']}")
        cache = load_cache_cpu(cache_path)
        if cache["schema_version"] != entry["schema_version"]:
            raise ValueError(f"cache schema mismatch for {entry['image_id']}")
        if cache["identifiers"]["image_id"] != entry["image_id"]:
            raise ValueError(f"cache image ID mismatch for {entry['image_id']}")
        if cache["provenance"]["source_image"]["sha256"] != entry["source_sha256"]:
            raise ValueError(f"cache source hash mismatch for {entry['image_id']}")
        extraction_hash = hashlib.sha256(
            json.dumps(cache["extraction"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if extraction_hash != entry["extraction_configuration_sha256"]:
            raise ValueError(f"cache extraction configuration mismatch for {entry['image_id']}")
    return payload


def save_cache(path: str | Path, record: dict[str, Any]) -> Path:
    validate_cache_record(record)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, output)
    return output


def load_cache_cpu(path: str | Path) -> dict[str, Any]:
    """Load evidence without importing Transformers or any Qwen class."""
    # PyTorch 2.6 made safe ``weights_only`` loading the default. Earlier pilot
    # records can contain PyTorch's harmless ``TorchVersion`` wrapper in runtime
    # provenance; allow only that concrete value type, never arbitrary pickle
    # globals. New records stringify version fields below their call sites.
    from torch.torch_version import TorchVersion

    with torch.serialization.safe_globals([TorchVersion]):
        record = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_cache_record(record)
    return record
