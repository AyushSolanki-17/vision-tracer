"""Small explicit torch cache format, readable with CPU-only dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


CACHE_SCHEMA_VERSION = 1


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


def validate_cache_record(record: dict[str, Any]) -> None:
    if record.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported cache schema: {record.get('schema_version')}")
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


def save_cache(path: str | Path, record: dict[str, Any]) -> Path:
    validate_cache_record(record)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, output)
    return output


def load_cache_cpu(path: str | Path) -> dict[str, Any]:
    """Load evidence without importing Transformers or any Qwen class."""
    record = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_cache_record(record)
    return record
