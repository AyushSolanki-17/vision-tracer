"""CPU-only summaries over synthetic instrumentation evidence caches."""

from __future__ import annotations

from typing import Any

import torch


def cached_evidence_analysis(record: dict[str, Any]) -> dict[str, Any]:
    """Summarize norms and selected attention mass over image placeholders."""
    if record.get("schema_version") == 2:
        return _pilot_evidence_analysis(record)
    indices = record["metadata"]["geometry"]["image_token_sequence_indices"]
    result: dict[str, Any] = {"representation_mean_l2_norm": {}, "attention_mass_over_image_keys": {}}
    for name, tensor in record["representations"].items():
        result["representation_mean_l2_norm"][name] = float(torch.linalg.vector_norm(tensor.float(), dim=-1).mean())
    for name, attention in record["attentions"].items():
        # Mean probability mass assigned to image-key positions, over batch,
        # query heads, and query positions. This is descriptive only.
        result["attention_mass_over_image_keys"][name] = float(attention.float()[..., indices].sum(dim=-1).mean())
    return result


def _pilot_evidence_analysis(record: dict[str, Any]) -> dict[str, Any]:
    indices = record["input"]["image_token_sequence_positions"]
    result: dict[str, Any] = {
        "representation_mean_l2_norm": {},
        "attention_mass_over_image_keys": {},
        "dino_geometry": record["input"].get("dino_geometry", {}),
    }
    for name, tensor in record["tensors"].items():
        if name.startswith("qwen.attention."):
            result["attention_mass_over_image_keys"][name] = float(tensor.float()[..., indices].sum(dim=-1).mean())
        else:
            result["representation_mean_l2_norm"][name] = float(torch.linalg.vector_norm(tensor.float(), dim=-1).mean())
    return result
