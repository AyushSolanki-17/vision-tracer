"""CPU-only summaries over synthetic instrumentation evidence caches."""

from __future__ import annotations

from typing import Any

import torch


def cached_evidence_analysis(record: dict[str, Any]) -> dict[str, Any]:
    """Summarize norms and selected attention mass over image placeholders."""
    indices = record["metadata"]["geometry"]["image_token_sequence_indices"]
    result: dict[str, Any] = {"representation_mean_l2_norm": {}, "attention_mass_over_image_keys": {}}
    for name, tensor in record["representations"].items():
        result["representation_mean_l2_norm"][name] = float(torch.linalg.vector_norm(tensor.float(), dim=-1).mean())
    for name, attention in record["attentions"].items():
        # Mean probability mass assigned to image-key positions, over batch,
        # query heads, and query positions. This is descriptive only.
        result["attention_mass_over_image_keys"][name] = float(attention.float()[..., indices].sum(dim=-1).mean())
    return result
