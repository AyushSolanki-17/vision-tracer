from __future__ import annotations

import hashlib

import torch

from vision_trace.analysis import cached_evidence_analysis
from vision_trace.cache import load_cache_cpu, make_cache_record, save_cache
from vision_trace.geometry import build_image_token_geometry
from vision_trace.synthetic import SYNTHETIC_IMAGES, render_synthetic_image
from vision_trace.synthetic_instrumentation import (
    ATTENTION_LAYER_INDICES,
    LANGUAGE_LAYER_INDICES,
    VISION_LAYER_INDICES,
    _validate_capture_shapes,
)


def _digest(spec) -> str:
    image = render_synthetic_image(spec)
    return hashlib.sha256(image.tobytes()).hexdigest()


def test_synthetic_images_are_deterministic() -> None:
    assert [_digest(spec) for spec in SYNTHETIC_IMAGES] == [_digest(spec) for spec in SYNTHETIC_IMAGES]
    assert len(set(_digest(spec) for spec in SYNTHETIC_IMAGES)) == len(SYNTHETIC_IMAGES)


def test_image_token_indices_match_merged_grid() -> None:
    geometry = build_image_token_geometry(torch.tensor([[1, 16, 24]]), torch.arange(10, 106))
    assert geometry.merged_grid_thw == (1, 8, 12)
    assert len(geometry.image_token_sequence_indices) == 96
    assert geometry.token_coordinates_thw[0] == (0, 0, 0)
    assert geometry.token_coordinates_thw[-1] == (0, 7, 11)


def test_capture_shape_validation_accepts_observed_qwen_layout() -> None:
    patch_count, merged_count, sequence_length = 384, 96, 114
    captures = {"visual.patch_embed": torch.zeros((patch_count, 1024), dtype=torch.float16)}
    captures.update({f"visual.blocks.{index}": torch.zeros((patch_count, 1024), dtype=torch.float16) for index in VISION_LAYER_INDICES})
    captures["visual.merger"] = torch.zeros((merged_count, 2560), dtype=torch.float16)
    captures.update({f"language.layers.{index}": torch.zeros((1, sequence_length, 2560), dtype=torch.float16) for index in LANGUAGE_LAYER_INDICES})
    attentions = {f"language.layers.{index}": torch.zeros((1, 32, sequence_length, sequence_length), dtype=torch.float16) for index in ATTENTION_LAYER_INDICES}
    _validate_capture_shapes(
        captures, attentions, patch_count=patch_count, merged_token_count=merged_count,
        visual_width=1024, sequence_length=sequence_length, language_width=2560, query_heads=32,
    )


def test_cache_round_trip_and_cpu_analysis(tmp_path) -> None:
    geometry = build_image_token_geometry([[1, 4, 4]], [3, 4, 5, 6])
    representations = {
        "visual.patch_embed": torch.arange(32, dtype=torch.float16).reshape(4, 8),
        "visual.merger": torch.ones((4, 8), dtype=torch.float16),
        "language.layers.0": torch.ones((1, 9, 8), dtype=torch.float16),
    }
    attention = torch.zeros((1, 2, 9, 9), dtype=torch.float16)
    attention[..., [3, 4, 5, 6]] = 0.25
    record = make_cache_record(
        metadata={
            "experiment_identifier": "test", "image_identifier": "synthetic",
            "model": {"identifier": "mock", "revision": "test", "transformers_version": "not-imported"},
            "geometry": geometry.metadata(), "extraction_configuration": {"batch_size": 1},
        },
        representations=representations,
        attentions={"language.layers.0": attention},
    )
    path = save_cache(tmp_path / "evidence.pt", record)
    del record, representations, attention
    reloaded = load_cache_cpu(path)
    assert all(tensor.device.type == "cpu" for tensor in reloaded["representations"].values())
    assert reloaded["representations"]["visual.patch_embed"].shape == (4, 8)
    analysis = cached_evidence_analysis(reloaded)
    assert analysis["attention_mass_over_image_keys"]["language.layers.0"] == 1.0
