from __future__ import annotations

import json
import hashlib

import pytest
import torch
from PIL import Image
from torch.torch_version import TorchVersion

from vision_trace.analysis import cached_evidence_analysis
from vision_trace.cache import (
    PILOT_CACHE_SCHEMA_VERSION,
    make_pilot_cache_record,
    save_cache,
    sha256_path,
    validate_cache_record,
    validate_integrity_manifest,
    write_integrity_manifest,
)
from vision_trace.geometry import build_image_token_geometry
from vision_trace.manifest import load_image_manifest, sha256_file, verified_image
from vision_trace.real_image_pilot import _processor_metadata


def _schema2_record() -> dict:
    geometry = build_image_token_geometry([[1, 4, 4]], [2, 3, 4, 5])
    tensor = torch.ones((1, 6, 8), dtype=torch.float16)
    attention = torch.full((1, 2, 6, 6), 1 / 6, dtype=torch.float16)
    descriptors = {
        "qwen.language.layer.0.output": {
            "shape": [1, 6, 8], "dtype": "torch.float16", "semantic_role": "language_decoder_output",
            "layer_id": 0, "token_space": "multimodal_sequence", "coordinate_space": "language_sequence",
            "module_path": "model.model.language_model.layers.0", "hook_direction": "forward",
            "decoder_addition_boundary": "not_applicable",
        },
        "qwen.attention.0": {
            "shape": [1, 2, 6, 6], "dtype": "torch.float16", "semantic_role": "attention_probability",
            "layer_id": 0, "token_space": "multimodal_sequence", "coordinate_space": "language_sequence",
            "module_path": "model.model.language_model.layers.0.self_attn", "hook_direction": "forward",
            "decoder_addition_boundary": "not_applicable",
        },
    }
    return make_pilot_cache_record(
        identifiers={"experiment_id": "pilot", "run_id": "run", "sample_id": "image-1", "image_id": "image-1", "manifest_id": "manifest"},
        provenance={"repository_commit": "abc", "extraction_timestamp_utc": "2026-09-05T00:00:00+00:00", "code_package_version": "test", "environment": {"torch": "test"}, "model": {"qwen": {"identifier": "qwen", "revision": "qwen-revision"}, "dino": {"identifier": "dino", "revision": "dino-revision"}}, "processor": {"qwen": {"identifier": "qwen", "revision": "qwen-revision", "config_sha256": "b" * 64}, "dino": {"identifier": "dino", "revision": "dino-revision", "config_sha256": "c" * 64}}, "source_image": {"source": "fixture.png", "sha256": "a" * 64, "original_width": 32, "original_height": 16}},
        input_metadata={"prompt_template_id": "prompt", "resolved_width": 64, "resolved_height": 64, "image_grid_thw": [1, 4, 4], "patch_grid_thw": [1, 4, 4], "merged_grid_thw": [1, 2, 2], "image_token_sequence_positions": list(geometry.image_token_sequence_indices), "text_token_sequence_positions": [0, 1], "token_coordinates": {"patch": [list(value) for value in geometry.patch_token_coordinates_thw], "merged": [list(value) for value in geometry.token_coordinates_thw]}, "dino_geometry": {"patch_grid_hw": [2, 2]}},
        extraction={"batch_size": 1}, attention={"layout": "(B,H,S,S)"},
        tensors={"qwen.language.layer.0.output": tensor, "qwen.attention.0": attention}, tensor_descriptors=descriptors,
    )


def test_manifest_hash_and_image_metadata_are_verified(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 16), color=(10, 20, 30)).save(image_path)
    manifest_path = tmp_path / "images.jsonl"
    manifest_path.write_text(json.dumps({"image_id": "sample", "source": "sample.png", "sha256": sha256_file(image_path), "original_width": 32, "original_height": 16}) + "\n")
    entry = load_image_manifest(manifest_path)[0]
    assert verified_image(entry, manifest_directory=tmp_path).size == (32, 16)
    image_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verified_image(entry, manifest_directory=tmp_path)


def test_schema2_round_trip_integrity_and_cpu_analysis(tmp_path) -> None:
    record = _schema2_record()
    validate_cache_record(record)
    cache_path = save_cache(tmp_path / "samples" / "image-1.pt", record)
    extraction_hash = hashlib.sha256(json.dumps(record["extraction"], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    manifest_path = write_integrity_manifest(tmp_path / "integrity_manifest.json", [{"image_id": "image-1", "source_sha256": "a" * 64, "cache_path": "samples/image-1.pt", "cache_sha256": sha256_path(cache_path), "schema_version": PILOT_CACHE_SCHEMA_VERSION, "repository_commit": "abc", "extraction_configuration_sha256": extraction_hash}])
    integrity = validate_integrity_manifest(manifest_path)
    assert integrity["records"][0]["image_id"] == "image-1"
    analysis = cached_evidence_analysis(record)
    assert analysis["attention_mass_over_image_keys"]["qwen.attention.0"] == pytest.approx(2 / 3, abs=1e-3)


def test_cpu_cache_loader_handles_legacy_torch_version_provenance(tmp_path) -> None:
    record = _schema2_record()
    record["provenance"]["environment"] = {"torch": TorchVersion("2.6.0")}
    record["provenance"]["processor"]["dino"]["config"] = {"resample": Image.Resampling.BICUBIC}
    cache_path = save_cache(tmp_path / "legacy-torch-version.pt", record)
    from vision_trace.cache import load_cache_cpu

    assert load_cache_cpu(cache_path)["provenance"]["environment"]["torch"] == "2.6.0"


def test_processor_metadata_serializes_pillow_enums_to_primitives() -> None:
    class Processor:
        def to_dict(self):
            return {"resample": Image.Resampling.BICUBIC, "nested": (Image.Resampling.NEAREST,)}

    metadata = _processor_metadata(Processor(), identifier="test", revision="revision")
    assert metadata["config"] == {"resample": "PIL.Image.Resampling.BICUBIC", "nested": ["PIL.Image.Resampling.NEAREST"]}


def test_schema2_rejects_missing_spatial_metadata() -> None:
    record = _schema2_record()
    del record["input"]["token_coordinates"]
    with pytest.raises(ValueError, match="input missing"):
        validate_cache_record(record)
