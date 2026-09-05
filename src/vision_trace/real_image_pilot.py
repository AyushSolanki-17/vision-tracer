"""Kaggle-only batch-one evidence extraction for an explicit image manifest."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from .cache import (
    PILOT_CACHE_SCHEMA_VERSION,
    make_pilot_cache_record,
    save_cache,
    sha256_path,
    validate_integrity_manifest,
    write_integrity_manifest,
)
from .geometry import build_image_token_geometry
from .manifest import ImageManifestEntry, load_image_manifest, verified_image
from . import __version__


QWEN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DINO_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
VISION_LAYER_INDICES = (0, 6, 12, 18, 23)
DEEPSTACK_SOURCE_INDICES = (5, 11, 17)
LANGUAGE_LAYER_INDICES = (0, 1, 2, 3, 6, 12, 18, 24, 30, 35)
ATTENTION_LAYER_INDICES = (0, 3, 18, 35)
DINO_INTERMEDIATE_LAYER_INDICES = (0, 6)
PROMPT_TEMPLATE_ID = "qwen3_vl_single_image_short_phrase_v1"
PROMPT = "Describe the image in one short phrase."
MAX_PIXELS = 262_144


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    if hasattr(value, "last_hidden_state"):
        return _first_tensor(value.last_hidden_state)
    raise TypeError(f"no tensor in {type(value).__name__}")


class CpuHookCapture:
    """Temporary hooks that retain detached CPU tensors only."""

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    def add(self, name: str, module: torch.nn.Module) -> None:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            self.tensors[name] = _first_tensor(output).detach().to("cpu").contiguous().clone()

        self._handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_hash(value: Any) -> str:
    return _sha256_text(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")))


def _json_primitive(value: Any) -> Any:
    """Convert processor metadata to values safe for PyTorch's safe loader."""
    if isinstance(value, Enum):
        return f"{type(value).__module__}.{type(value).__qualname__}.{value.name}"
    if isinstance(value, dict):
        return {str(key): _json_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_primitive(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("VISION_TRACE_GIT_COMMIT")


def _cuda_memory() -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _processor_metadata(processor: Any, *, identifier: str, revision: str) -> dict[str, Any]:
    image_processor = getattr(processor, "image_processor", processor)
    config = _json_primitive(image_processor.to_dict()) if hasattr(image_processor, "to_dict") else {}
    return {"identifier": identifier, "revision": revision, "config": config, "config_sha256": _json_hash(config)}


def _descriptor(
    tensor: torch.Tensor, *, semantic_role: str, layer_id: int | str, token_space: str,
    coordinate_space: str, module_path: str, hook_direction: str = "forward",
    decoder_addition_boundary: str = "not_applicable",
) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype), "semantic_role": semantic_role,
        "layer_id": layer_id, "token_space": token_space, "coordinate_space": coordinate_space,
        "module_path": module_path, "hook_direction": hook_direction,
        "decoder_addition_boundary": decoder_addition_boundary,
    }


def _qwen_inputs(processor: Any, image: Any, device: torch.device) -> tuple[dict[str, torch.Tensor], str]:
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    rendered = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[rendered], images=[image], return_tensors="pt", max_pixels=MAX_PIXELS)
    inputs = {name: value.to(device) for name, value in encoded.items() if isinstance(value, torch.Tensor)}
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    return inputs, rendered


def _register_qwen_hooks(model: Any) -> CpuHookCapture:
    visual = model.model.visual
    hooks = CpuHookCapture()
    hooks.add("qwen.visual.patch_embed", visual.patch_embed)
    for index in VISION_LAYER_INDICES:
        hooks.add(f"qwen.visual.block.{index}", visual.blocks[index])
    hooks.add("qwen.visual.merger", visual.merger)
    for source_index, merger in zip(DEEPSTACK_SOURCE_INDICES, visual.deepstack_merger_list, strict=True):
        hooks.add(f"qwen.visual.deepstack.{source_index}", merger)
    for index in LANGUAGE_LAYER_INDICES:
        hooks.add(f"qwen.language.layer.{index}.output", model.model.language_model.layers[index])
    return hooks


def _validate_qwen_captures(
    captures: dict[str, torch.Tensor], attentions: dict[str, torch.Tensor], *, patch_count: int,
    merged_count: int, sequence_length: int, visual_width: int, language_width: int, query_heads: int,
) -> None:
    expected = {"qwen.visual.patch_embed", "qwen.visual.merger"}
    expected |= {f"qwen.visual.block.{index}" for index in VISION_LAYER_INDICES}
    expected |= {f"qwen.visual.deepstack.{index}" for index in DEEPSTACK_SOURCE_INDICES}
    expected |= {f"qwen.language.layer.{index}.output" for index in LANGUAGE_LAYER_INDICES}
    if set(captures) != expected:
        raise ValueError(f"unexpected Qwen capture set: {sorted(captures)}")
    if set(attentions) != {f"qwen.attention.{index}" for index in ATTENTION_LAYER_INDICES}:
        raise ValueError(f"unexpected Qwen attention capture set: {sorted(attentions)}")
    for name in {"qwen.visual.patch_embed", *(f"qwen.visual.block.{index}" for index in VISION_LAYER_INDICES)}:
        if tuple(captures[name].shape) != (patch_count, visual_width):
            raise ValueError(f"unexpected Qwen shape for {name}: {tuple(captures[name].shape)}")
    for name in {"qwen.visual.merger", *(f"qwen.visual.deepstack.{index}" for index in DEEPSTACK_SOURCE_INDICES)}:
        if tuple(captures[name].shape) != (merged_count, language_width):
            raise ValueError(f"unexpected Qwen shape for {name}: {tuple(captures[name].shape)}")
    for index in LANGUAGE_LAYER_INDICES:
        name = f"qwen.language.layer.{index}.output"
        if tuple(captures[name].shape) != (1, sequence_length, language_width):
            raise ValueError(f"unexpected Qwen shape for {name}: {tuple(captures[name].shape)}")
    for index, tensor in attentions.items():
        if tuple(tensor.shape) != (1, query_heads, sequence_length, sequence_length):
            raise ValueError(f"unexpected Qwen attention shape for {index}: {tuple(tensor.shape)}")


def _qwen_descriptors(captures: dict[str, torch.Tensor], attentions: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, tensor in captures.items():
        if name == "qwen.visual.patch_embed":
            result[name] = _descriptor(tensor, semantic_role="vision_patch_embedding", layer_id="patch_embed", token_space="patch", coordinate_space="qwen_patch_grid", module_path="model.model.visual.patch_embed")
        elif ".visual.block." in name:
            index = int(name.rsplit(".", 1)[1])
            result[name] = _descriptor(tensor, semantic_role="vision_block_output", layer_id=index, token_space="patch", coordinate_space="qwen_patch_grid", module_path=f"model.model.visual.blocks.{index}")
        elif ".deepstack." in name:
            index = int(name.rsplit(".", 1)[1])
            result[name] = _descriptor(tensor, semantic_role="deepstack_merged_feature", layer_id=index, token_space="merged_visual", coordinate_space="qwen_merged_grid", module_path=f"model.model.visual.deepstack_merger_list[{DEEPSTACK_SOURCE_INDICES.index(index)}]")
        elif name == "qwen.visual.merger":
            result[name] = _descriptor(tensor, semantic_role="merged_visual_feature", layer_id="merger", token_space="merged_visual", coordinate_space="qwen_merged_grid", module_path="model.model.visual.merger")
        else:
            index = int(name.split(".")[3])
            result[name] = _descriptor(tensor, semantic_role="language_decoder_output", layer_id=index, token_space="multimodal_sequence", coordinate_space="language_sequence", module_path=f"model.model.language_model.layers.{index}", decoder_addition_boundary="decoder_output_before_parent_deepstack_addition_for_layers_0_to_2")
    for name, tensor in attentions.items():
        index = int(name.rsplit(".", 1)[1])
        result[name] = _descriptor(tensor, semantic_role="attention_probability", layer_id=index, token_space="multimodal_sequence", coordinate_space="language_sequence", module_path=f"model.model.language_model.layers.{index}.self_attn")
    return result


def _environment(transformers: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(), "torch": str(torch.__version__), "transformers": str(transformers.__version__),
        "cuda": str(torch.version.cuda), "gpu_name": torch.cuda.get_device_name(0),
    }


def run_real_image_pilot(
    *, manifest_path: str | Path, output_directory: str | Path, run_id: str, dino_revision: str,
    repository_root: str | Path, max_images: int | None = None,
) -> dict[str, Any]:
    """Execute the complete Qwen→DINO→CPU validation pipeline in one session."""
    if not torch.cuda.is_available():
        raise RuntimeError("real-image pilot requires a CUDA Kaggle runtime")
    from huggingface_hub import model_info
    import transformers
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration

    entries = load_image_manifest(manifest_path)
    if max_images is not None:
        entries = entries[:max_images]
    if not 1 <= len(entries) <= 20:
        raise ValueError("real-image pilot manifest must select between 1 and 20 images")
    root, manifest = Path(repository_root), Path(manifest_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    extraction_timestamp_utc = datetime.now(timezone.utc).isoformat()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    environment = _environment(transformers)
    commit = _git_commit(root)
    if commit is None:
        raise RuntimeError("repository commit is unavailable; set VISION_TRACE_GIT_COMMIT before extraction")
    # Resolve an explicit revision to an immutable commit before any DINO files
    # are fetched. A gated-model authorization failure stops the pilot early.
    resolved_dino_revision = model_info(DINO_MODEL_ID, revision=dino_revision).sha

    qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, revision=QWEN_MODEL_REVISION)
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(QWEN_MODEL_ID, revision=QWEN_MODEL_REVISION, torch_dtype=torch.float16, attn_implementation="eager").to(device).eval()
    qwen_metadata = {"identifier": QWEN_MODEL_ID, "revision": getattr(qwen_model.config, "_commit_hash", None) or QWEN_MODEL_REVISION, "config_sha256": _json_hash(qwen_model.config.to_dict())}
    qwen_processor_metadata = _processor_metadata(qwen_processor, identifier=QWEN_MODEL_ID, revision=QWEN_MODEL_REVISION)
    qwen_hooks = _register_qwen_hooks(qwen_model)
    qwen_records: dict[str, dict[str, Any]] = {}
    try:
        for entry in entries:
            qwen_hooks.tensors.clear()
            image = verified_image(entry, manifest_directory=manifest.parent)
            inputs, rendered_prompt = _qwen_inputs(qwen_processor, image, device)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = qwen_model(**inputs, output_attentions=True, output_hidden_states=False, return_dict=True)
            torch.cuda.synchronize(device)
            image_positions = torch.nonzero(inputs["input_ids"][0] == qwen_model.config.image_token_id, as_tuple=False).flatten().to("cpu")
            geometry = build_image_token_geometry(inputs["image_grid_thw"], image_positions)
            raw_attention = {index: outputs.attentions[index] for index in ATTENTION_LAYER_INDICES}
            if any(value is None for value in raw_attention.values()):
                raise RuntimeError("eager attention did not return all requested pilot layers")
            attention = {f"qwen.attention.{index}": value.detach().to("cpu").contiguous().clone() for index, value in raw_attention.items()}
            _validate_qwen_captures(qwen_hooks.tensors, attention, patch_count=len(geometry.patch_token_coordinates_thw), merged_count=len(geometry.token_coordinates_thw), sequence_length=int(inputs["input_ids"].shape[1]), visual_width=int(qwen_model.model.visual.config.hidden_size), language_width=int(qwen_model.config.text_config.hidden_size), query_heads=int(qwen_model.config.text_config.num_attention_heads))
            grid = geometry.image_grid_thw
            sequence_length = int(inputs["input_ids"].shape[1])
            image_position_set = set(geometry.image_token_sequence_indices)
            qwen_records[entry.image_id] = {
                "entry": entry,
                "tensors": {**qwen_hooks.tensors, **attention},
                "descriptors": _qwen_descriptors(qwen_hooks.tensors, attention),
                "input": {
                    "prompt_template_id": PROMPT_TEMPLATE_ID, "prompt_template_sha256": _sha256_text(PROMPT),
                    "rendered_prompt_sha256": _sha256_text(rendered_prompt), "batch_index": 0,
                    "resolved_width": grid[2] * 16, "resolved_height": grid[1] * 16,
                    "image_grid_thw": list(grid), "patch_grid_thw": list(grid), "merged_grid_thw": list(geometry.merged_grid_thw),
                    "patch_token_count": len(geometry.patch_token_coordinates_thw), "merged_visual_token_count": len(geometry.token_coordinates_thw),
                    "image_token_sequence_positions": list(geometry.image_token_sequence_indices),
                    "text_token_sequence_positions": [index for index in range(sequence_length) if index not in image_position_set],
                    "token_coordinates": {"patch": [list(item) for item in geometry.patch_token_coordinates_thw], "merged": [list(item) for item in geometry.token_coordinates_thw]},
                },
                "runtime_seconds": time.perf_counter() - started,
                "memory": _cuda_memory(),
            }
            del outputs, inputs, raw_attention, attention
    finally:
        qwen_hooks.close()
        del qwen_model, qwen_processor
        gc.collect()
        torch.cuda.empty_cache()
    qwen_release = _cuda_memory()
    if qwen_release["allocated_bytes"] >= 1_000_000_000:
        raise RuntimeError("Qwen release verification failed; refusing to load DINO")

    dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID, revision=resolved_dino_revision)
    dino_model = AutoModel.from_pretrained(DINO_MODEL_ID, revision=resolved_dino_revision, torch_dtype=torch.float16).to(device).eval()
    dino_metadata = {"identifier": DINO_MODEL_ID, "revision": resolved_dino_revision, "config_sha256": _json_hash(dino_model.config.to_dict())}
    integrity_records: list[dict[str, Any]] = []
    try:
        for entry in entries:
            image = verified_image(entry, manifest_directory=manifest.parent)
            encoded = dino_processor(images=image, return_tensors="pt")
            dino_inputs = {name: value.to(device, dtype=torch.float16) if value.is_floating_point() else value.to(device) for name, value in encoded.items()}
            started = time.perf_counter()
            with torch.inference_mode():
                dino_outputs = dino_model(**dino_inputs, output_hidden_states=True, return_dict=True)
            torch.cuda.synchronize(device)
            patch_size = int(dino_model.config.patch_size)
            resolved_height, resolved_width = (int(value) for value in dino_inputs["pixel_values"].shape[-2:])
            patch_rows, patch_cols = resolved_height // patch_size, resolved_width // patch_size
            dino_tensors: dict[str, torch.Tensor] = {}
            dino_descriptors: dict[str, dict[str, Any]] = {}
            for index in DINO_INTERMEDIATE_LAYER_INDICES:
                state = dino_outputs.hidden_states[index + 1].detach().to("cpu").contiguous().clone()
                patch_name = f"dino.layer.{index}.patch_tokens"
                dino_tensors[patch_name] = state[:, 1 + int(dino_model.config.num_register_tokens):]
                dino_descriptors[patch_name] = _descriptor(dino_tensors[patch_name], semantic_role="dino_patch_tokens", layer_id=index, token_space="dino_patch", coordinate_space="dino_patch_grid", module_path=f"model.encoder.layer.{index}")
            final = dino_outputs.last_hidden_state.detach().to("cpu").contiguous().clone()
            register_count = int(dino_model.config.num_register_tokens)
            dino_tensors["dino.final.patch_tokens"] = final[:, 1 + register_count:]
            dino_tensors["dino.final.cls_token"] = final[:, :1]
            dino_descriptors["dino.final.patch_tokens"] = _descriptor(dino_tensors["dino.final.patch_tokens"], semantic_role="dino_final_patch_tokens", layer_id="final", token_space="dino_patch", coordinate_space="dino_patch_grid", module_path="model.norm")
            dino_descriptors["dino.final.cls_token"] = _descriptor(dino_tensors["dino.final.cls_token"], semantic_role="dino_final_cls_token", layer_id="final", token_space="dino_cls", coordinate_space="none", module_path="model.norm")
            payload = qwen_records[entry.image_id]
            payload["input"]["dino_geometry"] = {"resolved_width": resolved_width, "resolved_height": resolved_height, "patch_size": patch_size, "patch_grid_hw": [patch_rows, patch_cols], "patch_token_count": patch_rows * patch_cols, "cls_index": 0, "register_token_count": register_count, "patch_token_start": 1 + register_count, "patch_coordinates": [[row, col] for row in range(patch_rows) for col in range(patch_cols)]}
            extraction = {"dtype": "float16", "batch_size": 1, "max_pixels": MAX_PIXELS, "attention_implementation": "eager", "qwen_vision_layers": list(VISION_LAYER_INDICES), "deepstack_source_layers": list(DEEPSTACK_SOURCE_INDICES), "qwen_language_layers": list(LANGUAGE_LAYER_INDICES), "attention_layers": list(ATTENTION_LAYER_INDICES), "dino_intermediate_layers": list(DINO_INTERMEDIATE_LAYER_INDICES), "dino_final": True, "model_mode": "eval"}
            record = make_pilot_cache_record(
                identifiers={"experiment_id": "real_image_pilot", "run_id": run_id, "sample_id": entry.image_id, "image_id": entry.image_id, "manifest_id": _json_hash([item.metadata() for item in entries])},
                provenance={"repository_commit": commit, "extraction_timestamp_utc": extraction_timestamp_utc, "code_package_version": __version__, "environment": environment, "model": {"qwen": qwen_metadata, "dino": dino_metadata}, "processor": {"qwen": qwen_processor_metadata, "dino": _processor_metadata(dino_processor, identifier=DINO_MODEL_ID, revision=resolved_dino_revision)}, "source_image": {"source": entry.source, "sha256": entry.sha256, "original_width": entry.original_width, "original_height": entry.original_height}},
                input_metadata=payload["input"], extraction={**extraction, "seed": None},
                attention={"layout": "(batch, query_heads, query_sequence, key_sequence)", "query_head_count": 32, "kv_head_count": 8, "head_dimension": 128, "query_sequence_length": len(payload["input"]["text_token_sequence_positions"]) + len(payload["input"]["image_token_sequence_positions"]), "key_sequence_length": len(payload["input"]["text_token_sequence_positions"]) + len(payload["input"]["image_token_sequence_positions"]), "query_key_position_classes": ["text", "image"]},
                tensors={**payload["tensors"], **dino_tensors}, tensor_descriptors={**payload["descriptors"], **dino_descriptors},
            )
            cache_path = save_cache(output / "samples" / f"{entry.image_id}.pt", record)
            payload["dino_runtime_seconds"] = time.perf_counter() - started
            payload["cache_bytes"] = cache_path.stat().st_size
            integrity_records.append({"image_id": entry.image_id, "source_sha256": entry.sha256, "cache_path": str(cache_path.relative_to(output)), "cache_sha256": sha256_path(cache_path), "schema_version": PILOT_CACHE_SCHEMA_VERSION, "repository_commit": commit, "extraction_configuration_sha256": _json_hash({**extraction, "seed": None})})
            del dino_outputs, dino_inputs, dino_tensors, record
    finally:
        del dino_model, dino_processor
        gc.collect()
        torch.cuda.empty_cache()
    dino_release = _cuda_memory()
    if dino_release["allocated_bytes"] >= 1_000_000_000:
        raise RuntimeError("DINO release verification failed")
    integrity_path = write_integrity_manifest(output / "integrity_manifest.json", integrity_records)
    validate_integrity_manifest(integrity_path)
    return {"run_id": run_id, "schema_version": PILOT_CACHE_SCHEMA_VERSION, "sample_count": len(entries), "integrity_manifest": str(integrity_path), "qwen_release_memory": qwen_release, "dino_release_memory": dino_release, "images": [{"image_id": item.image_id, "qwen_runtime_seconds": qwen_records[item.image_id]["runtime_seconds"], "dino_runtime_seconds": qwen_records[item.image_id]["dino_runtime_seconds"], "cache_bytes": qwen_records[item.image_id]["cache_bytes"], "memory": qwen_records[item.image_id]["memory"]} for item in entries], "environment": environment}
