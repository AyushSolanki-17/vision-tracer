"""Qwen3-VL synthetic instrumentation; model imports occur only here."""

from __future__ import annotations

import gc
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .cache import make_cache_record, save_cache
from .geometry import build_image_token_geometry
from .synthetic import SyntheticImageSpec, render_synthetic_image


QWEN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
# Validated smoke-test revision. Pinning prevents a moving Hub revision from
# silently changing the instrumentation target.
QWEN_MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
VISION_LAYER_INDICES = (0, 12, 23)
LANGUAGE_LAYER_INDICES = (0, 1, 2, 3, 18, 35)
ATTENTION_LAYER_INDICES = (0, 18, 35)
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
    """Forward hooks that retain detached CPU copies only."""

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


def _git_commit(repository_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return os.environ.get("VISION_TRACE_GIT_COMMIT")


def _cuda_memory() -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _qwen_inputs(processor: Any, image: Any, device: torch.device) -> dict[str, torch.Tensor]:
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[text], images=[image], return_tensors="pt", max_pixels=MAX_PIXELS)
    inputs = {name: value.to(device) for name, value in encoded.items() if isinstance(value, torch.Tensor)}
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    return inputs


def _validate_capture_shapes(
    captures: dict[str, torch.Tensor], attentions: dict[str, torch.Tensor], *, patch_count: int,
    merged_token_count: int, visual_width: int, sequence_length: int, language_width: int, query_heads: int,
) -> None:
    expected_visual = {"visual.patch_embed", *(f"visual.blocks.{index}" for index in VISION_LAYER_INDICES)}
    expected_language = {f"language.layers.{index}" for index in LANGUAGE_LAYER_INDICES}
    if set(captures) != expected_visual | expected_language | {"visual.merger"}:
        raise ValueError(f"unexpected capture set: {sorted(captures)}")
    for name in expected_visual:
        if tuple(captures[name].shape) != (patch_count, visual_width):
            raise ValueError(f"unexpected {name} shape: {tuple(captures[name].shape)}")
    if tuple(captures["visual.merger"].shape) != (merged_token_count, language_width):
        raise ValueError(f"unexpected visual.merger shape: {tuple(captures['visual.merger'].shape)}")
    for name in expected_language:
        if tuple(captures[name].shape) != (1, sequence_length, language_width):
            raise ValueError(f"unexpected {name} shape: {tuple(captures[name].shape)}")
    for index in ATTENTION_LAYER_INDICES:
        name = f"language.layers.{index}"
        if tuple(attentions[name].shape) != (1, query_heads, sequence_length, sequence_length):
            raise ValueError(f"unexpected {name} attention shape: {tuple(attentions[name].shape)}")


def run_synthetic_instrumentation(
    *, output_directory: str | Path, image_specifications: tuple[SyntheticImageSpec, ...],
    repository_root: str | Path,
) -> dict[str, Any]:
    """Run batch-one FP16 extraction and write one compact cache per image.

    This is intentionally Kaggle/GPU-only.  Callers must run CPU tests before
    invoking it and must reload its outputs via ``load_cache_cpu`` after this
    function returns and the model is released.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("synthetic instrumentation requires a CUDA Kaggle runtime")
    # Keep Qwen imports local so cache reload/analysis paths do not depend on it.
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    root = Path(repository_root)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    # Kaggle's PyTorch build rejects torch.device as the optional argument to
    # reset_peak_memory_stats. Make device 0 current and use the portable
    # no-argument form for all per-device memory calls below.
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, revision=QWEN_MODEL_REVISION)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, revision=QWEN_MODEL_REVISION, torch_dtype=torch.float16, attn_implementation="eager"
    ).to(device).eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    visual = model.model.visual
    language_layers = model.model.language_model.layers
    hooks = CpuHookCapture()
    hooks.add("visual.patch_embed", visual.patch_embed)
    for index in VISION_LAYER_INDICES:
        hooks.add(f"visual.blocks.{index}", visual.blocks[index])
    hooks.add("visual.merger", visual.merger)
    for index in LANGUAGE_LAYER_INDICES:
        hooks.add(f"language.layers.{index}", language_layers[index])

    cache_paths: list[str] = []
    image_reports: list[dict[str, Any]] = []
    model_revision = getattr(model.config, "_commit_hash", None) or QWEN_MODEL_REVISION
    try:
        for spec in image_specifications:
            hooks.tensors.clear()
            image = render_synthetic_image(spec)
            inputs = _qwen_inputs(processor, image, device)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                # Eager materializes all layers transiently in the model API;
                # only the three selected maps are copied into the cache.
                outputs = model(**inputs, output_attentions=True, output_hidden_states=False, return_dict=True)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            image_mask = inputs["input_ids"][0] == model.config.image_token_id
            sequence_indices = torch.nonzero(image_mask, as_tuple=False).flatten().to("cpu")
            geometry = build_image_token_geometry(inputs["image_grid_thw"], sequence_indices)
            raw_selected_attentions = {index: outputs.attentions[index] for index in ATTENTION_LAYER_INDICES}
            if any(value is None for value in raw_selected_attentions.values()):
                raise RuntimeError("eager attention did not return all selected language-layer maps")
            selected_attentions = {
                f"language.layers.{index}": value.detach().to("cpu").contiguous().clone()
                for index, value in raw_selected_attentions.items()
            }
            patch_count = int(torch.as_tensor(geometry.image_grid_thw).prod().item())
            _validate_capture_shapes(
                hooks.tensors, selected_attentions, patch_count=patch_count,
                merged_token_count=len(geometry.image_token_sequence_indices),
                visual_width=int(visual.config.hidden_size), sequence_length=int(inputs["input_ids"].shape[1]),
                language_width=int(model.config.text_config.hidden_size),
                query_heads=int(model.config.text_config.num_attention_heads),
            )
            metadata = {
                "experiment_identifier": "synthetic_instrumentation",
                "image_identifier": spec.identifier,
                "synthetic_image": spec.metadata(),
                "model": {
                    "identifier": QWEN_MODEL_ID, "revision": model_revision,
                    "transformers_version": transformers.__version__,
                },
                "repository_commit": _git_commit(root),
                "geometry": geometry.metadata(),
                "attention": {
                    "implementation": "eager", "selected_language_layers": list(ATTENTION_LAYER_INDICES),
                    "query_heads": int(model.config.text_config.num_attention_heads),
                    "kv_heads": int(model.config.text_config.num_key_value_heads),
                    "sequence_length": int(inputs["input_ids"].shape[1]),
                    "layout": "(batch, query_heads, query_sequence, key_sequence)",
                },
                "extraction_configuration": {
                    "dtype": "float16", "device": str(device), "batch_size": 1, "max_pixels": MAX_PIXELS,
                    "prompt": PROMPT, "vision_layers": list(VISION_LAYER_INDICES),
                    "language_layers": list(LANGUAGE_LAYER_INDICES),
                    "attention_layers": list(ATTENTION_LAYER_INDICES),
                },
            }
            record = make_cache_record(metadata=metadata, representations=hooks.tensors, attentions=selected_attentions)
            cache_path = save_cache(output / f"{spec.identifier}.pt", record)
            cache_paths.append(str(cache_path))
            image_reports.append({
                "image_identifier": spec.identifier, "runtime_seconds": elapsed,
                "cache_bytes": cache_path.stat().st_size, "memory": _cuda_memory(),
                "representation_shapes": record["tensor_descriptors"]["representations"],
                "attention_shapes": record["tensor_descriptors"]["attentions"],
            })
            del outputs, inputs, selected_attentions, record
    finally:
        hooks.close()
        # Explicit release is part of this instrumentation contract.
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "experiment_identifier": "synthetic_instrumentation",
        "model_loading_seconds": load_seconds,
        "cache_paths": cache_paths,
        "images": image_reports,
        "post_release_memory": _cuda_memory(),
        "environment": {"python": platform.python_version(), "torch_version": torch.__version__},
    }
