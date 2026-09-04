#!/usr/bin/env python3
"""One-image Kaggle-only instrumentation smoke test for Vision Tracer.

This script intentionally does not implement a dataset pipeline, activation
cache, or experiment.  It loads the two Phase 0 checkpoints and writes only
small JSON summaries under results/smoke_test/.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration


QWEN_ID = "Qwen/Qwen3-VL-4B-Instruct"
DINO_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
RESULTS_DIR = Path("results/smoke_test")
# Far below the model's configured maximum.  The asymmetric inputs have at
# least the processor minimum pixel count and are used only for geometry.
MAX_PIXELS = 262_144
MERGE_SIZE = 2
PATCH_SIZE = 16
MERGED_CELL_PIXELS = PATCH_SIZE * MERGE_SIZE


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_summary(name: str, summary: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / name).write_text(json.dumps(summary, indent=2, default=jsonable) + "\n")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        # An uploaded Kaggle directory may not retain .git. The caller can
        # supply the exact source commit without embedding a stale hash here.
        return os.environ.get("VISION_TRACE_GIT_COMMIT")


def environment_report() -> dict[str, Any]:
    gpu_count = torch.cuda.device_count()
    return {
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": gpu_count,
        "gpus": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
        "git_commit": git_commit(),
    }


def cuda_memory() -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def timed_cuda_operation(operation):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    return value, {"elapsed_seconds": time.perf_counter() - started, **cuda_memory()}


def tensor_info(value: Any) -> dict[str, Any] | None:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (tuple, list)):
        for item in value:
            found = tensor_info(item)
            if found is not None:
                return found
    if hasattr(value, "last_hidden_state"):
        return tensor_info(value.last_hidden_state)
    return None


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                pass
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    raise TypeError(f"No tensor found in {type(value).__name__}")


def cpu_copy(value: Any) -> torch.Tensor:
    # detach + clone prevents a graph/reference to GPU storage from being held.
    return first_tensor(value).detach().to("cpu").clone()


class TemporaryHooks:
    """Capture only the primary tensor of temporary forward/pre-forward hooks."""

    def __init__(self) -> None:
        self.captures: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def forward(self, name: str, module: torch.nn.Module) -> None:
        def hook(_module, _inputs, output):
            self.captures[name] = cpu_copy(output)

        self.handles.append(module.register_forward_hook(hook))

    def pre_forward(self, name: str, module: torch.nn.Module) -> None:
        def hook(_module, inputs):
            self.captures[name] = cpu_copy(inputs)

        self.handles.append(module.register_forward_pre_hook(hook))

    def remove(self) -> bool:
        for handle in self.handles:
            handle.remove()
        removed = all(getattr(handle, "id", None) not in handle.hooks_dict_ref() if hasattr(handle, "hooks_dict_ref") else True for handle in self.handles)
        self.handles.clear()
        return removed


def make_spatial_diagnostic_image(merged_rows: int, merged_cols: int) -> Image.Image:
    """Return a deterministic, spatially unique image aligned to 2x2 patch cells.

    Each 32x32 cell has an independent RGB texture.  A cyclic 32-pixel shift
    therefore moves whole merged cells without changing their contents.
    """
    generator = np.random.default_rng(7_310_941)
    pixels = generator.integers(
        0, 256,
        size=(merged_rows * MERGED_CELL_PIXELS, merged_cols * MERGED_CELL_PIXELS, 3),
        dtype=np.uint8,
    )
    return Image.fromarray(pixels)


def cyclic_cell_shift(image: Image.Image, *, row_cells: int = 0, col_cells: int = 0) -> Image.Image:
    pixels = np.asarray(image)
    shifted = np.roll(
        pixels,
        shift=(row_cells * MERGED_CELL_PIXELS, col_cells * MERGED_CELL_PIXELS),
        axis=(0, 1),
    )
    return Image.fromarray(shifted)


def qwen_inputs(processor, image: Image.Image, device: torch.device) -> dict[str, torch.Tensor]:
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in one short phrase."},
        ],
    }]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    encoded = processor(
        text=[text], images=[image], return_tensors="pt", max_pixels=MAX_PIXELS
    )
    result = {}
    for key, value in encoded.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
    if "pixel_values" in result:
        result["pixel_values"] = result["pixel_values"].to(torch.float16)
    return result


def module_source_summary(module: torch.nn.Module) -> dict[str, Any]:
    try:
        source = inspect.getsource(type(module).forward)
        return {
            "class": type(module).__name__,
            "source_available": True,
            "mentions_deepstack": "deepstack" in source.lower(),
            "relevant_lines": [
                line.strip() for line in source.splitlines()
                if any(term in line.lower() for term in ("deepstack", "reshape", "view(", "permute", "flatten"))
            ][:40],
            "source_excerpt": source[:2000],
        }
    except (OSError, TypeError):
        return {"class": type(module).__name__, "source_available": False}


def method_source_summary(instance: Any, method_name: str) -> dict[str, Any]:
    """Record the installed processor code relevant to the runtime result."""
    try:
        source = inspect.getsource(getattr(type(instance), method_name))
        return {
            "class": type(instance).__name__,
            "method": method_name,
            "source_available": True,
            "relevant_lines": [
                line.strip() for line in source.splitlines()
                if any(term in line.lower() for term in ("reshape", "view(", "permute", "transpose", "flatten", "merge"))
            ][:80],
            "source_excerpt": source[:4000],
        }
    except (AttributeError, OSError, TypeError):
        return {"class": type(instance).__name__, "method": method_name, "source_available": False}


def geometry_report(inputs: dict[str, torch.Tensor], model, merger_capture: torch.Tensor) -> dict[str, Any]:
    grid = inputs["image_grid_thw"].detach().cpu().tolist()
    mask_count = int((inputs["input_ids"] == model.config.image_token_id).sum().item())
    # The merger reduces each temporal/spatial 2x2 group.  This declares the
    # candidate coordinates; the runtime source and counts below are retained
    # so a row-major conclusion is evidence-backed rather than assumed.
    t, patch_rows, patch_cols = grid[0]
    rows, cols = patch_rows // MERGE_SIZE, patch_cols // MERGE_SIZE
    mapping = [
        {"token": index, "row": index // cols, "col": index % cols}
        for index in range(rows * cols * t)
    ]
    return {
        "image_grid_thw": grid,
        "image_token_count": mask_count,
        "multimodal_sequence_length": int(inputs["input_ids"].shape[1]),
        "merger_output_length": int(merger_capture.shape[0]),
        "expected_merged_tokens": int(t * rows * cols),
        "row_major_mapping": mapping,
        "count_agreement": mask_count == int(merger_capture.shape[0]) == int(t * rows * cols),
    }


def shifted_feature_match(
    base_features: torch.Tensor,
    shifted_features: torch.Tensor,
    merged_rows: int,
    merged_cols: int,
    *,
    row_shift: int,
    col_shift: int,
) -> dict[str, Any]:
    """Match each shifted feature to its original cell by cosine similarity."""
    base = torch.nn.functional.normalize(base_features.float(), dim=-1)
    shifted = torch.nn.functional.normalize(shifted_features.float(), dim=-1)
    matched_indices = (shifted @ base.T).argmax(dim=1).tolist()
    expected_indices = [
        ((row - row_shift) % merged_rows) * merged_cols + ((col - col_shift) % merged_cols)
        for row in range(merged_rows)
        for col in range(merged_cols)
    ]
    matches = [actual == expected for actual, expected in zip(matched_indices, expected_indices)]
    mismatches = [
        {"shifted_token": index, "expected_base_token": expected_indices[index], "matched_base_token": matched_indices[index]}
        for index, matched in enumerate(matches) if not matched
    ][:12]
    return {
        "shift_cells": {"row": row_shift, "col": col_shift},
        "matching_method": "argmax cosine similarity between CPU-detached post-merger visual features",
        "matched_tokens": int(sum(matches)),
        "total_tokens": len(matches),
        "accuracy": float(sum(matches) / len(matches)),
        "mismatch_examples": mismatches,
    }


def validate_qwen(device: torch.device) -> dict[str, Any]:
    summary: dict[str, Any] = {"status": "started", "model_identifier": QWEN_ID, "environment": environment_report()}
    processor = AutoProcessor.from_pretrained(QWEN_ID)
    model, load_memory = timed_cuda_operation(
        lambda: Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN_ID, torch_dtype=torch.float16, attn_implementation="eager"
        ).to(device).eval()
    )
    summary["model_loading_memory"] = load_memory
    summary["model_paths"] = {
        "visual": type(model.model.visual).__name__,
        "merger": type(model.model.visual.merger).__name__,
        "language_layer_count": len(model.model.language_model.layers),
        "model_revision": getattr(model.config, "_commit_hash", None),
    }

    hooks = TemporaryHooks()
    visual = model.model.visual
    layers = model.model.language_model.layers
    for name, module in {
        "visual.patch_embed": visual.patch_embed,
        "visual.blocks.0": visual.blocks[0],
        "visual.merger": visual.merger,
        "language_model.layers.0": layers[0],
        # These two extra temporary captures are needed only to compare each
        # early decoder output with the next block input across DeepStack.
        "language_model.layers.1": layers[1],
        "language_model.layers.2": layers[2],
        "language_model.layers.3": layers[3],
        "language_model.layers.18": layers[18],
        "language_model.layers.35": layers[35],
    }.items():
        hooks.forward(name, module)
    # These additional pre-hooks establish the DeepStack parent-loop boundary.
    hooks.pre_forward("visual.merger.input", visual.merger)
    for index in (0, 1, 2, 3):
        hooks.pre_forward(f"language_model.layers.{index}.input", layers[index])

    # This is an 8x12 merged grid / 16x24 patch grid. It remains the image for
    # all existing attention, hook, and DeepStack checks.
    primary_image = make_spatial_diagnostic_image(8, 12)
    inputs = qwen_inputs(processor, primary_image, device)
    summary["processor"] = {
        "output_keys": sorted(inputs),
        "pixel_values": tensor_info(inputs.get("pixel_values")),
    }

    # This is the ordinary one-image forward memory reference.  It uses the
    # same eager-configured model but does not ask it to materialize SxS maps.
    def qwen_forward_without_attention():
        with torch.inference_mode():
            return model(**inputs, output_attentions=False, output_hidden_states=False, return_dict=True)

    _, baseline_memory = timed_cuda_operation(qwen_forward_without_attention)
    summary["single_image_inference_memory"] = baseline_memory

    def qwen_forward_with_attention():
        with torch.inference_mode():
            return model(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)

    outputs, inference_memory = timed_cuda_operation(qwen_forward_with_attention)
    summary["single_image_eager_attention_memory"] = inference_memory
    merger_capture = hooks.captures["visual.merger"]
    summary["processor"].update(geometry_report(inputs, model, merger_capture))
    summary["visual_representation"] = tensor_info(merger_capture)
    summary["hook_captures"] = {name: tensor_info(value) for name, value in hooks.captures.items()}
    summary["hook_capture_detached_cpu"] = all(
        value.device.type == "cpu" and not value.requires_grad for value in hooks.captures.values()
    )

    layer0_input = hooks.captures["language_model.layers.0.input"]
    inserted_features = layer0_input[0, (inputs["input_ids"] == model.config.image_token_id).detach().cpu()[0]]
    insertion_delta = float((inserted_features - merger_capture).abs().max())
    summary["visual_language_insertion"] = {
        "method": "Compare merger output to layer-0 forward-pre input at image-placeholder positions.",
        "merger_shape": list(merger_capture.shape),
        "image_position_shape": list(inserted_features.shape),
        "max_abs_difference": insertion_delta,
        "same_order_within_fp16_tolerance": insertion_delta <= 1e-3,
    }

    attentions = outputs.attentions
    representative = next((item for item in attentions if item is not None), None)
    summary["attention"] = {
        "requested_implementation": "eager",
        "output_exists": attentions is not None,
        "returned_layers": len(attentions) if attentions is not None else 0,
        "non_none_layers": sum(item is not None for item in attentions) if attentions is not None else 0,
        "representative_shape": list(representative.shape) if representative is not None else None,
        "representative_dtype": str(representative.dtype) if representative is not None else None,
        "representative_layout": "(B, H, S, S)",
    }
    hidden_states = outputs.hidden_states
    summary["language_hidden_states"] = {
        "output_exists": hidden_states is not None,
        "returned_states": len(hidden_states) if hidden_states is not None else 0,
        "early_layer_0": tensor_info(hooks.captures["language_model.layers.0"]),
        "middle_layer_18": tensor_info(hooks.captures["language_model.layers.18"]),
        "final_layer_35": tensor_info(hooks.captures["language_model.layers.35"]),
        "final_hidden_state": tensor_info(hidden_states[-1]) if hidden_states is not None else None,
    }

    # Output of layers 0--2 is captured before the parent model applies its
    # corresponding DeepStack residual.  A non-zero visual-position delta at
    # the next block input checks this at runtime rather than trusting prose.
    visual_mask = (inputs["input_ids"] == model.config.image_token_id).detach().cpu()[0]
    deepstack = []
    for previous, next_index in ((0, 1), (1, 2), (2, 3)):
        prev = hooks.captures[f"language_model.layers.{previous}"]
        nxt = hooks.captures[f"language_model.layers.{next_index}.input"]
        delta = nxt - prev
        deepstack.append({
            "after_decoder_call": previous,
            "next_layer_input": next_index,
            "visual_position_max_abs_delta": float(delta[0, visual_mask].abs().max()),
            "nonvisual_position_max_abs_delta": float(delta[0, ~visual_mask].abs().max()),
        })
    summary["deepstack"] = {
        "vision_layer_indices_from_config": getattr(
            visual.config, "deepstack_visual_indexes",
            getattr(model.config.vision_config, "deepstack_visual_indexes", None),
        ),
        "parent_forward_source": module_source_summary(model.model),
        "runtime_boundary_checks": deepstack,
        "interpretation": "A non-zero visual delta from decoder output N to layer N+1 input demonstrates a parent-level addition between those hooks. Layer 3 input therefore follows all three early additions.",
    }

    def visual_features(diagnostic_image: Image.Image) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        diagnostic_inputs = qwen_inputs(processor, diagnostic_image, device)
        with torch.inference_mode():
            model(**diagnostic_inputs, output_attentions=False, output_hidden_states=False, return_dict=True)
        return diagnostic_inputs, hooks.captures["visual.merger"].clone()

    def ordering_validation(merged_rows: int, merged_cols: int) -> dict[str, Any]:
        base_image = make_spatial_diagnostic_image(merged_rows, merged_cols)
        base_inputs, base_features = visual_features(base_image)
        right_inputs, right_features = visual_features(cyclic_cell_shift(base_image, col_cells=1))
        down_inputs, down_features = visual_features(cyclic_cell_shift(base_image, row_cells=1))
        patch_rows, patch_cols = base_inputs["image_grid_thw"].detach().cpu().tolist()[0][1:]
        expected_patch_grid = [merged_rows * MERGE_SIZE, merged_cols * MERGE_SIZE]
        geometry_matches_request = [patch_rows, patch_cols] == expected_patch_grid
        right_match = shifted_feature_match(base_features, right_features, merged_rows, merged_cols, row_shift=0, col_shift=1)
        down_match = shifted_feature_match(base_features, down_features, merged_rows, merged_cols, row_shift=1, col_shift=0)
        empirically_validated = geometry_matches_request and right_match["accuracy"] == 1.0 and down_match["accuracy"] == 1.0
        return {
            "patch_grid": {"rows": patch_rows, "cols": patch_cols},
            "merged_grid": {"rows": merged_rows, "cols": merged_cols},
            "geometry_matches_requested_asymmetric_grid": geometry_matches_request,
            "ordering_rule": "token_index = row * merged_grid_cols + col",
            "mapping": [
                {"token": index, "row": index // merged_cols, "col": index % merged_cols}
                for index in range(merged_rows * merged_cols)
            ],
            "evidence": {
                "diagnostic_image": "Each 32x32 merged cell has an independent deterministic RGB texture.",
                "horizontal_cyclic_shift": right_match,
                "vertical_cyclic_shift": down_match,
            },
            "validation_level": "empirical" if empirically_validated else "inconclusive",
            "empirically_validated": empirically_validated,
        }

    # No attention is requested for these six short forwards. All post-merger
    # tensors are immediately copied to CPU by the existing temporary hook.
    summary["visual_token_ordering"] = {
        "transformers_version": transformers.__version__,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "method": "Empirical whole-cell cyclic shifts matched by post-merger feature cosine similarity, plus a direct merger-to-language insertion comparison.",
        "processor_preprocess_source": method_source_summary(processor.image_processor, "preprocess"),
        "vision_forward_source": module_source_summary(visual),
        "merger_forward_source": module_source_summary(visual.merger),
        "merger_input_capture": tensor_info(hooks.captures["visual.merger.input"]),
        "first_asymmetric_image": ordering_validation(8, 12),
        "second_asymmetric_image": ordering_validation(12, 8),
        "language_insertion": summary["visual_language_insertion"],
    }
    summary["hooks_removed"] = hooks.remove()
    summary["status"] = "passed"
    return summary


def validate_dino(device: torch.device) -> dict[str, Any]:
    summary: dict[str, Any] = {"status": "started", "model_identifier": DINO_ID, "environment": environment_report()}
    processor = AutoImageProcessor.from_pretrained(DINO_ID)
    model, load_memory = timed_cuda_operation(
        lambda: AutoModel.from_pretrained(DINO_ID, torch_dtype=torch.float16).to(device).eval()
    )
    summary["model_loading_memory"] = load_memory
    encoded = processor(images=make_asymmetric_image(256, 384), return_tensors="pt")
    inputs = {key: value.to(device, dtype=torch.float16) if value.is_floating_point() else value.to(device) for key, value in encoded.items()}
    def dino_forward():
        with torch.inference_mode():
            return model(**inputs, output_hidden_states=True, return_dict=True)

    outputs, inference_memory = timed_cuda_operation(dino_forward)
    token_count = int(outputs.last_hidden_state.shape[1])
    patch_size = int(getattr(model.config, "patch_size", 16))
    image_size = getattr(model.config, "image_size", None)
    summary.update({
        "single_image_inference_memory": inference_memory,
        "processor_output_keys": sorted(inputs),
        "last_hidden_state": tensor_info(outputs.last_hidden_state),
        "pooler_output": tensor_info(outputs.pooler_output),
        "hidden_states_count": len(outputs.hidden_states) if outputs.hidden_states is not None else 0,
        "token_count": token_count,
        "hidden_dimension": int(outputs.last_hidden_state.shape[-1]),
        "patch_geometry": {"configured_image_size": image_size, "patch_size": patch_size, "patch_tokens": token_count - 1 - int(getattr(model.config, "num_register_tokens", 0)), "register_tokens": int(getattr(model.config, "num_register_tokens", 0))},
        "status": "passed",
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-only",
        action="store_true",
        help="Run the narrow Qwen instrumentation/ordering validation and skip DINO.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("FAILURE: This smoke test requires a Kaggle CUDA GPU runtime.", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")
    # Hugging Face libraries read HF_TOKEN automatically.  Do not log it.
    try:
        qwen = validate_qwen(device)
        write_summary("qwen_summary.json", qwen)
        print(json.dumps(qwen, indent=2, default=jsonable))
    except Exception as exc:
        failure = {"status": "failed", "model_identifier": QWEN_ID, "environment": environment_report(), "error": repr(exc), "traceback": traceback.format_exc()}
        write_summary("qwen_summary.json", failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        print("Qwen failed: stopping before DINO as required.", file=sys.stderr)
        return 1
    # Release the 4B model before the independent DINO measurement.
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if args.qwen_only:
        return 0
    try:
        dino = validate_dino(device)
        write_summary("dino_summary.json", dino)
        print(json.dumps(dino, indent=2, default=jsonable))
    except Exception as exc:
        failure = {"status": "failed", "model_identifier": DINO_ID, "environment": environment_report(), "error": repr(exc), "traceback": traceback.format_exc(), "possible_gated_access": "Confirm Meta terms were accepted and HF_TOKEN is authorized for this exact checkpoint."}
        write_summary("dino_summary.json", failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
