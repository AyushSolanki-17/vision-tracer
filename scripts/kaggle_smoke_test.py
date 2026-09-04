#!/usr/bin/env python3
"""One-image Kaggle-only instrumentation smoke test for Vision Tracer.

This script intentionally does not implement a dataset pipeline, activation
cache, or experiment.  It loads the two Phase 0 checkpoints and writes only
small JSON summaries under results/smoke_test/.
"""

from __future__ import annotations

import inspect
import json
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
        return None


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


def make_asymmetric_image(height: int, width: int) -> Image.Image:
    """Make a deterministic spatially labelled RGB image without downloading data."""
    y, x = np.indices((height, width))
    # Unequal gradients plus four differently colored corners make orientation
    # visible if this image is displayed while investigating a failed mapping.
    pixels = np.stack(
        [((x * 251) // max(width - 1, 1)).astype(np.uint8),
         ((y * 241) // max(height - 1, 1)).astype(np.uint8),
         (((3 * x + 5 * y) * 239 // max(3 * width + 5 * height - 8, 1))).astype(np.uint8)],
        axis=-1,
    )
    pixels[:16, :16] = (255, 0, 0)
    pixels[:16, -16:] = (0, 255, 0)
    pixels[-16:, :16] = (0, 0, 255)
    pixels[-16:, -16:] = (255, 255, 0)
    return Image.fromarray(pixels, mode="RGB")


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


def geometry_report(inputs: dict[str, torch.Tensor], model, merger_capture: torch.Tensor) -> dict[str, Any]:
    grid = inputs["image_grid_thw"].detach().cpu().tolist()
    mask_count = int((inputs["input_ids"] == model.config.image_token_id).sum().item())
    # The merger reduces each temporal/spatial 2x2 group.  This declares the
    # candidate coordinates; the runtime source and counts below are retained
    # so a row-major conclusion is evidence-backed rather than assumed.
    t, patch_rows, patch_cols = grid[0]
    rows, cols = patch_rows // 2, patch_cols // 2
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
        "candidate_row_major_mapping": mapping,
        "count_agreement": mask_count == int(merger_capture.shape[0]) == int(t * rows * cols),
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
    }

    hooks = TemporaryHooks()
    visual = model.model.visual
    layers = model.model.language_model.layers
    for name, module in {
        "visual.blocks.0": visual.blocks[0],
        "visual.merger": visual.merger,
        "language_model.layers.0": layers[0],
        "language_model.layers.3": layers[3],
        "language_model.layers.18": layers[18],
        "language_model.layers.35": layers[35],
    }.items():
        hooks.forward(name, module)
    # These additional pre-hooks establish the DeepStack parent-loop boundary.
    for index in (1, 2, 3):
        hooks.pre_forward(f"language_model.layers.{index}.input", layers[index])

    primary_image = make_asymmetric_image(256, 384)
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

    # A second shape is intentionally run without attention output to keep the
    # eager quadratic allocation confined to exactly one inspection forward.
    second_inputs = qwen_inputs(processor, make_asymmetric_image(384, 256), device)
    with torch.inference_mode():
        model(**second_inputs, output_attentions=False, output_hidden_states=False, return_dict=True)
    second_merger = hooks.captures["visual.merger"]
    summary["visual_token_ordering"] = {
        "method": "Two asymmetric synthetic images; processor grid/mask/merger counts plus the installed merger forward source are recorded. Confirm the candidate mapping only if its source shows the corresponding flatten order.",
        "merger_forward_source": module_source_summary(visual.merger),
        "first_asymmetric_image": geometry_report(inputs, model, merger_capture),
        "second_asymmetric_image": geometry_report(second_inputs, model, second_merger),
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
