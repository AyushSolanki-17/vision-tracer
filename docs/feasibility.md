# Model Feasibility

## Evidence and scope

No checkpoint, dataset, or activation cache was downloaded or executed locally.
Statements labelled **implementation verified** were read from current upstream
Transformers source/configs on 2026-09-05. Values labelled **estimated** require
the Kaggle smoke test. Future package name: `vision_tracer`.

## Initial environment

Use Kaggle Python 3.11 and retain its CUDA-compatible PyTorch build (PyTorch
>=2.4); record exact Torch, CUDA, driver, and package versions per run. T4 has
no native BF16 tensor cores, so use FP16 for Qwen3-VL.

```text
torch>=2.4                 torchvision>=0.19
transformers>=4.57,<5      accelerate>=1.0
huggingface_hub>=0.26      safetensors>=0.4
numpy>=1.26                scipy>=1.11
scikit-learn>=1.4          matplotlib>=3.8
PyYAML>=6.0                Pillow>=10
```

This is the Phase 0/1 minimum. Do not add `timm`, Flash Attention,
bitsandbytes, the DINOv3 research repository, or a training framework until a
concrete phase needs one. Locally, Python 3.14.4 was inspected; only NumPy
2.5.2 and PyYAML 6.0.3 of this list are installed. Nothing was installed.

Create the environment later from a versioned `pyproject.toml` and lock file,
using a project-local `.venv` (for example, `python -m venv .venv` followed
by `.venv/bin/python -m pip install -e '.[dev]'`). Do not create that
packaging configuration in this documentation-only phase; it must pin the
tested Kaggle resolution of the version ranges above before model execution.

## Selected checkpoints

| Model | Checkpoint | Parameter scale | Expected VRAM | Expected dtype | Why selected | Alternative |
|---|---|---:|---|---|---|---|
| Qwen3-VL | `Qwen/Qwen3-VL-4B-Instruct` | 4B | **Estimated:** 8.89 GB checkpoint; ~11–14 GB for controlled batch-1 short-prompt FP16 inference. Eager attention/long visual sequences may exceed 16 GB. | FP16 | Smallest practical Qwen3-VL Instruct model; public Apache-2.0 checkpoint, official Transformers support, explicit internals. | 8B only after a hook-preserving two-GPU sharding pilot; not default under 16 GB/device. |
| DINOv3 | `facebook/dinov3-vits16-pretrain-lvd1689m` | ViT-S, ~21M (**estimated**) | **Estimated:** well below 1 GB for weights plus batch-1 activations. | FP16; FP32 also practical at batch 1 | Small 12-layer 384-dimensional ViT/16, cheap frozen extraction, intermediate states exposed. | `facebook/dinov3-vits16plus-pretrain-lvd1689m` if later evidence justifies it. |

Qwen files total 8.88 GB. Selected DINO is **gated**: the Kaggle account must
accept Meta's terms and provide an authorized HF token. Sources: [Qwen files](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/main), [DINO files/access](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/tree/main).

## Qwen3-VL internals

The paths below are rooted at `Qwen3VLForConditionalGeneration` and are
**implementation verified** from current Transformers source.

| Component | Exact path | Tensor semantics |
|---|---|---|
| Multimodal base | `model.model` | `Qwen3VLModel` |
| Vision encoder | `model.model.visual` | `Qwen3VLVisionModel` |
| Patch embedder | `model.model.visual.patch_embed` | Conv3d projection; selected config output `(P,1024)`, P=sum(T×H×W). |
| Vision blocks | `model.model.visual.blocks[i]`, i=0..23 | 24 vision blocks. |
| Main merger | `model.model.visual.merger` | Spatial 2×2 merge then 1024→2560 language-width projection. |
| DeepStack mergers | `model.model.visual.deepstack_merger_list[j]` | Three merger paths from vision indices [5,11,17]. |
| Text model | `model.model.language_model` | `Qwen3VLTextModel`. |
| Input embedding | `model.model.language_model.embed_tokens` | `(B,S,2560)` before visual replacement. |
| Decoder blocks | `model.model.language_model.layers[i]`, i=0..35 | 36 blocks; post-attention/MLP residual `(B,S,2560)`. |
| Text attention | `model.model.language_model.layers[i].self_attn` | 32 query heads, 8 KV heads, 128 head dim; `q_proj,k_proj,v_proj,o_proj`. |
| Final state/logits | `model.model.language_model.norm`; `model.lm_head` | Final norm then vocabulary logits. |

The checkpoint config verifies text width 2560, 36 layers, 32 attention heads,
8 KV heads; vision width 1024, 24 layers, 16 heads, patch size 16, spatial merge
2. Sources: [config](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/config.json), [implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py).

### Visual feature path and insertion

**Implementation verified:** `.visual` returns unmerged vision output,
`.merger` output, and DeepStack features. Main merged visual features replace
image-placeholder embeddings in the language sequence. For normal `input_ids`,
`input_ids == model.config.image_token_id` is the image-position mask;
`mm_token_type_ids` labels text/image/video as 0/1/2.

DeepStack is material: features from vision blocks [5,11,17] are added at visual
positions after language decoder calls 0, 1, and 2. Therefore an early language
layer state is not merely processing initial merged visual features. Record
whether a capture is before or after this parent-level addition.

### Hidden states and attention

**Implementation verified:** source declares `Qwen3VLTextDecoderLayer` a
recordable hidden-state source and `Qwen3VLTextAttention` a recordable
attention source; conditional-generation output includes both fields.

**Requires Kaggle validation:** attention probabilities are backend dependent.
Eager attention materializes `(B,32,S,S)`; SDPA/Flash Attention may return
none. Run `attn_implementation="eager"`, one image, batch 1, short prompt,
and verify all 36 returned entries are non-None. SDPA support is not attention
map availability.

## Visual token geometry

**Implementation/configuration verified:** the processor smart-resizes with
near-preserved aspect ratio, pixel bounds 65,536 to 16,777,216, patch 16,
temporal patch 2, merge 2. Resize factor is 32; H' and W' are multiples of 32.

```text
RGB image
 -> smart resize H' × W', normalize (mean/std 0.5)
 -> image_grid_thw = (1,H'/16,W'/16), patch-major pixel_values
 -> visual.patch_embed: H'/16 × W'/16 patch grid
 -> visual.merger: 2×2 spatial merge
 -> image placeholders: H'/32 × W'/32 language visual grid
```

Still-image visual-token count is `(H'/32)*(W'/32)` or
`prod(image_grid_thw)/2²`. Image-mask positions are spatial feature tokens;
vision start/end are delimiters, not spatial features. There is no CLS token in
this visual-to-language sequence. Sources: [preprocessor](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/preprocessor_config.json), [processor implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py).

**Kaggle validated (Transformers 4.57.6; Qwen revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`; repository commit
`49cf89fcb9a29d1ca5145f864a22cc646fb63ebc`):** the visual-token ordering is
row-major over the post-2×2-merge grid:

```text
token_index = row * merged_grid_cols + col
```

Evidence is empirical at the patch-embedding boundary, before positional
encoding and vision-block contextualization. Deterministic, content-unique
32×32 merged cells were cyclically shifted one cell horizontally and vertically.
For both asymmetric grids, all 384 patch embeddings matched their expected
pre-shift content coordinates (384/384 for each shift):

| Patch grid | Merged grid | Horizontal match | Vertical match |
|---|---|---:|---:|
| 16×24 | 8×12 | 384/384 | 384/384 |
| 24×16 | 12×8 | 384/384 | 384/384 |

The observed merger input was `(384,1024)` and merger output `(96,2560)`, i.e.
four contiguous patch embeddings per merged token. The installed vision forward
records no subsequent spatial reordering, and the merger output equalled layer
0's image-placeholder positions exactly (maximum absolute difference 0.0).
This establishes input token indexing; it does not make an attention or causal
claim. Keep the conservative `max_pixels`; the configured maximum is not a safe
T4 budget.

## Minimum hook strategy

B=batch, S=full multimodal sequence, V=visual-token count, P=pre-merge patch
count, D=2560.

| Target | Module path | Shape | Hook | Risk |
|---|---|---|---|---|
| Vision representation | `model.model.visual.blocks[i]` | `(P,1024)` | forward | Flattened multi-image layout, not language geometry. |
| Merged visual feature | `model.model.visual.merger` | `(V,2560)` | forward | Preserve image boundaries from `image_grid_thw`. |
| Decoder residual | `model.model.language_model.layers[i]` | `(B,S,2560)` | forward | Outputs 0–2 are before their DeepStack addition. |
| Post-DeepStack boundary | `model.model.language_model.layers[i+1]` | `(B,S,2560)` | forward-pre | Next-block input is preceding post-addition state. |
| Attention probabilities | API output / `.self_attn` | eager `(B,32,S,S)` | API first | Quadratic memory, backend dependence. |
| Head contribution | `.layers[i].self_attn.o_proj` | input `(B,S,4096)` | forward-pre | Verify head layout in smoke test. |

## Causal intervention feasibility

### Activation patching

**Feasible, implementation verified:** a forward-pre hook can replace selected
positions in the input residual of `model.model.language_model.layers[k]`.
Safest first target: full `(B,S,2560)` input to `layers[3]`, copied from a
donor with identical template, image-token layout, length, dtype, and device.
It follows all three early DeepStack additions and precedes block 3. Verify only
intended token positions change. Patching a block output is less precise because
the parent loop may add DeepStack afterward.

### Head ablation

**Technically feasible, requires runtime validation:** attention concatenates
head outputs before `.self_attn.o_proj`. A forward-pre hook can clone its
`(B,S,4096)` input and zero `head_id*128:(head_id+1)*128` for one of 32 query
heads. This leaves weights and other head slices unchanged. It is preferable to
editing Q/K/V weights, but is an output-contribution ablation because `o_proj`
mixes heads. Validate exactly one zeroed slice on eager and SDPA.

## DINOv3 feasibility

Use Transformers rather than Meta's research repository, sharing processors and
output semantics with Qwen. **Implementation/configuration verified:** selected
`DINOv3ViTModel`: patch 16, width 384, 12 layers, 6 heads, 224 input, four
register tokens. At 224²: `1 CLS + 4 registers + 14×14 patches = 201` tokens.
`pooler_output` is normalized CLS `(B,384)`; `last_hidden_state` is
`(B,201,384)`.

- `model.embeddings.patch_embeddings` — patch projection.
- `model.embeddings` — CLS/register/patch sequence.
- `model.model.layer[i]`, i=0..11 — transformer layers.
- `model.model.layer[i].attention` — attention.
- `model.norm` — final norm.

Frozen extraction uses `.eval()` and `torch.inference_mode()`;
`output_hidden_states=True` exposes intermediate states. `DINOv3ViTBackbone`
can select/reshape feature maps. Sources: [docs](https://huggingface.co/docs/transformers/model_doc/dinov3), [implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py).

## Kaggle-only smoke test

Run only this batch-1 FP16 inspection before dataset work:

```bash
python - <<'PY'
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
c = "Qwen/Qwen3-VL-4B-Instruct"
m = Qwen3VLForConditionalGeneration.from_pretrained(
    c, torch_dtype=torch.float16, attn_implementation="eager"
).cuda().eval()
p = AutoProcessor.from_pretrained(c)
print(type(m), type(m.model.visual), type(m.model.language_model.layers[0]))
PY
```

Then, with one supplied image, inspect processor fields, token count/order, hook
shapes, non-None attention, and peak VRAM. Do not start CUB extraction, caches,
or interventions before it passes.

## Decision

Qwen3-VL 4B plus DINOv3 ViT-S/16 is suitable for later instrumentation on
2×T4 subject to this smoke test. Risks: Qwen sequence length, eager attention's
quadratic memory, and gated DINO access—not missing hookable internals.
