# Kaggle smoke test

This is the Phase 0B, Kaggle-only instrumentation check. It performs one
batch-1 FP16 Qwen forward with eager attention, a second low-cost asymmetric
geometry forward, and one DINOv3 forward. It does not download CUB, create an
activation cache, or run a research experiment.

## Kaggle assumptions

- Start a fresh Kaggle notebook with Internet enabled and one CUDA GPU (a T4
  is sufficient for the planned batch-1 check).
- Keep Kaggle's installed CUDA-compatible PyTorch. Do not reinstall `torch`.
- The notebook working directory contains this repository, or the script is
  uploaded/copied into the notebook working directory.
- Qwen downloads from its public Hugging Face repository. DINOv3 is gated:
  accept the model terms for
  `facebook/dinov3-vits16-pretrain-lvd1689m` while signed into Hugging Face.

Use a Kaggle Secret named `HF_TOKEN` for the authorized Hugging Face token.
Never print it, put it in a notebook cell, commit it, or pass it on a command
line. In a notebook setup cell, expose the secret only to the process:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Install only the libraries not already present, retaining Kaggle's torch:

```bash
pip install -q "transformers>=4.57,<5" "accelerate>=1.0" "huggingface_hub>=0.26" "safetensors>=0.4" Pillow
```

## Run

From the repository root in the Kaggle terminal or a notebook shell cell:

```bash
python scripts/kaggle_smoke_test.py
```

To rerun only the final Qwen visual-token-ordering check (and retain the DINO
summary already obtained), run:

```bash
python scripts/kaggle_smoke_test.py --qwen-only
```

`VISION_TRACE_GIT_COMMIT` is only needed when the Kaggle working directory was
uploaded without its `.git` metadata. Before running an uploaded copy, set it
to the exact commit that was uploaded, for example:

```bash
export VISION_TRACE_GIT_COMMIT="<exact uploaded commit SHA>"
python scripts/kaggle_smoke_test.py --qwen-only
```

A normal clone records its commit automatically. Do not replace this with a
guessed revision.

The script makes two deterministic asymmetric synthetic RGB images locally;
no image or dataset is downloaded. It writes only these small, run-specific
files:

```text
results/smoke_test/qwen_summary.json
results/smoke_test/dino_summary.json
```

They are ignored by Git. Download them from Kaggle output or copy their
contents into the research plan; do not commit them as scientific results.

## What success looks like

The structured console report and JSON summaries should show:

- CUDA environment, model identifiers, versions, GPU name/count, load/inference
  elapsed time, and allocated/reserved/peak allocated/peak reserved memory.
- Qwen processor keys, `image_grid_thw`, image-placeholder count, full sequence
  length, merger length, and agreement of the three visual-token counts.
- CPU-detached captures for the requested vision block, merger, language layers
  0/3/18/35, and final language hidden state.
- An eager attention result with non-`None` entries, layer count, and one
  `(B, H, S, S)` shape.
- A visual-token-ordering report for both `16×24 → 8×12` and `24×16 → 12×8`
  grids. It records installed processor/vision/merger source excerpts,
  patch-embed and merger-input hook shapes, and the direct comparison between
  merger output and language layer-0 image positions.
- Empirical cyclic-shift evidence: every merged `32×32` cell has a unique
  deterministic texture. One-cell horizontal and vertical cyclic shifts are
  matched against CPU-detached patch-embedding features, before positional
  features and vision-block contextualization can confound content matching.
  Treat the ordering as established only when both shift checks report 100%
  expected matches and `empirically_validated: true`; otherwise the report is
  explicitly `inconclusive` and Phase 0B must not be closed.
- DeepStack source/boundary evidence: deltas between a decoder layer output and
  the next decoder layer input at visual positions. This establishes whether
  the parent loop adds DeepStack features in between; layer 3 input should be
  checked as the post-third-addition boundary.
- DINO load, `last_hidden_state`, `pooler_output`, hidden-state count, token
  count, hidden width, register-token count, and patch geometry.

Copy the exact package versions, GPU/memory/runtime values, Qwen sequence and
visual-token counts, attention availability/shape, hook shapes, DeepStack
boundary findings, DINO token geometry, and any failures into the research
plan. If runtime behavior contradicts `docs/feasibility.md`, update that
document with the observed value and retain the JSON as run evidence.

## Failure handling

If Qwen fails to load or run, the script writes `qwen_summary.json`, prints the
exception, and stops before DINO. Do not substitute another Qwen model or claim
any skipped check passed.

If DINO fails, `dino_summary.json` records the exception and highlights likely
gated access. First verify that the terms were accepted for the exact model and
that the Kaggle `HF_TOKEN` secret is authorized. Do not silently substitute a
different DINO checkpoint.

An eager-attention OOM is a negative feasibility result, not a reason to switch
to SDPA inside this test: record its peak memory and sequence length, then
escalate the smallest decision (lower pixel budget versus a different GPU
configuration). Likewise, missing attention tensors or a visual-token count
mismatch requires inspection before later phases begin.

If the ordering check is conclusive, its exact rule is reported as
`token_index = row * merged_grid_cols + col`, where `(row, col)` indexes the
post-2×2-merge grid. This is a runtime conclusion, not a pre-run assumption.
