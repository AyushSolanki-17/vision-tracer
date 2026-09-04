# Kaggle Runtime Plan

## Environment

Use Kaggle Python 3.11, its CUDA-compatible PyTorch, and the minimal versions
in [feasibility.md](feasibility.md), with 2×16 GB T4 GPUs when available. Use
FP16 for Qwen3-VL. Record GPU name/memory, Torch, CUDA, Transformers,
checkpoint revision, and Git commit for every run.

Create and activate a project-local `.venv` from the future versioned
`pyproject.toml`/lock file before installing project dependencies. Preserve
Kaggle's CUDA-compatible Torch wheel rather than replacing it blindly. This
Phase 0 task does not create the environment or TOML file.

Qwen is public. The selected DINOv3 checkpoint is gated: accept its terms
beforehand and pass an HF token only through Kaggle Secrets; never print or
commit it.

## CPU-first tasks

Complete without loading a large checkpoint or reserving a GPU:

- CUB metadata parsing, indexing, deterministic manifests, split validation,
  bounding-box transforms.
- Probe train/test separation, CKA/RSA, aggregation, result loading, plotting.
- Cache serialization round trips, config/provenance, checkpoint/resume, and
  focused unit tests.

These later-phase tasks are listed only to protect GPU time; Phase 0 implements
none of them.

## GPU tasks

- One-image Qwen/DINO load and shape smoke tests.
- Qwen inference and frozen DINO feature extraction.
- Selected-layer activation, later attention extraction, targeted patching and
  head-output ablation; student training only if later justified.

Start Qwen at batch 1. Place Qwen4B FP16 on one T4 for hook pilots and use the
second for independent DINO work, not automatic sharding. Sharding complicates
hooks and must be benchmarked before use.

## Storage

| Location | Contents | Rule |
|---|---|---|
| Git repository | Source, docs, configs, tests, small protocols/manifests | Never weights, CUB, caches, secrets, large outputs. |
| Kaggle input | Attached CUB/model datasets | Read-only; record revisions. |
| Kaggle working | Temporary downloads/work | Ephemeral, not artifact storage. |
| Kaggle output | Deliberately selected caches/results/figures | Save provenance; version intentionally; never commit. |

Download models only in Kaggle or use authorized Kaggle inputs. Cache GPU
results once; CPU analyses consume them rather than re-running Qwen.

## GPU budget model

30 GPU-hours is a cap, not a throughput estimate. Benchmark a representative
one-image forward before planning a run.

| Variable | Meaning / principal scaling |
|---|---|
| `N` | Images; ordinary frozen extraction scales linearly. |
| `T` | Measured seconds/image at fixed resolution, prompt, backend, capture set. |
| `L` | Recorded/intervened layers; cache grows with L; repeated interventions may too. |
| `H` | Examined heads; one-head-at-a-time ablations can scale with H. |
| `S` | Full sequence length; materialized attention is roughly L×H×S². |
| `R` | Donor/recipient/intervention repeats; patching scales with R. |

```text
baseline extraction      ≈ N × T
selected-layer capture   ≈ N × T_capture(L)
one-head ablation        ≈ N × H × T_ablate
selected heads only      ≈ N × |H_selected| × T_ablate
activation patching      ≈ N × R × T_patch
attention storage        ∝ N × L × H × S²
```

Dangerous budget users: full-layer/full-head eager attention, one-forward-per-
head ablations, donor/recipient patching over all images. Two T4s normally
provide two separate 16 GB address spaces (about 30–32 GB in aggregate), not a
pooled 30 GB model allocation. They do not double a single forward's speed or
change the 16 GB/device limit unless explicitly sharded/distributed execution
has been validated.

## Execution sequence

1. **Pilot:** 1–2 images; verify paths/shapes, image-token geometry, attention
   backend behavior, peak VRAM.
2. **Benchmark:** fixed small image-size/prompt set; measure T, peak VRAM,
   cache size, eager versus SDPA.
3. **Extrapolate:** choose N, selected L/H, with margin under 30 hours.
4. **Production:** one planned extraction pass; cache selected reusable evidence
   with complete provenance.

Abort/escalate rather than retry blindly if Qwen4B batch-1 controlled resolution
exceeds 16 GB, eager attention is unavailable, visual-token order is unclear,
or DINO access is absent. Do not silently change model, preprocessing, or scope.
