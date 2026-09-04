# Storage and GPU budget

## Measured-shape assumptions

Estimates are uncompressed FP16 tensor payloads, excluding small metadata and
archive overhead. Phase 1A 256×384 evidence had `P=384` pre-merge patches,
`V=96` merged tokens, `S=114` sequence positions, visual width 1024, language
width 2560, and 32 query heads. “CUB-scale” is only a 11,788-image sizing
reference, not an authorized extraction plan.

| Evidence group | Formula | MiB/image | 10 images | 100 images | 1,000 images | 11,788 images |
|---|---|---:|---:|---:|---:|---:|
| Sparse Qwen representations | `4×P×1024×2 + V×2560×2 + 6×S×2560×2` | 6.81 | 0.07 GiB | 0.66 GiB | 6.65 GiB | 78.38 GiB |
| Every-layer Qwen representations | `25×P×1024×2 + V×2560×2 + 36×S×2560×2` | 39.26 | 0.38 GiB | 3.83 GiB | 38.34 GiB | 451.92 GiB |
| Three full attention layers | `3×32×S×S×2` | 2.38 | 0.02 GiB | 0.23 GiB | 2.32 GiB | 27.39 GiB |
| All 36 full attention layers | `36×32×S×S×2` | 28.56 | 0.28 GiB | 2.79 GiB | 27.89 GiB | 328.72 GiB |
| DINO three 201-token layers | `3×201×384×2` | 0.44 | 0.004 GiB | 0.04 GiB | 0.43 GiB | 5.08 GiB |

Three projected DeepStack features add `3×V×2560×2 = 1.41 MiB/image`
(16.19 GiB at the sizing reference). The observed archives were 9,643,509
bytes at 256×384 and 6,243,257 bytes at 256×256, consistent with 6.81 MiB of
representations plus 2.38 MiB of selected attention and archive overhead.
Costs grow with image geometry; attention also grows quadratically with `S`.

## Staged GPU strategy

| Stage | Scope | Evidence / storage | Cost and risk | Success / hard stop |
|---|---|---|---|---|
| 1. Tiny real-image pilot | 5–10 approved images, batch 1, one T4 | Hybrid representations + full attention at 0/3/18/35; ~13–15 MiB/image | First synthetic cold forward 3.15 s; warmed 0.11–0.13 s, but real sequences can differ | Exact mapping, cache validation, safe VRAM. Stop on mismatch/missing attention/unsafe memory. |
| 2. Pilot expansion | 50–100 approved images | Same evidence; ~0.7–1.5 GiB by geometry | Geometry/sequence tails can invalidate synthetic estimate | Inspect extrapolation. Stop before broad extraction if tails exceed budget. |
| 3. Production observation | Size selected only after Stage 2 | Hybrid Qwen + selected DINO; reduce attention only if Stage 1 justifies loss | Linear count cost, sequence-sensitive eager memory | One immutable cache pass. Stop on changed processor/model/config. |
| 4. Targeted causal work | Small candidate subset | Inputs, targeted donor/baseline evidence, intervention results | Repeated Qwen forwards; no per-intervention timing measured | Pre-registered target and precise hook validation. Stop if unintended positions change. |

No exact GPU-hour projection follows from four synthetic images. Model loading
took 43.56 seconds in the recorded run; measure each approved stage rather
than extrapolating blindly.
