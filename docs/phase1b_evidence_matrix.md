# Evidence matrix

This is a cache-design specification, not an experimental result. “CPU” means
the question can be analyzed from evidence already cached; it does not mean the
underlying VLM computation can be reproduced on CPU. Attention observations are
not explanations, and no listed observational evidence establishes causality.

## Research question to evidence

| Research question | Raw tensors and metadata required | CPU-derived quantities | GPU later? | Permanent cache / recomputable? | Approximate added storage at Phase 1A reference shape |
|---|---|---|---|---|---|
| Where does visual information enter language processing? | Main merger; three DeepStack merger outputs; decoder outputs and next-layer inputs around 0–3; image positions/grid | Insertion equality and residual deltas at image/non-image positions | No for observation | **KEEP**; no, parent additions cannot be recovered from later residuals | 1.41 MiB DeepStack + chosen early language states |
| How does the vision tower evolve? | Patch embedding; vision blocks 0, 12, 23; patch/merged maps | Norms, spatial maps, later RSA | No | **KEEP** checkpoints; no | 3.00 MiB for patch plus 3 blocks |
| How does visual information evolve through decoder? | Selected language residuals; image sequence positions; prompt positions | Image/text norm and similarity trajectories | No | **KEEP** hybrid layers; no | 3.34 MiB for six current layers |
| Which decoder layers retain/use visual information? | Same residuals; selected attention | Descriptive image-token separation and attention summaries | GPU for causal use | **KEEP** residuals; no | Included above |
| Which attention heads interact with image tokens? | Attention by head; query/key classes; image indices | Per-head mass, entropy, spatial concentration | No for observation | **KEEP** selected pilot layers; no | 0.79 MiB/full layer |
| How does visual attention change with depth? | Selected-layer attention; IDs; grid and sequence positions | Depth trajectories and rankings | No | **KEEP** selected attention; no | 2.38 MiB/3 full layers |
| How do spatial regions map into language representations? | Pre-merge/merged states; coordinates; merger; language image-token residuals | Grid heatmaps and regional pooling | No | **KEEP** spatial tokens/maps; no | Included in representations |
| How do Qwen and DINO compare? | Original-image hash; both processor configs and geometry; Qwen checkpoints; DINO patch tokens | Common-grid pooling, CKA/RSA later | No after both extractions | **KEEP** native tokens; no | DINO: 0.44 MiB/3 full 201-token layers |
| Which representations correlate with semantic/category information? | Frozen checkpoint states; separate immutable manifest labels/splits | Later probe matrices and correlations | No | **KEEP** representations; no | Included above |
| Which observations can later be tested causally? | Candidate layer/head scores; exact input and provenance | Candidate selection only | **Yes** | **KEEP** selection evidence; cache cannot establish effects | Metadata only |
| What is required for activation patching? | Donor/recipient reproducible inputs; exact layout; target-layer *input* residual donor values | Compatibility checks and donor selection | **Yes** | **OPTIONAL** states only after design approval | 0.56 MiB/layer/sample at reference |
| What is required for head ablation? | Inputs, revision/backend, head IDs, live `o_proj` input layout, baseline outputs | Candidate selection/baselines | **Yes** | **DO NOT CACHE** all head outputs | 0.89 MiB/layer/sample before repeats |

## Candidate tensor decisions

| Candidate | Decision | Reason |
|---|---|---|
| `visual.patch_embed` | **KEEP** | Earliest spatial representation. |
| Vision block 0 | **KEEP** | Cheap early contextual checkpoint. |
| Additional early vision block (for example 5) | **OPTIONAL** | Needed only for a DeepStack-branch investigation. |
| Middle/final vision block | **KEEP** | Broad tower trajectory endpoints. |
| Main merger output | **KEEP** | Exact visual feature inserted into placeholders. |
| DeepStack merger outputs | **KEEP** observational production | Small and uniquely identify early language injections. |
| Decoder layers 0–3 | **KEEP** | Required around known early DeepStack boundaries. |
| Selected middle layers and final layer | **KEEP** | Hybrid trajectory. Suggest 6, 12, 18, 24, 30, 35. |
| Every vision/decoder layer | **OPTIONAL** diagnostic set only | Exact localization at high permanent cost. |
| All-layer attention | **DO NOT CACHE** by default | Quadratic storage and eager runtime. |
| Full selected-layer attention | **KEEP** first real-image pilot | Avoid irreversible reduction before inspection. |
| Per-head aggregate statistics only | **DO NOT CACHE** as sole evidence | Too lossy; use as derived sidecar. |
| DINO patch tokens, selected layers | **KEEP** when comparison begins | Preserves spatial alternatives for CPU alignment. |
| DINO CLS separately | **DO NOT CACHE** | Derivable from a cached full DINO layer state. |

## Attention representation choices

Phase 1A stored eager probabilities `(B, 32, S, S)`. With `B=1, S=114`, one
FP16 layer is 831,744 bytes (0.79 MiB).

| Form | Retains / permanently loses | Storage/layer | CPU possibilities | Causal relevance |
|---|---|---:|---|---|
| A. Full matrix | Every head/query/key relation; loses nothing returned by API | 0.79 MiB | Any positional/head summary | Cannot yield an ablation result; rerun still required |
| B. Image-key columns | All query types into image keys; loses non-image-key allocation | 0.67 MiB | Text/image-to-image trajectories | Cannot reconstruct full matrix |
| C. Per-head summaries | Mass/entropy; loses query position, key map, distribution | Bytes to KiB | Ranking/depth plots | Candidate selection only |
| D. Stratified blocks | Text→image and image→image; loses omitted blocks | ~0.57 MiB with 18 text + 96 image rows | Spatial language attention plus image self-attention | Candidate selection only |

**First real-image pilot recommendation:** full matrices for layers 0, 3, 18,
35 plus query/key class indices. At the reference sequence length this is 3.17
MiB/image. After inspection, choose a reduced production form empirically; the
leading option is stratified text→image and image→image blocks plus derived
per-head summaries. Representations cannot reconstruct attention.

## Layer sampling and DINO alignment

| Strategy | Representation cost/reference image | Use | Limitation |
|---|---:|---|---|
| Sparse current 0/12/23 vision; 0/1/2/3/18/35 language | 6.81 MiB | 5–10 image pilot | Can miss transition points |
| Every layer | 39.26 MiB | Small diagnostic set only | ~5.8× cost; not causal evidence |
| Hybrid: patch; vision 0/6/12/18/23; DeepStack; language 0–3/6/12/18/24/30/35 | ~10–11 MiB before attention | Production observation | Targeted rerun needed for exact transitions |

Keep each model’s native FP16 patch tokens, layer IDs, processor configuration,
resolved geometry, and token coordinates. Qwen has 16-pixel pre-merge patches
and 32-pixel merged tokens; selected DINO has 16-pixel patches, CLS plus four
register tokens, width 384, and normally 14×14 patches at 224 pixels. Exclude
DINO CLS/register tokens for spatial comparison unless intentionally analyzed.
Choose pooling/interpolation and any projection on CPU; extraction-time
projection would discard alternative alignment methods. Do not implement CKA.

## Causal boundary

Cache supports analysis and candidate selection only. Activation patching and
head ablation require a Qwen rerun with identical revision, processor, prompt,
layout, and source image bytes (or verified hash/source). Patching also needs
compatible donor/recipient layer inputs; ablation needs the live concatenated
head output before `o_proj`. Do not pre-cache every possible intervention
tensor; cache donor states only after a specific approved intervention design.
