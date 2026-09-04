# Tiny real-image pilot: pre-Kaggle audit and run plan

## Decision

**NO-GO.** Do not start a Kaggle session with the current extractor. This is a
CPU-only audit conclusion, not a negative result about Qwen feasibility.

The minimal unresolved implementation work is concrete:

1. `run_synthetic_instrumentation` accepts only `SyntheticImageSpec` and
   internally renders synthetic pixels. It cannot accept an approved real-image
   manifest, source bytes, or source-image provenance.
2. Its capture lists are not the agreed pilot lists: vision 6/18 and all
   DeepStack branches are absent; language 6/12/24/30 are absent; attention 3
   is absent.
3. No DINO extraction or DINO cache evidence exists in `src/vision_trace/`.
4. Schema-1 lacks immutable real-image provenance, processor configuration,
   resolved dimensions, patch coordinates, text positions, environment
   metadata, and integrity manifest required by the schema-2 contract.

These omissions would irreversibly discard pilot evidence. They must be
implemented and CPU-tested in the next explicitly assigned implementation task
before GPU time is spent.

## Cache-contract audit

| Area | Current schema-1 status | Pilot consequence |
|---|---|---|
| Schema version, experiment/image IDs, model ID/revision, Transformers version, repository commit | Present | Satisfies a useful baseline. |
| Merged grid, image sequence positions, row-major merged coordinates | Present and validated for one still image | Adequate for merged-token spatial analysis. |
| Tensor CPU copies, shape/dtype descriptors, CPU reload | Present and tested | No GPU tensor is intentionally serialized. |
| Attention layout, selected layers, query/KV head count, sequence length | Present | Sufficient for the Phase 1A selected attention cache. |
| Source real-image identity/hash and original/resolved dimensions | Missing | Cannot identify or reproduce an individual real input. |
| Processor ID/revision/config/hash and preprocessing details | Missing | Smart-resize and normalization cannot be audited later. |
| Prompt-template ID/hash, rendered prompt, text positions | Missing | Prompt/layout differences cannot be distinguished. |
| Patch-grid coordinates and explicit patch→spatial mapping | Missing | Pre-merge spatial analysis/alignment is incomplete. |
| PyTorch/CUDA/GPU versions, timestamp, cache SHA-256 manifest | Missing | Provenance and integrity are incomplete. |
| DeepStack tensors/boundary inputs and hybrid layer evidence | Missing | Cannot answer the intended early insertion question after the run. |
| DINO evidence | Missing | Qwen↔DINO comparison is impossible from the pilot cache. |

Schema-2 itself is intentionally deferred: implementing it now without the
real-image extraction interface would not make the pilot runnable. Schema-1
remains the immutable completed synthetic evidence format.

## Real-image compatibility

The single-image geometry helper is sound for variable still-image grids: it
uses actual `image_grid_thw`, validates merged-token count, and creates
row-major merged coordinates. The new CPU test covers 16×24, 24×16, and 20×12
patch grids. The current shape validator likewise derives patch count and
sequence length at runtime, so variable dimensions do not intrinsically break
batch-one extraction.

However, its API is synthetic-only. It also intentionally rejects multi-image
`image_grid_thw`; this is correct for a batch-one, one-image-per-forward pilot.
Different geometries must remain separate forwards and separate cache records;
they must not be padded or combined into one cache record. The real-image
implementation must preserve this rule.

## Instrumentation audit

| Required pilot target | Current state | Audit finding |
|---|---|---|
| `visual.patch_embed`, blocks 0/12/23, merger | Captured | Correct validated Phase 1A paths. |
| Vision blocks 6/18 | Not captured | Required hybrid checkpoint evidence missing. |
| DeepStack branches 5/11/17 | Not captured | Required insertion evidence missing. Capture projected branch outputs and identify their exact source paths before GPU use. |
| Language 0/1/2/3/18/35 | Captured | Correct current hooks. |
| Language 6/12/24/30 | Not captured | Required hybrid trajectory evidence missing. |
| Attention 0/18/35 | Captured | Correct for current selection. |
| Attention 3 | Not captured | Required first-pilot selection missing. |
| DINO patch tokens 0/6/11, native token metadata | No implementation | Required pilot evidence missing. |

The audit does not infer unverified new module paths. DeepStack capture must
first inspect the installed Kaggle Transformers implementation and confirm
tensor shapes before it is added.

## DINO requirements

The completed smoke test verifies the selected DINO model’s `(1,201,384)` FP16
output at 224 pixels: 196 patch tokens, one CLS token, and four register
tokens. The pilot must save the processor’s resolved preprocessing settings,
original and resolved dimensions, `patch_size=16`, token types, and row-major
14×14 patch coordinates. It must retain native patch tokens (not an extraction
projection), and run DINO sequentially after Qwen is released in the same
Kaggle session. No alignment or CKA belongs in this pilot.

## Memory and performance assessment

One T4 and batch size 1 remain the appropriate starting point. The recorded
synthetic Qwen run used 9.03 GB peak allocated and 9.11 GB peak reserved VRAM,
leaving roughly 6.9 GB on a 16 GB T4 at the tested shape. CPU hook copies occur
during each forward, and per-image output/input references are deleted. The
current release code also deletes parent/submodule references; the next pilot
must report `model_release_verified` before sequential DINO loading.

Eager attention remains acceptable only for the 10–20 image pilot, with one
image per forward and full attention retained at four selected layers. It still
materializes all eager attention layers transiently, so the pilot must stop on
unexpected sequence-length or memory growth. No exact runtime range is
justified: recorded Qwen inference was 3.15 seconds for the first synthetic
image and 0.11–0.13 seconds thereafter, while real geometry and I/O differ.
DINO’s smoke-test inference was 0.10 seconds/image after a 3.73 second load.

At the Phase 1A 256×384 reference shape, the proposed hybrid Qwen evidence is
about 13.6 MiB/image (five vision checkpoints, merger, three DeepStack
features, ten language outputs, and three early boundary inputs). Four full
attention maps add 3.17 MiB and three DINO states add 0.44 MiB: approximately
17.2 MiB/image before metadata/archive overhead, or about 0.17–0.34 GiB for
10–20 images. Actual cost varies with `P`, `V`, and especially `S²` attention.

## Required reproducibility record

Each future sample cache must record repository commit; Qwen/DINO identifiers,
revisions/config hashes; Transformers, PyTorch, CUDA and GPU versions;
processor configuration/hash; source-image manifest ID, SHA-256, original and
resolved dimensions; prompt-template/rendered prompt hashes; image grid,
patch/merged mappings, sequence positions; extraction configuration; random
seed where applicable; and tensor descriptors. The run manifest records cache
SHA-256, sample count, timing, and peak memory.

## One-session procedure after blockers are fixed

1. Start one Kaggle notebook with Internet, one T4, and an approved 10–20-image
   pilot manifest. Do not restart between samples or between Qwen and DINO.
2. Install project/Qwen dependencies without replacing Kaggle Torch; record
   environment before model loading.
3. Run local CPU contract validation against the manifest and a mocked record.
4. Load Qwen once in FP16/eager on `cuda:0`; extract one image at a time with
   the approved hybrid capture list, immediately save each CPU cache, and
   report runtime/peak memory.
5. Release Qwen; require release verification; then load DINO once and cache
   native selected patch-token evidence for the same manifest entries.
6. Write immutable per-sample caches and the run manifest; reload every cache
   using CPU-only code and validate descriptors/geometry.
7. Emit one final JSON report. Hard-stop on any missing tensor, mapping/count
   mismatch, cache validation failure, unverified release, or unsafe memory.

There is deliberately no executable Kaggle command yet: the required
real-image and DINO extraction entry point does not exist. Running the current
synthetic script would not satisfy this plan.
