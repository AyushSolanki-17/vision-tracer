# Tiny real-image pilot: pre-Kaggle audit and run plan

## Pilot implementation

The readiness blockers identified by the audit are resolved by a narrow,
batch-one pilot path. This is not a dataset pipeline or production extractor.

The pilot uses schema 2. Schema-1 synthetic caches remain immutable and are
read through the existing compatibility path.

## CUB pilot source and manifest preparation

The Phase 1C pilot uses CUB-200-2011 only as a small real-photograph source;
it does not implement the production CUB experiment. The source is the
official CaltechDATA release, [CUB-200-2011 version 1.0](https://data.caltech.edu/records/65de6-vp158)
(DOI `10.22002/D1.20098`). The publisher lists the `CUB_200_2011.tgz` archive
as 1.2 GB with MD5 `97eceeb196236b17998738112f37df78`, and restricts image use
to non-commercial research and educational purposes. Review those terms before
using it.

No images need to be prepared or downloaded on the local development machine.
In Kaggle, the preparation command either accesses an already-attached official
CUB archive/directory or, with Internet enabled, downloads the verified
official archive to ephemeral working storage. In the download mode it verifies
the publisher's MD5, extracts only `images.txt`, `image_class_labels.txt`,
`classes.txt`, and the selected images, then deletes the archive. It never
extracts or retains the full image tree.

The default deterministic selection is 16 images. Sort distinct numeric CUB
class IDs; choose class positions `floor(i * class_count / 16)` for `i=0..15`;
then select the lowest numeric image ID in each chosen class. There is no random
sampling and the recorded seed is `null`. This gives one image from each of 16
distributed categories without asserting a full-dataset balance.

`scripts/prepare_cub_pilot_manifest.py` writes both `pilot_images.jsonl` and
`pilot_source_provenance.json`. The latter records the source dataset/release
and location, selection rule, selected CUB image IDs, class IDs/names,
source-relative paths, byte hashes, and dimensions. The JSONL duplicates this
per-image provenance as additional fields while retaining the five fields read
by the existing extractor.

## Manifest format

Supply an explicit JSONL file; no image discovery occurs. Each source path is
relative to the manifest file and is byte-hashed and dimension-checked before
every model forward:

```json
{"image_id":"pilot-001","source":"images/pilot-001.jpg","sha256":"<64 lowercase hex chars>","original_width":640,"original_height":480}
```

The manifest must select 1–20 images. The driver fails on duplicate IDs,
invalid hashes, changed image bytes, or dimension mismatch.

## Cache contract

Schema-2 stores source identity/hash/dimensions; Qwen and DINO identifiers,
revisions and configuration hashes; processor configuration hashes; commit and
runtime environment; prompt hashes; resolved geometry; patch/merged coordinate
maps; token positions; capture configuration; raw CPU tensors; and rich tensor
descriptors. Its integrity manifest verifies each cache hash, source hash,
schema version, commit, and extraction-configuration hash. Schema-1 remains
readable but is not promoted or rewritten.

## Real-image compatibility

The single-image geometry helper is sound for variable still-image grids: it
uses actual `image_grid_thw`, validates merged-token count, and creates
row-major merged coordinates. The new CPU test covers 16×24, 24×16, and 20×12
patch grids. The current shape validator likewise derives patch count and
sequence length at runtime, so variable dimensions do not intrinsically break
batch-one extraction.

The pilot consumes manifest entries rather than synthetic specs. It intentionally rejects multi-image
`image_grid_thw`; this is correct for a batch-one, one-image-per-forward pilot.
Different geometries must remain separate forwards and separate cache records;
they must not be padded or combined into one cache record. The real-image
implementation must preserve this rule.

## Instrumentation audit

| Required pilot target | Pilot state | Capture |
|---|---|---|
| `visual.patch_embed`, blocks 0/6/12/18/23, merger | Captured | Validated Phase 0B module paths. |
| DeepStack branches 5/11/17 | Captured | `visual.deepstack_merger_list` outputs labelled with validated source-layer IDs. |
| Language 0/1/2/3/6/12/18/24/30/35 | Captured | Batch-one decoder outputs. |
| Attention 0/3/18/35 | Captured | Full eager attention matrices. |
| DINO patch tokens 0/6 and final patch/CLS | Captured | Native DINO geometry/token metadata retained. |

The extraction validates each Qwen capture shape against actual processor/model
dimensions and fails before writing a cache on any mismatch.

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

At the Phase 1A 256×384 reference shape, the implemented Qwen evidence is
about 11.94 MiB/image (patch embedding, five vision blocks, merger, three
DeepStack features, and ten language outputs). Four full attention maps add
3.17 MiB and three DINO states add 0.44 MiB: approximately 15.55 MiB/image
before metadata/archive overhead, or about 0.15–0.31 GiB for 10–20 images.
Actual cost varies with `P`, `V`, and especially `S²` attention.

## Required reproducibility record

Each future sample cache must record repository commit; Qwen/DINO identifiers,
revisions/config hashes; Transformers, PyTorch, CUDA and GPU versions;
processor configuration/hash; source-image manifest ID, SHA-256, original and
resolved dimensions; prompt-template/rendered prompt hashes; image grid,
patch/merged mappings, sequence positions; extraction configuration; random
seed where applicable; and tensor descriptors. The run manifest records cache
SHA-256, sample count, timing, and peak memory.

## One-session procedure

1. Start one Kaggle notebook with Internet enabled and one T4. Review the
   official CUB terms and, for DINO, accept its Hugging Face terms and add an
   `HF_TOKEN` Kaggle Secret. Do not enable or use a second GPU.
2. Install the repository and its Qwen dependencies without replacing Kaggle
   Torch. In a Python cell, load the secret without printing it:

   ```python
   import os
   from kaggle_secrets import UserSecretsClient

   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
   ```

   Then prepare the CUB manifest on CPU in the same session:

   ```bash
   pip install -q -e '.[qwen,dev]'
   python scripts/prepare_cub_pilot_manifest.py \
     --download-official --accept-cub-terms --count 16 \
     --output-dir /kaggle/working/cub_pilot
   python -m pytest -q tests/test_cub_pilot_cpu.py tests/test_real_image_pilot_cpu.py
   ```

   The download is the only source acquisition step; it retains only the 16
   selected photographs and metadata in `/kaggle/working/cub_pilot`. If Kaggle
   Internet is unavailable, manually attach the official archive as a Kaggle
   input (not a manually curated image directory) and replace
   `--download-official` with
   `--archive /kaggle/input/<official-cub-archive>/CUB_200_2011.tgz`.
3. Load Qwen once in FP16/eager on `cuda:0`; extract one image at a time with
   the approved hybrid capture list, immediately save each CPU cache, and
   report runtime/peak memory.
4. Release Qwen; require release verification; then load DINO once and cache
   native selected patch-token evidence for the same manifest entries.
5. Write immutable per-sample caches and the run manifest; reload every cache
   using CPU-only code and validate descriptors/geometry.
6. Emit one final JSON report. Hard-stop on any missing tensor, mapping/count
   mismatch, cache validation failure, unverified release, or unsafe memory.

## Exact capture configuration

- Qwen vision: patch embedding; blocks 0, 6, 12, 18, 23; main merger;
  DeepStack merger outputs associated with vision layers 5, 11, 17.
- Qwen language outputs: 0, 1, 2, 3, 6, 12, 18, 24, 30, 35.
- Qwen eager attention: full matrices at 0, 3, 18, 35.
- DINO: native patch tokens at layers 0 and 6; final native patch tokens and
  CLS token. DINO register-token counts/indices and patch coordinates are
  metadata, not cached values.

## Output and CPU validation

Each image produces `samples/<image_id>.pt` with schema-2 raw tensors and
provenance. `integrity_manifest.json` maps image/source hashes to cache hashes.
After DINO is released, the driver verifies that manifest, reloads every cache
without importing model classes, validates descriptors/geometry, computes only
norm and image-key attention summaries, and writes `run_report.json`.

## Kaggle command

Use one T4. The driver resolves the
requested DINO revision through Hugging Face to an immutable commit before any
DINO model files are loaded; that commit is used for both processor/model and
recorded in every cache. Supply a commit directly with `--dino-revision` when
one has already been selected. Do not restart the session between Qwen and DINO:

```bash
pip install -q -e '.[qwen]'
VISION_TRACE_GIT_COMMIT=$(git rev-parse HEAD) \
python scripts/run_real_image_pilot.py \
  --manifest /kaggle/working/cub_pilot/pilot_images.jsonl \
  --dino-revision main \
  --run-id real-image-pilot-001 \
  --output-dir results/real_image_pilot
```

The command fails immediately on a critical validation error and does not
continue to subsequent images. It must be preceded only by the documented
Kaggle dependency installation; it must not be run locally.
