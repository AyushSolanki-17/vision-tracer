# Synthetic instrumentation validation

This document specifies the narrow synthetic instrumentation slice following
the completed smoke test. It is not a dataset experiment and makes no
scientific claim about visual representations or attention.

## Scope and architecture

`scripts/run_synthetic_instrumentation.py` drives one Qwen3-VL-4B-Instruct
model on CUDA, FP16, eager attention, and batch size one. Reusable components
live under `src/vision_trace/`:

- `synthetic.py` creates four deterministic, in-memory RGB images: two
  asymmetric cell-texture images (256x384 and 384x256), a distinct-region
  image, and a uniform control.
- `synthetic_instrumentation.py` registers temporary forward hooks, moves each
  capture directly to CPU, validates runtime shapes, writes evidence, then
  explicitly releases model and processor references.
- `geometry.py` builds the exact still-image placeholder mapping from processor
  `image_grid_thw` and `input_ids == image_token_id`.
- `cache.py` is deliberately independent of Transformers/Qwen; it saves and
  loads CPU tensors using `torch.save`/`torch.load`.
- `analysis.py` computes descriptive tensor-norm and attention-mass summaries
  from an already loaded cache.

The model revision is pinned to the Phase 0B-validated Qwen revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. No checkpoint is loaded locally.

## Captures

Forward hooks retain the first tensor output of these Phase 0B-validated paths:

| Evidence | Captures |
|---|---|
| Vision | `visual.patch_embed`, `visual.blocks[0]`, `visual.blocks[12]`, `visual.blocks[23]`, `visual.merger` |
| Language | `language_model.layers[0]`, `[1]`, `[2]`, `[3]`, `[18]`, `[35]` |
| Attention | Eager API outputs for language layers 0, 18, and 35 only |

For an observed 256x384 synthetic image in the completed smoke test,
`image_grid_thw=(1,16,24)`, pre-merge captures are `(384,1024)`, merger output
is `(96,2560)`, decoder captures are `(1,114,2560)`, and selected attention is
`(1,32,114,114)`. The runtime extractor validates the actual dimensions rather
than assuming this example applies to every processor result.

Qwen's eager API necessarily materializes attention in each decoder layer
during its forward call. The cache copies only layers 0, 18, and 35, then drops
the model output; this is controlled retention, not a claim that the model API
selectively computes only those layers.

## Geometry and sequence mapping

The cache stores both input `image_grid_thw=(T,Hpatch,Wpatch)` and merged grid
`(T,Hpatch/2,Wpatch/2)`, image-placeholder sequence indices, and one
`(time,row,column)` coordinate per visual token. Phase 0B empirically verified
the mapping for still images:

```text
merged_token_index = row * merged_grid_columns + column
```

The implementation checks that placeholder count exactly equals
`T * merged_rows * merged_columns`. It intentionally supports one still image
per forward in this validation slice; multiple-image/video layouts remain out
of scope.

## Cache schema

Each synthetic image produces one ignored `.pt` file under
`results/synthetic_instrumentation/`:

```text
{
  schema_version: 1,
  metadata: {
    experiment_identifier, image_identifier, synthetic_image,
    model: {identifier, revision, transformers_version}, repository_commit,
    geometry: {image_grid_thw, merged_grid_thw,
               image_token_sequence_indices, token_coordinates_thw},
    attention: {implementation, selected_language_layers, query_heads,
                kv_heads, sequence_length, layout},
    extraction_configuration
  },
  representations: {name: CPU tensor},
  attentions: {name: CPU tensor},
  tensor_descriptors: {representations: {name: {shape, dtype}},
                       attentions: {name: {shape, dtype}}}
}
```

The run report records each exact cache byte count, runtime, peak allocated and
reserved VRAM, shapes, and CPU reload/analysis output. At the smoke-test
example shape, the selected evidence is expected to be only a few megabytes
per image in FP16; report the measured value after Kaggle execution rather
than treating that estimate as a result.

The initial Kaggle run wrote all four caches and completed CPU analysis, with
3.15 seconds for the first 256x384 extraction and 0.11--0.13 seconds for
subsequent warmed forwards. 256x384 caches were 9,643,509 bytes and 256x256
caches were 6,243,257 bytes. Peak allocated VRAM was 9,030,223,360 bytes and
peak reserved VRAM was 9,110,028,288 bytes. This first run is not a successful
completion of the release requirement: it reported 8,166,265,344 allocated
bytes after release. The cause was retained Python references to vision/text
submodules; the extractor now deletes those references and reports an explicit
`model_release_verified` status. Rerun before treating this validation as
complete.

## CPU reload and analysis

After extraction returns, the script reloads every cache with
`load_cache_cpu`, which imports neither Transformers nor a Qwen class. It
validates tensor descriptors and reports mean L2 norms plus mean attention mass
assigned to image-key sequence positions. These are instrumentation checks and
descriptive quantities, not explanations or causal evidence.

## Local CPU validation

Create the local environment without model/data downloads:

```bash
PYENV_VERSION=3.13.11 pyenv exec python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

## Kaggle command

Use Kaggle Python 3.11 with one T4 and retain its CUDA-compatible Torch. From
the repository root, install the project plus Qwen dependencies without
reinstalling Torch, then run:

```bash
pip install -q -e '.[qwen]'
python scripts/run_synthetic_instrumentation.py --output-dir results/synthetic_instrumentation
```

The script writes `run_report.json`, which is the authoritative source for
measured runtime, peak VRAM, cache sizes, and the CPU reload result.

## Limitations and next boundary

- GPU/Kaggle execution has not yet been performed by this local change.
- No CUB data, training, probing, CKA, attribution, patching, or ablation is
  included.
- This slice captures decoder outputs; early decoder captures precede the
  parent-level DeepStack addition documented by the completed smoke test.
- It does not establish biological, semantic, correlational, or causal claims.
