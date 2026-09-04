# Cache contract

## Scope and versioning

The schema-1 `.pt` files are valid immutable synthetic-instrumentation
evidence. They are not retroactively rewritten. This document specifies schema
2 for future observational extraction. A reader dispatches by
`schema_version`, rejects unknown major versions, and never silently
reinterprets a tensor name or geometry.

Format: one `torch.save` archive per sample plus an immutable JSON manifest
listing path, SHA-256, schema version, and run provenance. Tensors are
contiguous CPU FP16 unless declared otherwise. Archives contain raw evidence;
statistics, rankings, and plots are separate derived artifacts keyed by raw
cache digest.

## Required schema-2 envelope

```text
{
  schema_version: 2,
  record_kind: "sample_evidence",
  identifiers: {experiment_id, run_id, sample_id, image_id, manifest_id},
  provenance: {
    repository_commit, extraction_timestamp_utc, code_package_version,
    environment: {python, torch, transformers, cuda, gpu_name},
    model: {identifier, revision, config_sha256},
    processor: {identifier, revision, config_sha256, preprocessing},
    source_image: {uri_or_manifest_key, sha256, original_width, original_height}
  },
  input: {
    prompt_template_id, prompt_template_sha256, rendered_prompt_sha256,
    generation_settings, batch_index,
    image_grid_thw, patch_grid_thw, merged_grid_thw,
    image_token_sequence_positions, text_token_sequence_positions,
    token_coordinates: {patch: [...], merged: [...]}
  },
  extraction: {
    dtype, attention_implementation, requested_tensors, layer_ids,
    seed, max_pixels, model_mode
  },
  attention: {
    layout, query_head_count, kv_head_count, head_dimension,
    query_sequence_length, key_sequence_length, query_key_position_classes
  },
  tensors: {canonical_tensor_name: CPU tensor},
  tensor_descriptors: {
    canonical_tensor_name: {
      shape, dtype, semantic_role, layer_id, token_space, coordinate_space,
      module_path, hook_direction, decoder_addition_boundary
    }
  }
}
```

Patch/merged grids and explicit coordinate maps remain present even when
derivable. This protects against a reader applying an undocumented ordering.
Names use model-neutral roles with a model prefix, for example
`qwen.visual.block.12`, `qwen.visual.merger`,
`qwen.language.layer.18.output`, and
`qwen.language.layer.18.attention.full`.

## Raw evidence and derived statistics

Raw evidence is immutable: tensors, descriptors, spatial/token mappings,
model/processor identity, source-image identity, and prompt/input layout. Do
not overwrite it with labels, normalization, projections, or summaries.

Derived statistics include norms, attention mass, head rankings, region pools,
similarities, figures, and later probe inputs. They must record source-cache
SHA-256s, code commit, configuration, and random seed. Recompute them from raw
evidence whenever practical.

## Required validation

1. Required envelope fields and non-empty identifiers.
2. Model/processor revision and package versions.
3. Image SHA-256 and original/resolved dimensions.
4. For a still image, image-position count equals `prod(merged_grid_thw)`.
5. Coordinate count and ordering agree with both spatial grids.
6. Every descriptor agrees with tensor shape, dtype, and CPU device.
7. Layer IDs/requested names agree with descriptors.
8. Attention layout/head counts agree with descriptor metadata.
9. Manifest/cache SHA-256 integrity.

## Minimum implementation delta

No production extraction code is implemented in this phase. Current schema-1
helpers remain compatibility reader/writer support for completed Phase 1A
caches. Before real-image extraction, add only a schema-2 dataclass/validator,
explicit provenance and processor metadata, archive SHA-256 helper, and schema
dispatch preserving schema-1 reading. Add focused tests for schema-2 required
metadata, descriptor mismatch, geometry count, and schema-1 compatibility. Do
not add a dataset pipeline, extraction framework, or experiment abstraction.
