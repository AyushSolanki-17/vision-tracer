# Progress

## Synthetic instrumentation

- 2026-09-05: Reviewed the completed feasibility and smoke-test evidence.
  Confirmed that the implementation will use the empirically validated Qwen3-VL
  module paths and row-major merged visual-token ordering, without modifying
  the smoke test.
- 2026-09-05: Added CPU-first synthetic-image, geometry, cache, and analysis
  modules plus the Kaggle extraction entry point. GPU execution remains pending
  Kaggle validation; no model, CUB dataset, or activation artifact has been
  downloaded locally.
- 2026-09-05: Created `.venv` with pyenv Python 3.13.11 and installed the
  editable project with CPU-only test dependencies. No model weights or CUB
  data were downloaded.
- 2026-09-05: CPU test suite passed (4 tests), as did source compilation and
  whitespace validation. Kaggle GPU instrumentation is ready but deliberately
  has not been run from the local machine.
- 2026-09-05: Kaggle validation exposed a PyTorch API compatibility error in
  peak-memory reset before model loading. Updated the extractor to select CUDA
  device 0 and use the portable no-argument reset call; rerun is pending.
- 2026-09-05: First Kaggle extraction wrote four compact caches and completed
  cache-only analysis, but model-release verification failed: 8.17 GB remained
  allocated because local variables retained Qwen submodules. Fixed the
  references and added an explicit release-verification result; rerun pending.
