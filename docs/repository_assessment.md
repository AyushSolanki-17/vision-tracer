# Repository Assessment

## Current Structure

The tracked repository contains only `.gitignore` and `README.md`. `README.md`
contains the project title (`vision-tracer`) and no technical description. At
the time of assessment, `AGENTS.md` is present as an untracked working-tree
file. There are no source, test, notebook, script, configuration, CI, or
documentation directories in the initial commit.

The repository has one commit (`c3a2165`, "Initial commit") on `main`, and an
`origin` remote pointing to the GitHub repository.

## Existing Infrastructure

- Git repository with a single linear initial commit and `main` as the local
  branch.
- `.gitignore` is based on a broad Python template. It excludes virtual
  environments, Python build products, common test/coverage caches, notebook
  checkpoints, `.env`, and several editor/tool caches.
- No Python package metadata (`pyproject.toml`, `setup.cfg`, or
  `requirements*.txt`) is present.
- No automated test runner, formatter, linter, type checker, CI workflow, or
  runtime configuration is configured.

## Reusable Components

- Preserve the existing Git history and remote configuration.
- Preserve `.gitignore` as a useful baseline for Python development; extend it
  later only when concrete artifact and tooling choices are made.
- Preserve `AGENTS.md` as the repository operating and research-integrity
  policy once it is committed.

There is no existing data, model, experiment, activation-cache, or analysis
abstraction to reuse.

## Existing Dependencies

No project dependencies, lockfiles, Python-version declaration, or package
manager configuration are declared. Dependency and tooling choices therefore
remain open and should be made deliberately before implementation.

## Potential Problems

- The project identity is described as `vision-trace` in `AGENTS.md`, while
  the repository directory, GitHub remote, and README title use
  `vision-tracer`. This naming ambiguity should be resolved before publishing
  package metadata or import paths; no rename is warranted during this audit.
- The current ignore rules do not explicitly cover research artifacts such as
  model checkpoints, downloaded datasets, activation caches, or experiment
  outputs. The policy in `AGENTS.md` prohibits committing them, so concrete
  ignore rules will be needed when their locations are chosen.
- The README does not yet state the research scope, setup expectations, or the
  distinction between reusable code and external artifacts.
- Reproducibility metadata, CI, CPU-only validation tooling, and configuration
  conventions have not been selected.

## What Should NOT Be Rewritten

- Do not replace the repository history, remote, or baseline `.gitignore`.
- Do not introduce compatibility layers for absent legacy code: no existing
  implementation needs migration.
- Do not rename the repository, directory, or README title until the
  `vision-trace` versus `vision-tracer` naming decision is made.
- Do not infer model internals or introduce model/data/experiment APIs before
  the relevant Phase 0 evidence exists.

## Proposed vision-trace Structure

When implementation begins, use a conventional, dataset- and model-neutral
layout rather than placing reusable code in notebooks:

```text
src/vision_trace/       reusable package code
tests/                  small, high-value CPU-first tests
configs/                versioned experiment configuration
scripts/                narrow command-line entry points
docs/                   design, protocol, and assessment documentation
notebooks/              research narrative importing package code
```

Within `src/vision_trace`, introduce dataset, representation extraction,
analysis, and artifact/provenance modules only as their respective phases
require them. CUB-specific behavior should remain in the dataset layer.

## Migration Plan

1. Resolve the package/repository naming decision and document the project
   scope in the README.
2. Add package metadata and only the dependencies needed for the first
   CPU-first feasibility work.
3. Define artifact locations and extend `.gitignore` for externally stored
   datasets, checkpoints, caches, and outputs.
4. Implement Phase 0 incrementally, with synthetic or tiny CPU validation
   before any model download or GPU execution.

No migration of existing implementation is required because none exists.

## Open Questions

- Should the canonical public/package name be `vision-trace` or
  `vision-tracer`?
- Which Python/package manager and minimum supported Python version should be
  standardized?
- What configuration format and experiment-result directory convention should
  be adopted?
- Which artifact store and path configuration will be used for Kaggle outputs,
  datasets, and activation caches?
- Which CI checks are justified for the initial CPU-only codebase?
