# [AGENTS.md](http://AGENTS.md)

## Project

This repository is **vision-trace**.

> Tracing visual information through multimodal models.

The project investigates how fine-grained visual information is represented and causally used inside vision-language models, with comparisons against dedicated vision foundation models such as DINOv3.

The initial experimental dataset is CUB-200-2011, but the repository must NOT be architecturally tied to birds or CUB.

CUB is an experimental substrate, not the identity of the project.

---

# 1. Core Research Goal

Investigate:

1. Where visual information becomes linearly decodable inside a VLM.
2. How representations evolve across layers.
3. How VLM representations compare with DINOv3 representations.
4. Which attention components correlate with visual-region processing.
5. Which components are causally important under intervention.
6. Whether selected task-relevant representations can be distilled into a compact model.

Do not assume the outcome.

The repository must distinguish:

- observation
- correlation
- causal evidence
- hypothesis
- limitation

Never make a stronger scientific claim than the experiment supports.

---

# 2. Agent Operating Principles

## 2.1 Minimal scope

Implement ONLY the task explicitly assigned to you.

Do not opportunistically implement future phases.

Do not create abstractions for hypothetical future requirements.

Do not refactor unrelated code unless the assigned task genuinely requires it.

If you discover a useful future improvement, record it in a short TODO or handoff note rather than implementing it.

---

## 2.2 Preserve existing work

Before modifying an existing repository:

1. Inspect the current structure.
2. Identify reusable code.
3. Understand current dependencies.
4. Reuse working infrastructure where practical.
5. Avoid replacing existing systems without evidence that replacement is necessary.

Never perform a wholesale rewrite simply because a different structure looks cleaner.

---

## 2.3 No speculative implementation

Do not guess model internals.

For model-specific behavior:

1. Inspect the actual installed implementation/version.
2. Verify module names and tensor shapes.
3. Verify behavior with a minimal example.
4. Document the finding.
5. Only then build abstractions around it.

This is especially important for Qwen3-VL visual tokens, hidden states, attention, and multimodal integration.

---

# 3. Model/Agent Routing Policy

The project may be worked on by multiple coding/reasoning agents.

The conceptual routing policy is:

### Luna — high-reasoning / research decisions

Use for:

- research architecture
- experiment design
- interpreting unexpected results
- difficult conceptual debugging
- deciding whether an experiment is scientifically valid
- resolving ambiguous model internals
- reviewing major architectural changes
- final research-quality review

Do NOT use Luna for:

- trivial file edits
- formatting
- simple config changes
- boilerplate
- repetitive test creation
- renaming variables
- straightforward documentation

---

### Sol — primary implementation

Use for:

- PyTorch implementation
- model integration
- activation hooks
- data pipelines
- experiment implementation
- performance optimization
- non-trivial debugging
- refactoring within an assigned module

Sol should implement the requested scope and stop.

Do not expand into unrelated research experiments.

---

### Terra — low-complexity implementation

Use for:

- small utilities
- straightforward tests
- configuration changes
- CLI changes
- simple plotting helpers
- documentation edits
- repetitive cleanup
- mechanical refactors

Terra must NOT:

- redesign architecture
- choose research methodology
- change model architecture
- alter causal intervention methodology
- expand task scope
- make scientific claims
- replace major abstractions

If a task becomes conceptually ambiguous, stop and request escalation rather than improvising.

---

### Other/open-source agents

Suitable for:

- boilerplate
- formatting
- mechanical transformations
- simple documentation
- repetitive non-critical tasks

Do not delegate research-critical code or model-internal reasoning to an agent that has not been explicitly assigned that responsibility.

---

# 4. Escalation Policy

Escalate instead of improvising when:

- the model architecture differs from assumptions
- tensor semantics are unclear
- a hook captures the wrong representation
- visual-token mapping is uncertain
- an intervention changes unintended tensors
- train/test leakage is possible
- a scientific claim depends on an unresolved methodological choice
- a result contradicts an established project assumption

Before escalation, provide:

1. What was expected.
2. What actually happened.
3. Relevant tensor/module information.
4. What was already attempted.
5. The smallest unresolved decision.

Do not repeatedly retry the same failed approach.

---

# 5. Compute Policy

## Local machine

The local development machine MUST NOT be used for downloading or storing large model checkpoints or the CUB dataset unless explicitly requested.

Do not download:

- Qwen3-VL weights
- DINOv3 weights
- CUB-200-2011
- large activation caches
- large experiment artifacts

Local work should focus on:

- source code
- configuration
- static inspection
- lightweight tests
- synthetic tensors
- CPU-compatible analysis
- documentation

---

## Cloud/GPU environment

Use Kaggle for expensive computation.

The available GPU budget is approximately:

- 30 hours total
- 2×T4 when available

Treat this as a finite research budget.

Never launch expensive GPU experiments before a CPU/debug path has succeeded.

---

# 6. CPU-First Policy

Before using GPU compute, test as much as possible using:

- synthetic tensors
- tiny datasets
- 1–5 examples
- reduced dimensions
- mocked model outputs where scientifically safe
- CPU implementations of analysis algorithms

CPU-first validation should cover:

- dataset indexing
- manifest generation
- train/test split integrity
- bounding-box transformations
- activation-cache serialization
- probe training
- CKA/RSA implementation
- statistical aggregation
- plotting
- configuration
- checkpoint/resume logic
- result loading

Do not waste GPU time debugging Python plumbing.

---

# 7. GPU Budgeting

GPU execution should follow this hierarchy:

### Tier 0 — smoke test

Minimal:

- 1–2 images
- one forward pass
- selected hooks
- tensor-shape validation

### Tier 1 — pilot

Small:

- few species
- few images
- limited layers

Purpose:

Validate that the experiment behaves correctly.

### Tier 2 — production extraction

Only after Tier 1 succeeds.

Run the expensive model once and cache reusable representations.

### Tier 3 — targeted experiments

Run only experiments justified by previous measurements.

Never blindly run every layer × head × image combination.

---

# 8. Cache Everything Expensive

The expensive principle is:

> GPU computes evidence once; CPU analyzes the evidence many times.

Do not repeatedly run Qwen3-VL because another experiment needs the same representation.

Cache:

- model metadata
- evaluation manifest
- hidden representations
- attention data when required
- predictions
- candidate scores
- experiment configuration
- random seed
- git commit
- package versions

Downstream analysis must consume cached artifacts whenever possible.

---

# 9. Storage Policy

Never commit:

- model weights
- datasets
- secrets
- API tokens
- Kaggle credentials
- massive activation dumps
- temporary files
- notebook checkpoints

Large artifacts belong in Kaggle persistent/output storage or another explicitly configured artifact location.

Git should contain the machinery that reproduces the experiment, not the entire experiment's raw storage footprint.

---

# 10. Testing Policy

We value correctness over test-count.

Write tests only where failure could invalidate the research.

High-value tests include:

- dataset split integrity
- bbox conversion
- manifest determinism
- model wrapper loading
- activation hook registration
- activation tensor shape
- visual-token identification
- intervention targeting
- probe train/test separation
- cache round-trip integrity

Do NOT create tests for:

- trivial getters
- one-line wrappers
- logging
- cosmetic plotting details
- every obvious helper

Target a small, high-value test suite rather than maximal coverage.

---

# 11. Research Integrity

Never:

- fabricate results
- fabricate tensor shapes
- fabricate model internals
- fabricate performance
- cherry-pick examples without documenting selection
- silently alter preprocessing
- silently change evaluation subsets
- report only successful experiments
- call attention maps explanations
- infer causality from attention correlation
- claim representation equivalence from CKA
- use generated judgments as the primary evaluation metric

If an experiment fails, document the failure.

A failed experiment is preferable to an invented result.

---

# 12. Experiment Phases

The default order is:

## Phase 0

Feasibility + smoke test

## Phase 1

CUB pipeline + VLM baseline

## Phase 2

Activation extraction

## Phase 3

Layerwise representation probing

## Phase 4

DINOv3 representation comparison

## Phase 5

Attention analysis

## Phase 6

Head ablation

## Phase 7

Activation patching

## Phase 8

Representation distillation

## Phase 9

Final figures + Kaggle notebook + report

Agents MUST NOT jump ahead unless explicitly instructed.

---

# 13. Stop Conditions

Every assigned task should have a clear completion condition.

When the completion condition is met:

STOP.

Do not continue improving unrelated areas.

For example:

> "Implement activation cache and add shape validation."

Completion means:

- cache implemented
- required tests pass
- shape validation works
- documentation updated

It does NOT mean:

- implementing probes
- implementing CKA
- refactoring the entire model layer
- improving unrelated notebooks

---

# 14. Change Discipline

For every meaningful change:

1. Explain what changed.
2. Explain why.
3. List files modified.
4. List tests executed.
5. State any unresolved issues.

Keep changes reviewable.

Prefer small coherent commits over giant generated changes.

---

# 15. Scientific Reproducibility

Every experiment must record:

- model identifier
- dataset manifest
- configuration
- random seed
- code version/git commit
- relevant package versions
- sample count
- species count
- preprocessing configuration

Results must be traceable back to their experimental configuration.

---

# 16. Notebook Policy

Notebooks are for:

- research narrative
- exploratory analysis
- final visualization
- communicating findings

Notebooks are NOT the primary implementation location.

Reusable logic belongs under:

`src/vision_trace/`

The Kaggle notebook should import the repository code rather than contain duplicated implementations.

---

# 17. Repository Identity

The project must remain model- and dataset-general.

Avoid names such as:

- bird_classifier
- cub_model
- birdscope

Prefer terminology such as:

- vision
- representation
- multimodal
- trace
- latent
- intervention

CUB and bird-specific logic should exist in the dataset layer, not throughout the architecture.

---

# 18. Final Quality Bar

The repository should feel like:

> A compact, reproducible research codebase for studying visual information flow inside multimodal foundation models.

It should NOT feel like:

> A Kaggle notebook converted into a GitHub repository.

It should NOT feel like:

> A generic AI demo.

It should NOT contain unnecessary engineering complexity.

The goal is high scientific signal per line of code.