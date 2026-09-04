#!/usr/bin/env python3
"""Kaggle-only synthetic Qwen instrumentation validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vision_trace.analysis import cached_evidence_analysis
from vision_trace.cache import load_cache_cpu
from vision_trace.synthetic_instrumentation import run_synthetic_instrumentation
from vision_trace.synthetic import SYNTHETIC_IMAGES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic_instrumentation"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    report = run_synthetic_instrumentation(
        output_directory=args.output_dir, image_specifications=SYNTHETIC_IMAGES, repository_root=root
    )
    # No Transformers/Qwen import occurs in these cache/analysis modules.
    reload_results = {}
    for cache_path in report["cache_paths"]:
        record = load_cache_cpu(cache_path)
        reload_results[Path(cache_path).stem] = cached_evidence_analysis(record)
    report["cpu_reload_and_analysis"] = reload_results
    report_path = args.output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["model_release_verified"]:
        print("FAILURE: Qwen CUDA allocation remained above the release threshold.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
