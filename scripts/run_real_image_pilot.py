#!/usr/bin/env python3
"""One-session Kaggle driver for the small real-image evidence pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vision_trace.analysis import cached_evidence_analysis
from vision_trace.cache import load_cache_cpu, validate_integrity_manifest
from vision_trace.real_image_pilot import run_real_image_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Explicit JSONL image manifest.")
    parser.add_argument("--dino-revision", default="main", help="DINO Hugging Face revision; the resolved commit is recorded in every cache.")
    parser.add_argument("--run-id", required=True, help="Stable identifier for this one-session pilot.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/real_image_pilot"))
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    report = run_real_image_pilot(
        manifest_path=args.manifest, output_directory=args.output_dir, run_id=args.run_id,
        dino_revision=args.dino_revision, repository_root=root, max_images=args.max_images,
    )
    integrity = validate_integrity_manifest(report["integrity_manifest"])
    cpu_analysis = {}
    for entry in integrity["records"]:
        cache = load_cache_cpu(args.output_dir / entry["cache_path"])
        cpu_analysis[entry["image_id"]] = cached_evidence_analysis(cache)
    report["cpu_reload_and_analysis"] = cpu_analysis
    report_path = args.output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
