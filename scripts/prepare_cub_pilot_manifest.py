#!/usr/bin/env python3
"""Prepare a tiny deterministic CUB pilot manifest for Kaggle extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vision_trace.cub_pilot import (
    CUB_OFFICIAL_RECORD_URL,
    download_official_cub_archive,
    extract_cub_pilot_from_archive,
    generate_cub_pilot_manifest,
    verify_official_cub_archive,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cub-root", type=Path, help="Existing CUB_200_2011 directory, usually a Kaggle input.")
    source.add_argument("--archive", type=Path, help="Existing official CUB_200_2011.tgz archive, usually a Kaggle input.")
    source.add_argument("--download-official", action="store_true", help="Download the verified official archive in the Kaggle session.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--accept-cub-terms", action="store_true", help="Required acknowledgement of the official CUB non-commercial research/education restriction.")
    args = parser.parse_args()
    if args.download_official and not args.accept_cub_terms:
        parser.error("--download-official requires --accept-cub-terms after reviewing the official dataset terms")
    if args.cub_root:
        root = args.cub_root
        archive_receipt = None
    else:
        archive = args.archive
        if args.download_official:
            archive = download_official_cub_archive(args.output_dir / "CUB_200_2011.tgz")
        else:
            assert archive is not None
            archive = verify_official_cub_archive(archive)
        assert archive is not None
        archive_receipt = str(archive)
        try:
            root = extract_cub_pilot_from_archive(archive, args.output_dir / "selected_source", count=args.count)
        finally:
            if args.download_official:
                Path(archive_receipt).unlink(missing_ok=True)
    manifest, provenance = generate_cub_pilot_manifest(root, args.output_dir, count=args.count)
    print(json.dumps({"manifest": str(manifest), "provenance": str(provenance), "source_location": CUB_OFFICIAL_RECORD_URL, "count": args.count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
