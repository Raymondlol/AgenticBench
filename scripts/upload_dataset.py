#!/usr/bin/env python3
"""
Push the local Arrow dataset + splits.json to a private HuggingFace dataset repo.

Run this AFTER build_dataset.py + split.py have produced the local artifacts.

Usage:
  python scripts/upload_dataset.py --repo-id yourname/asmr-pilot-50h
                                    [--config train/config.yaml]
                                    [--token-from-env HF_TOKEN]

Prerequisite:
  pip install huggingface_hub
  huggingface-cli login   # or pass --token-from-env HF_TOKEN
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True,
                        help="HF Hub repo id, e.g. yourname/asmr-pilot-50h")
    parser.add_argument("--config", default="train/config.yaml")
    parser.add_argument("--token-from-env", default=None,
                        help="Read HF token from this env var instead of ~/.huggingface")
    parser.add_argument("--public", action="store_true",
                        help="Make repo public (default: private)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    ds_path = Path(cfg["dataset"]["local_path"])
    splits_path = Path(cfg["dataset"]["splits_path"])

    if not ds_path.exists():
        print(f"ERROR: dataset not found at {ds_path}. Run build_dataset.py first.",
              file=sys.stderr)
        sys.exit(1)
    if not splits_path.exists():
        print(f"ERROR: splits not found at {splits_path}. Run split.py first.",
              file=sys.stderr)
        sys.exit(1)

    token = None
    if args.token_from_env:
        token = os.environ.get(args.token_from_env)
        if not token:
            print(f"ERROR: env var {args.token_from_env} not set", file=sys.stderr)
            sys.exit(1)

    from huggingface_hub import HfApi, create_repo
    from datasets import load_from_disk

    api = HfApi(token=token)

    # Create repo (private by default)
    print(f"Creating repo: {args.repo_id} (private={not args.public})", flush=True)
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
        token=token,
    )

    # Load dataset and push (uses datasets.push_to_hub)
    print("Loading local dataset...", flush=True)
    ds = load_from_disk(str(ds_path))
    print(f"Dataset: {ds}", flush=True)
    print(f"Pushing to hub (this can take 10-20 min for ~15GB)...", flush=True)
    ds.push_to_hub(args.repo_id, private=not args.public, token=token)

    # Also upload splits.json
    print("Uploading splits.json...", flush=True)
    api.upload_file(
        path_or_fileobj=str(splits_path),
        path_in_repo="splits.json",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
    )

    print(f"\nDone. Update train/config.yaml with:", flush=True)
    print(f"  dataset:\n    hub_repo: {args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
