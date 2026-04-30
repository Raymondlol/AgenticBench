"""Shared dataset/splits loading helpers.

Handles both local-disk Arrow (Mac dev) and HuggingFace Hub (vast.ai) sources.

Lookup priority:
  1. If cfg.dataset.hub_repo is set, load from Hub (auto-cached).
  2. Else, load from cfg.dataset.local_path on disk.

For splits.json:
  1. If hub_repo set, hf_hub_download() it.
  2. Else, read from cfg.dataset.splits_path.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_pilot_dataset(cfg, split: str | None = None):
    """Return a HF Dataset (single split or DatasetDict) based on config."""
    from datasets import load_dataset, load_from_disk, DatasetDict

    hub_repo = cfg.get("dataset", {}).get("hub_repo")
    if hub_repo:
        # load_dataset returns DatasetDict if multiple splits exist on Hub.
        # We always uploaded a single split named "train" via push_to_hub.
        ds = load_dataset(hub_repo)
        if isinstance(ds, DatasetDict):
            # Use the first split (typically "train") as the full corpus
            ds = ds[list(ds.keys())[0]]
        return ds

    local_path = cfg["dataset"]["local_path"]
    return load_from_disk(str(Path(local_path)))


def load_pilot_splits(cfg) -> dict:
    """Return splits.json contents (dict with train/val/test keys)."""
    hub_repo = cfg.get("dataset", {}).get("hub_repo")
    if hub_repo:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=hub_repo,
            filename="splits.json",
            repo_type="dataset",
        )
        return json.loads(Path(path).read_text())

    splits_path = cfg["dataset"]["splits_path"]
    return json.loads(Path(splits_path).read_text())
