#!/usr/bin/env python3
"""
Build train/val/test splits BY work_id.

Critical: same work_id must NOT appear in multiple splits, otherwise
acoustic/voice features leak between train and eval, inflating metrics.

Usage:
  python train/split.py [--config train/config.yaml] [--ratios 0.8,0.1,0.1]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/config.yaml")
    parser.add_argument("--ratios", default="0.8,0.1,0.1",
                        help="train,val,test ratios")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ds_path = Path(cfg["dataset"]["local_path"])
    splits_path = Path(cfg["dataset"]["splits_path"])

    ratios = [float(r) for r in args.ratios.split(",")]
    assert len(ratios) == 3, "Need 3 ratios: train,val,test"
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1, got {sum(ratios)}"

    # Load dataset
    from datasets import load_from_disk
    ds = load_from_disk(str(ds_path))
    print(f"Loaded dataset: {len(ds)} examples", flush=True)

    # Group examples by work_id
    work_segs = defaultdict(list)
    work_durs = defaultdict(float)
    for i, ex in enumerate(ds):
        wid = ex["work_id"]
        work_segs[wid].append(i)
        work_durs[wid] += float(ex["dur_s"])

    work_ids = sorted(work_segs.keys())
    print(f"Unique works: {len(work_ids)}", flush=True)

    # Deterministic shuffle of work_ids
    rng = random.Random(args.seed)
    rng.shuffle(work_ids)

    # Greedy assignment: aim for exact ratios by work count, but log dur balance
    n = len(work_ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val

    train_works = work_ids[:n_train]
    val_works = work_ids[n_train:n_train + n_val]
    test_works = work_ids[n_train + n_val:]

    splits = {
        "train": {"work_ids": train_works,
                   "n_works": len(train_works),
                   "n_examples": sum(len(work_segs[w]) for w in train_works),
                   "total_dur_s": sum(work_durs[w] for w in train_works)},
        "val":   {"work_ids": val_works,
                   "n_works": len(val_works),
                   "n_examples": sum(len(work_segs[w]) for w in val_works),
                   "total_dur_s": sum(work_durs[w] for w in val_works)},
        "test":  {"work_ids": test_works,
                   "n_works": len(test_works),
                   "n_examples": sum(len(work_segs[w]) for w in test_works),
                   "total_dur_s": sum(work_durs[w] for w in test_works)},
        "config": {
            "ratios": ratios,
            "seed": args.seed,
            "split_strategy": "by_work_id",
        },
    }

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits, ensure_ascii=False, indent=2))

    # Print summary
    print(f"\n{'split':<8s}  {'n_works':>8s}  {'n_examples':>12s}  {'dur (h)':>10s}")
    print("-" * 48)
    for name in ["train", "val", "test"]:
        s = splits[name]
        print(f"{name:<8s}  {s['n_works']:>8d}  {s['n_examples']:>12d}  "
              f"{s['total_dur_s']/3600:>10.2f}")

    # Sanity checks
    all_work_set = set(train_works) | set(val_works) | set(test_works)
    assert len(all_work_set) == len(train_works) + len(val_works) + len(test_works), \
        "OVERLAP detected between splits!"
    assert all_work_set == set(work_ids), "Some works dropped from splits!"

    print(f"\nSaved: {splits_path}", flush=True)
    print("Sanity checks passed (no work overlap)", flush=True)


if __name__ == "__main__":
    main()
