#!/usr/bin/env python3
"""
Generate Japanese pseudo-labels for the training split using Anime-Whisper.

Output: HF dataset with columns (audio_id, work_id, segment_idx, ja_text, zh_text)
Saved to ASMR-Data/hf_cache/train_with_ja/ as Arrow shards.

This is the prerequisite for LoRA fine-tuning Qwen on (ja, zh) translation pairs.

Usage:
  python train/generate_ja_pseudo.py [--config train/config.yaml]
                                      [--split train]
                                      [--batch-size 16]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/config.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default=None,
                        help="Default: ASMR-Data/hf_cache/train_with_ja/")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap (smoke test)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # Load dataset + splits
    sys.path.insert(0, str(Path(__file__).parent))
    from data_utils import load_pilot_dataset, load_pilot_splits
    ds = load_pilot_dataset(cfg)
    splits = load_pilot_splits(cfg)
    target_works = set(splits[args.split]["work_ids"])

    indices = [i for i, ex in enumerate(ds) if ex["work_id"] in target_works]
    ds = ds.select(indices)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"Split: {args.split}  | Samples: {len(ds)}", flush=True)

    # Load Anime-Whisper
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    asr_id = cfg["eval"]["cascade_variants"]["anime"]["asr_model"]
    print(f"Loading ASR: {asr_id}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    asr_proc = WhisperProcessor.from_pretrained(asr_id)
    asr_model = WhisperForConditionalGeneration.from_pretrained(
        asr_id, torch_dtype=dtype).to(device).eval()
    forced_ids = asr_proc.get_decoder_prompt_ids(language="japanese", task="transcribe")

    # Run inference
    out_records = []
    t0 = time.time()
    n = len(ds)
    for i in range(0, n, args.batch_size):
        batch = ds[i:i + args.batch_size]
        audios = [a["array"] for a in batch["audio"]]
        sr = batch["audio"][0]["sampling_rate"]

        inputs = asr_proc(audios, sampling_rate=sr, return_tensors="pt", padding=True)
        feats = inputs.input_features.to(device, dtype=dtype)
        with torch.no_grad():
            ids = asr_model.generate(
                feats,
                forced_decoder_ids=forced_ids,
                no_repeat_ngram_size=0,
                repetition_penalty=1.0,
                max_new_tokens=128,
                num_beams=1,  # greedy is fine for pseudo-labels
            )
        ja_texts = asr_proc.batch_decode(ids, skip_special_tokens=True)

        for j, ja in enumerate(ja_texts):
            out_records.append({
                "work_id": batch["work_id"][j],
                "audio_file": batch["audio_file"][j],
                "segment_idx": int(batch["segment_idx"][j]),
                "start_s": float(batch["start_s"][j]),
                "end_s": float(batch["end_s"][j]),
                "ja_text": ja.strip(),
                "zh_text": batch["zh_text"][j],
            })

        if (i // args.batch_size) % 5 == 0:
            elapsed = time.time() - t0
            rate = (i + len(audios)) / max(elapsed, 1e-6)
            eta = (n - i - len(audios)) / max(rate, 1e-6)
            print(f"  [{i+len(audios)}/{n}] rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)

    print(f"\nTotal time: {(time.time()-t0)/60:.1f}min", flush=True)
    print(f"Total records: {len(out_records)}", flush=True)

    # Save as JSONL (lighter than Arrow + works directly with LoRA training)
    out_path = Path(args.output) if args.output else Path(
        "/workspace/ASMR-Data/hf_cache" if Path("/workspace").exists()
        else "/Users/raymond/WorkSpace/ASMR-Data/hf_cache"
    ) / f"{args.split}_with_ja.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved to {out_path}", flush=True)
    print(f"Sample first 3 records:", flush=True)
    for r in out_records[:3]:
        print(f"  ja: {r['ja_text'][:60]}", flush=True)
        print(f"  zh: {r['zh_text'][:60]}", flush=True)
        print()


if __name__ == "__main__":
    main()
