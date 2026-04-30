#!/usr/bin/env python3
"""
End-to-end SeamlessM4T-v2 speech translation eval (zero-shot or fine-tuned).

  --zero-shot         use base model from HF hub (no fine-tuning)
  --ckpt PATH         load LoRA adapter from checkpoint dir

Reports BLEU + chrF + BERTScore vs. VTT reference on the chosen split.

Usage:
  # zero-shot baseline
  python train/eval_e2e.py --zero-shot --output out/zeroshot_results.json

  # fine-tuned
  python train/eval_e2e.py --ckpt out/seamless-lora-pilot/best --output out/ft_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--zero-shot", action="store_true",
                        help="Use base model without LoRA")
    parser.add_argument("--ckpt", default=None,
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default="out/e2e_results.json")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if not args.zero_shot and not args.ckpt:
        print("ERROR: must pass either --zero-shot or --ckpt PATH", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(Path(args.config).read_text())
    eval_cfg = cfg["eval"]

    # Load dataset + splits (auto-detects local disk vs HF Hub)
    sys.path.insert(0, str(Path(__file__).parent))
    from data_utils import load_pilot_dataset, load_pilot_splits
    ds = load_pilot_dataset(cfg)
    splits = load_pilot_splits(cfg)
    target_works = set(splits[args.split]["work_ids"])

    indices = [i for i, ex in enumerate(ds) if ex["work_id"] in target_works]
    ds = ds.select(indices)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"Eval split: {args.split}", flush=True)
    print(f"Examples: {len(ds)}", flush=True)

    # Load model
    import torch
    from transformers import (
        AutoProcessor, SeamlessM4Tv2ForSpeechToText,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading SeamlessM4T-v2: {cfg['model_id']}", flush=True)
    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        cfg["model_id"], torch_dtype=dtype
    ).to(device)

    if args.ckpt:
        print(f"Loading LoRA adapter from: {args.ckpt}", flush=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.ckpt)
        model = model.merge_and_unload()  # merge for faster inference

    model.eval()

    # Inference
    refs = []
    hyps = []
    metas = []

    t0 = time.time()
    n = len(ds)
    tgt_lang = cfg["tgt_lang"]

    for i in range(0, n, args.batch_size):
        batch = ds[i:i + args.batch_size]
        audios = [a["array"] for a in batch["audio"]]
        sr = batch["audio"][0]["sampling_rate"]

        inputs = processor(audios=audios, sampling_rate=sr,
                           return_tensors="pt", padding=True)
        # Move to device with proper dtype for input_features
        inputs = {k: v.to(device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
                  for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                tgt_lang=tgt_lang,
                num_beams=eval_cfg.get("num_beams", 5),
                max_new_tokens=eval_cfg["max_new_tokens"],
            )
        # SeamlessM4Tv2 returns SequenceClassifierOutput in some versions; handle both
        if hasattr(out_ids, "sequences"):
            out_ids = out_ids.sequences

        zh_texts = processor.batch_decode(out_ids, skip_special_tokens=True)

        refs.extend(batch["zh_text"])
        hyps.extend(zh_texts)
        metas.extend([
            {"work_id": w, "audio_file": af, "segment_idx": int(si),
             "start_s": float(st), "dur_s": float(d)}
            for w, af, si, st, d in zip(
                batch["work_id"], batch["audio_file"],
                batch["segment_idx"], batch["start_s"], batch["dur_s"]
            )
        ])

        if (i // args.batch_size) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + len(audios)) / max(elapsed, 1e-6)
            eta = (n - i - len(audios)) / max(rate, 1e-6)
            print(f"  [{i+len(audios)}/{n}] rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)

    print(f"\nInference time: {(time.time()-t0)/60:.1f}min", flush=True)

    # Compute metrics
    print("Computing metrics...", flush=True)
    sys.path.insert(0, str(Path(__file__).parent))
    from compute_metrics import compute_all_metrics
    metrics = compute_all_metrics(hyps, refs, normalize=True, skip_bertscore=False)

    # Build output
    output = {
        "method": "e2e_seamlessm4t_v2",
        "model_id": cfg["model_id"],
        "ckpt": args.ckpt,
        "zero_shot": args.zero_shot,
        "split": args.split,
        "tgt_lang": tgt_lang,
        "n_examples": len(refs),
        "metrics": metrics,
        "samples": [
            {"ref": r, "hyp": h, **m}
            for r, h, m in zip(refs, hyps, metas)
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    label = "ZERO-SHOT" if args.zero_shot else "FINE-TUNED"
    print(f"\n{'='*60}")
    print(f"  END-TO-END SEAMLESSM4T-V2 RESULTS [{label}]")
    print(f"{'='*60}")
    print(f"  Model:    {cfg['model_id']}")
    if args.ckpt:
        print(f"  Adapter:  {args.ckpt}")
    print(f"  Split:    {args.split}  ({len(refs)} examples)")
    print(f"\n  --- Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, list):
            continue
        print(f"  {k:<24s}: {v}")

    print(f"\n  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
