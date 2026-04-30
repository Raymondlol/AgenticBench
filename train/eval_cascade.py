#!/usr/bin/env python3
"""
Cascade baseline: ja_audio --(ASR)--> ja_text --(MT)--> zh_text.

Two cascade variants:
  --variant generic  : Whisper-large-v3 + NLLB-200-distilled-600M
                       (weak generic baseline: shows what off-the-shelf does)
  --variant anime    : Anime-Whisper (litagin/anime-whisper) + Qwen2.5-7B-Instruct
                       (domain-expert baseline: ASMR/voice-acting-tuned ASR
                        + strong Chinese LLM with ASMR-aware prompt)

Reports BLEU + chrF + BERTScore vs. VTT reference.

Usage:
  python train/eval_cascade.py --variant generic --output out/cascade_generic.json
  python train/eval_cascade.py --variant anime   --output out/cascade_anime.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import yaml


# ── Anime-Whisper note: requires no_repeat_ngram_size=0, repetition_penalty=1.0,
#    NO initial prompt. Different from Whisper-large-v3's defaults.
ANIME_WHISPER_GEN_KWARGS = {
    "no_repeat_ngram_size": 0,
    "repetition_penalty": 1.0,
}


# ── ASR stage ───────────────────────────────────────────────────────


def transcribe_whisper(audios, sr, processor, model, device, dtype,
                       num_beams: int, max_new_tokens: int,
                       extra_gen_kwargs: dict | None = None) -> List[str]:
    """Generic Whisper transcription (works for whisper-large-v3 and anime-whisper)."""
    import torch

    inputs = processor(audios, sampling_rate=sr, return_tensors="pt", padding=True)
    input_features = inputs.input_features.to(device, dtype=dtype)

    forced_ids = processor.get_decoder_prompt_ids(language="japanese", task="transcribe")

    gen_kwargs = dict(
        forced_decoder_ids=forced_ids,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    if extra_gen_kwargs:
        gen_kwargs.update(extra_gen_kwargs)

    with torch.no_grad():
        pred_ids = model.generate(input_features, **gen_kwargs)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)


# ── MT stage ────────────────────────────────────────────────────────


def translate_nllb(ja_texts: List[str], tokenizer, model, device, dtype,
                   src_lang: str, tgt_lang: str,
                   num_beams: int, max_new_tokens: int) -> List[str]:
    """NLLB encoder-decoder translation."""
    import torch
    tokenizer.src_lang = src_lang
    inputs = tokenizer(ja_texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=256).to(device)
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)


def translate_qwen(ja_texts: List[str], tokenizer, model, device, dtype,
                   prompt_template: str,
                   num_beams: int, max_new_tokens: int) -> List[str]:
    """Qwen causal LLM translation via chat template + prompt."""
    import torch

    outputs = []
    # Qwen2.5-7B is large enough that we process one-by-one to keep memory predictable.
    # Could batch with padding if performance becomes an issue.
    for ja in ja_texts:
        prompt = prompt_template.format(ja_text=ja.strip())
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # deterministic for eval
                num_beams=num_beams,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # Strip input prompt
        new_ids = gen_ids[:, inputs.input_ids.shape[1]:]
        text_out = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]
        outputs.append(text_out.strip())
    return outputs


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/config.yaml")
    parser.add_argument("--variant", choices=["generic", "anime"], default="generic")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None,
                        help="Default: out/cascade_{variant}_results.json")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override config eval.batch_size")
    parser.add_argument("--mt-lora-ckpt", default=None,
                        help="Path to LoRA adapter dir to apply on top of MT model "
                             "(only valid for --variant anime / mt_type qwen)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    eval_cfg = cfg["eval"]
    variant_cfg = eval_cfg["cascade_variants"][args.variant]
    batch_size = args.batch_size or eval_cfg["batch_size"]

    if args.output is None:
        args.output = f"out/cascade_{args.variant}_results.json"

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

    print(f"Variant:  {args.variant}", flush=True)
    print(f"ASR:      {variant_cfg['asr_model']}", flush=True)
    print(f"MT:       {variant_cfg['mt_model']} (type={variant_cfg['mt_type']})", flush=True)
    print(f"Split:    {args.split} ({len(ds)} examples)", flush=True)

    # Load ASR model (Whisper-architecture, both variants)
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"\nLoading ASR: {variant_cfg['asr_model']}", flush=True)
    asr_proc = WhisperProcessor.from_pretrained(variant_cfg["asr_model"])
    asr_model = WhisperForConditionalGeneration.from_pretrained(
        variant_cfg["asr_model"], torch_dtype=dtype
    ).to(device)
    asr_model.eval()

    # Anime-Whisper extra gen kwargs
    extra_asr_kwargs = ANIME_WHISPER_GEN_KWARGS if args.variant == "anime" else None

    # Load MT model
    mt_type = variant_cfg["mt_type"]
    print(f"Loading MT: {variant_cfg['mt_model']}", flush=True)
    if mt_type == "nllb":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        mt_tok = AutoTokenizer.from_pretrained(
            variant_cfg["mt_model"], src_lang=variant_cfg["mt_src_lang"]
        )
        mt_model = AutoModelForSeq2SeqLM.from_pretrained(
            variant_cfg["mt_model"], torch_dtype=dtype
        ).to(device)
    elif mt_type == "qwen":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        mt_tok = AutoTokenizer.from_pretrained(variant_cfg["mt_model"])
        mt_model = AutoModelForCausalLM.from_pretrained(
            variant_cfg["mt_model"], torch_dtype=dtype, device_map="auto"
        )
        # Optionally apply LoRA adapter (e.g. ASMR-domain fine-tune)
        if args.mt_lora_ckpt:
            print(f"Applying LoRA adapter: {args.mt_lora_ckpt}", flush=True)
            from peft import PeftModel
            mt_model = PeftModel.from_pretrained(mt_model, args.mt_lora_ckpt)
            mt_model = mt_model.merge_and_unload()
    else:
        raise ValueError(f"Unknown mt_type: {mt_type}")
    mt_model.eval()

    # Run cascade
    refs, ja_hyps, zh_hyps, metas = [], [], [], []
    t0 = time.time()
    n = len(ds)

    for i in range(0, n, batch_size):
        batch = ds[i:i + batch_size]
        audios = [a["array"] for a in batch["audio"]]
        sr = batch["audio"][0]["sampling_rate"]

        # Stage 1: ASR
        ja_texts = transcribe_whisper(
            audios, sr, asr_proc, asr_model, device, dtype,
            num_beams=eval_cfg.get("num_beams", 5),
            max_new_tokens=eval_cfg["max_new_tokens"],
            extra_gen_kwargs=extra_asr_kwargs,
        )

        # Stage 2: MT
        if mt_type == "nllb":
            zh_texts = translate_nllb(
                ja_texts, mt_tok, mt_model, device, dtype,
                src_lang=variant_cfg["mt_src_lang"],
                tgt_lang=variant_cfg["mt_tgt_lang"],
                num_beams=eval_cfg.get("num_beams", 5),
                max_new_tokens=eval_cfg["max_new_tokens"],
            )
        else:  # qwen
            zh_texts = translate_qwen(
                ja_texts, mt_tok, mt_model, device, dtype,
                prompt_template=variant_cfg["mt_prompt"],
                num_beams=1,  # greedy is fine for translation
                max_new_tokens=eval_cfg["max_new_tokens"],
            )

        refs.extend(batch["zh_text"])
        ja_hyps.extend(ja_texts)
        zh_hyps.extend(zh_texts)
        metas.extend([
            {"work_id": w, "audio_file": af, "segment_idx": int(si),
             "start_s": float(st), "dur_s": float(d)}
            for w, af, si, st, d in zip(
                batch["work_id"], batch["audio_file"],
                batch["segment_idx"], batch["start_s"], batch["dur_s"]
            )
        ])

        if (i // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + len(audios)) / max(elapsed, 1e-6)
            eta = (n - i - len(audios)) / max(rate, 1e-6)
            print(f"  [{i+len(audios)}/{n}] rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)

    print(f"\nInference time: {(time.time()-t0)/60:.1f}min", flush=True)

    # Compute metrics
    print("Computing metrics...", flush=True)
    sys.path.insert(0, str(Path(__file__).parent))
    from compute_metrics import compute_all_metrics
    metrics = compute_all_metrics(zh_hyps, refs, normalize=True, skip_bertscore=False)

    output = {
        "method": f"cascade_{args.variant}",
        "asr_model": variant_cfg["asr_model"],
        "mt_model": variant_cfg["mt_model"],
        "mt_type": mt_type,
        "split": args.split,
        "n_examples": len(refs),
        "metrics": metrics,
        "samples": [
            {"ref": r, "ja_hyp": jh, "zh_hyp": zh, **m}
            for r, jh, zh, m in zip(refs, ja_hyps, zh_hyps, metas)
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    # Print summary
    print(f"\n{'='*64}")
    print(f"  CASCADE BASELINE [{args.variant.upper()}]")
    print(f"{'='*64}")
    print(f"  ASR:    {variant_cfg['asr_model']}")
    print(f"  MT:     {variant_cfg['mt_model']}")
    print(f"  Split:  {args.split} ({len(refs)} examples)")
    print(f"\n  --- Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, list):
            continue
        print(f"  {k:<24s}: {v}")
    print(f"\n  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
