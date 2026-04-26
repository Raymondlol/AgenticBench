#!/usr/bin/env python3
"""
LoRA fine-tuning of SeamlessM4T-v2-large for ja->zh ASMR speech translation.

Architecture:
  - Frozen base model
  - LoRA on attention modules (q/k/v/out_proj) of speech encoder + text decoder
  - Loss: cross-entropy on target zh tokens (teacher forcing)

Usage:
  accelerate launch train/train_lora.py [--config train/config.yaml] [--max-steps N]

Smoke test (forward+backward only, no eval):
  python train/train_lora.py --max-steps 10 --no-eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# ── Data collator ───────────────────────────────────────────────────


@dataclass
class SeamlessSTCollator:
    """Pad audio inputs and tokenize labels for SeamlessM4T-v2 ST training.

    Uses the model's processor for both audio feature extraction and
    text tokenization. Labels are tokenized in the target language so
    that decoder generates target tokens directly.
    """
    processor: Any
    tgt_lang: str
    max_target_tokens: int = 128
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict]) -> dict:
        import torch

        audios = [f["audio"]["array"] for f in features]
        sr = features[0]["audio"]["sampling_rate"]
        labels_text = [f["zh_text"] for f in features]

        # Audio features
        audio_inputs = self.processor(
            audios=audios, sampling_rate=sr,
            return_tensors="pt", padding=True,
        )

        # Labels: tokenize target text in tgt_lang
        # SeamlessM4T processor accepts text via `text_target` arg in newer versions,
        # or we set src/tgt lang on tokenizer side.
        if hasattr(self.processor, "tokenizer"):
            tokenizer = self.processor.tokenizer
            tokenizer.tgt_lang = self.tgt_lang
            label_inputs = tokenizer(
                labels_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_target_tokens,
            )
        else:
            label_inputs = self.processor(
                text=labels_text,
                src_lang=self.tgt_lang,
                return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_target_tokens,
            )

        labels = label_inputs.input_ids.clone()
        # Mask padding tokens in loss
        if hasattr(label_inputs, "attention_mask"):
            labels[label_inputs.attention_mask == 0] = self.label_pad_token_id

        batch = {**{k: v for k, v in audio_inputs.items()}, "labels": labels}
        return batch


# ── Model setup ─────────────────────────────────────────────────────


def setup_model(model_id: str, lora_cfg: dict, dtype, gradient_checkpointing: bool):
    from transformers import SeamlessM4Tv2ForSpeechToText
    from peft import LoraConfig, get_peft_model

    print(f"Loading base model: {model_id}", flush=True)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_id, torch_dtype=dtype)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Inspect modules to verify target_modules are reachable
    target_modules = lora_cfg["target_modules"]
    matching = [n for n, _ in model.named_modules()
                if any(n.endswith(t) for t in target_modules)]
    print(f"Found {len(matching)} matching modules for LoRA targets {target_modules}",
          flush=True)
    if not matching:
        print("WARNING: no modules matched target_modules. Listing first 30 module names:",
              flush=True)
        for n, _ in list(model.named_modules())[:30]:
            print(f"    {n}")

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=target_modules,
        bias=lora_cfg.get("bias", "none"),
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# ── Eval (chrF on val split) ────────────────────────────────────────


def eval_chrf_on_dataset(model, processor, ds, tgt_lang: str,
                         max_new_tokens: int, num_beams: int,
                         max_eval_examples: int, batch_size: int, device, dtype) -> dict:
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from compute_metrics import compute_chrf, normalize_zh

    if max_eval_examples and len(ds) > max_eval_examples:
        ds = ds.select(range(max_eval_examples))

    refs, hyps = [], []
    model.eval()
    for i in range(0, len(ds), batch_size):
        batch = ds[i:i + batch_size]
        audios = [a["array"] for a in batch["audio"]]
        sr = batch["audio"][0]["sampling_rate"]
        inputs = processor(audios=audios, sampling_rate=sr,
                           return_tensors="pt", padding=True)
        inputs = {k: v.to(device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
                  for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, tgt_lang=tgt_lang,
                max_new_tokens=max_new_tokens, num_beams=num_beams,
            )
        if hasattr(out_ids, "sequences"):
            out_ids = out_ids.sequences
        zh_texts = processor.batch_decode(out_ids, skip_special_tokens=True)
        refs.extend([normalize_zh(t) for t in batch["zh_text"]])
        hyps.extend([normalize_zh(t) for t in zh_texts])

    model.train()
    return compute_chrf(hyps, refs)


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/config.yaml")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override num_train_epochs with hard step cap (smoke test)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip eval during training (smoke test)")
    parser.add_argument("--max-eval-samples", type=int, default=200,
                        help="Cap eval examples for periodic eval")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint dir")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    train_cfg = cfg["train"]

    import torch
    from transformers import (
        AutoProcessor, Trainer, TrainingArguments, set_seed,
    )
    from datasets import load_from_disk

    set_seed(train_cfg["seed"])

    # Load dataset + splits
    print("Loading dataset...", flush=True)
    ds = load_from_disk(str(Path(cfg["dataset"]["local_path"])))
    splits = json.loads(Path(cfg["dataset"]["splits_path"]).read_text())

    train_works = set(splits["train"]["work_ids"])
    val_works = set(splits["val"]["work_ids"])

    # Filter to train/val by work_id
    all_idx = list(range(len(ds)))
    train_idx = [i for i in all_idx if ds[i]["work_id"] in train_works]
    val_idx = [i for i in all_idx if ds[i]["work_id"] in val_works]
    train_ds = ds.select(train_idx)
    val_ds = ds.select(val_idx)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}", flush=True)

    # Filter audio segments that exceed max_audio_seconds
    max_aud_s = train_cfg["max_audio_seconds"]
    sr = cfg["segmentation"]["sample_rate"]
    train_ds = train_ds.filter(lambda x: x["dur_s"] <= max_aud_s)
    val_ds = val_ds.filter(lambda x: x["dur_s"] <= max_aud_s)
    print(f"After max_audio_seconds={max_aud_s} filter: train={len(train_ds)} val={len(val_ds)}",
          flush=True)

    # Processor + Model
    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    dtype = torch.bfloat16 if train_cfg.get("bf16") else torch.float32
    model = setup_model(cfg["model_id"], cfg["lora"], dtype,
                        train_cfg.get("gradient_checkpointing", False))

    # Data collator
    collator = SeamlessSTCollator(
        processor=processor,
        tgt_lang=cfg["tgt_lang"],
        max_target_tokens=train_cfg["max_target_tokens"],
    )

    # Training arguments
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ta_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["grad_accum_steps"],
        learning_rate=train_cfg["lr"],
        warmup_steps=train_cfg["warmup_steps"],
        lr_scheduler_type=train_cfg["scheduler"],
        num_train_epochs=train_cfg["num_train_epochs"],
        bf16=train_cfg.get("bf16", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        eval_strategy="steps" if not args.no_eval else "no",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg.get("save_total_limit", 3),
        logging_steps=train_cfg.get("logging_steps", 25),
        load_best_model_at_end=not args.no_eval,
        metric_for_best_model=train_cfg.get("metric_for_best_model", "chrf"),
        greater_is_better=train_cfg.get("greater_is_better", True),
        report_to=train_cfg.get("report_to", "none"),
        seed=train_cfg["seed"],
        remove_unused_columns=False,
        label_names=["labels"],
    )

    if args.max_steps:
        ta_kwargs["max_steps"] = args.max_steps
        ta_kwargs.pop("num_train_epochs", None)

    training_args = TrainingArguments(**ta_kwargs)

    # Custom Trainer with chrF eval (HF Trainer's compute_metrics expects logits;
    # here we use generation, so override evaluate())
    class GenerativeTrainer(Trainer):
        def evaluate(self, eval_dataset=None, ignore_keys=None,
                     metric_key_prefix="eval"):
            # Run loss-based eval first (default)
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            # Then run generative chrF eval
            ds_to_eval = eval_dataset if eval_dataset is not None else self.eval_dataset
            try:
                chrf_metrics = eval_chrf_on_dataset(
                    self.model, processor, ds_to_eval,
                    tgt_lang=cfg["tgt_lang"],
                    max_new_tokens=cfg["eval"]["max_new_tokens"],
                    num_beams=cfg["eval"].get("num_beams", 1),  # use 1 during training for speed
                    max_eval_examples=args.max_eval_samples,
                    batch_size=train_cfg["batch_size"],
                    device=self.args.device,
                    dtype=dtype,
                )
                for k, v in chrf_metrics.items():
                    metrics[f"{metric_key_prefix}_{k}"] = v
            except Exception as e:
                print(f"[chrf eval error] {e}", flush=True)
            self.log(metrics)
            return metrics

    trainer = GenerativeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if not args.no_eval else None,
        data_collator=collator,
        tokenizer=processor,
    )

    # Train
    print("Starting training...", flush=True)
    train_result = trainer.train(resume_from_checkpoint=args.resume)

    # Save final adapter
    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    print(f"Saved best adapter to: {best_dir}", flush=True)

    # Save training stats
    metrics = train_result.metrics
    metrics["train_examples"] = len(train_ds)
    metrics["val_examples"] = len(val_ds)
    (output_dir / "train_results.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    print(f"Train metrics: {metrics}", flush=True)


if __name__ == "__main__":
    main()
