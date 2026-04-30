#!/usr/bin/env python3
"""
LoRA fine-tune Qwen2.5-7B-Instruct on ASMR ja->zh translation pairs.

Reads (ja_text, zh_text) pairs from train_with_ja.jsonl (output of
generate_ja_pseudo.py), formats them as instruction-response pairs,
and trains a LoRA adapter on Qwen.

Usage:
  accelerate launch train/train_qwen_lora.py [--config train/config.yaml]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def format_example(ja_text: str, zh_text: str, prompt_template: str,
                   tokenizer, max_len: int = 512) -> dict:
    """Format one (ja, zh) pair as a chat example for SFT."""
    user_prompt = prompt_template.format(ja_text=ja_text.strip())
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": zh_text.strip()},
    ]
    # Build the full conversation as a single string with proper template tokens.
    # We'll mask the user/system parts in labels later.
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # For label masking: tokenize user-only prefix to know where assistant starts
    user_only = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(full_text, truncation=True, max_length=max_len,
                         padding=False, return_tensors=None)["input_ids"]
    prefix_ids = tokenizer(user_only, truncation=True, max_length=max_len,
                            padding=False, return_tensors=None)["input_ids"]

    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    labels = labels[:len(full_ids)]
    if len(labels) < len(full_ids):
        labels = labels + [-100] * (len(full_ids) - len(labels))

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/config.yaml")
    parser.add_argument("--data-jsonl", default=None,
                        help="Default: ASMR-Data/hf_cache/train_with_ja.jsonl")
    parser.add_argument("--output-dir", default="out/qwen-lora-pilot")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override num_train_epochs (smoke test)")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    variant = cfg["eval"]["cascade_variants"]["anime"]
    prompt_template = variant["mt_prompt"]

    data_jsonl = Path(args.data_jsonl) if args.data_jsonl else Path(
        "/workspace/ASMR-Data/hf_cache/train_with_ja.jsonl"
        if Path("/workspace").exists()
        else "/Users/raymond/WorkSpace/ASMR-Data/hf_cache/train_with_ja.jsonl"
    )
    print(f"Reading training data from: {data_jsonl}", flush=True)
    if not data_jsonl.exists():
        print(f"ERROR: not found. Run generate_ja_pseudo.py first.", file=sys.stderr)
        sys.exit(1)

    # Read jsonl
    records = [json.loads(line) for line in open(data_jsonl)]
    print(f"Records: {len(records)}", flush=True)

    # Filter: drop pairs where ja or zh text is suspicious
    records = [r for r in records if 1 <= len(r["ja_text"]) <= 500
                                       and 1 <= len(r["zh_text"]) <= 500]
    print(f"After length filter: {len(records)}", flush=True)

    # Train/val split (small val from train for monitoring loss)
    import random
    random.Random(42).shuffle(records)
    n_val = min(200, len(records) // 50)
    val_records = records[:n_val]
    train_records = records[n_val:]
    print(f"Train: {len(train_records)}  | Val: {len(val_records)}", flush=True)

    # Load tokenizer + model
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, set_seed,
        Trainer, TrainingArguments, DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    set_seed(42)
    print(f"Loading tokenizer + model: {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, attn_implementation="sdpa")
    model.gradient_checkpointing_enable()

    # LoRA: target attention + MLP projections (typical for Qwen2.5)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Tokenize datasets
    def tokenize_fn(record):
        return format_example(record["ja_text"], record["zh_text"],
                              prompt_template, tokenizer, max_len=args.max_len)

    print("Tokenizing...", flush=True)
    train_ds = Dataset.from_list(train_records).map(
        tokenize_fn, remove_columns=["work_id", "audio_file", "segment_idx",
                                       "start_s", "end_s", "ja_text", "zh_text"],
        num_proc=4)
    val_ds = Dataset.from_list(val_records).map(
        tokenize_fn, remove_columns=["work_id", "audio_file", "segment_idx",
                                       "start_s", "end_s", "ja_text", "zh_text"],
        num_proc=4)
    # Filter overly long sequences (after tokenization)
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) <= args.max_len)
    val_ds = val_ds.filter(lambda x: len(x["input_ids"]) <= args.max_len)
    print(f"After length filter: train={len(train_ds)} val={len(val_ds)}", flush=True)

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ngpu = torch.cuda.device_count()
    per_device_bs = 2  # 7B + 384-token seq + LoRA → fits ~12GB on each 4090
    grad_accum = max(1, 16 // (per_device_bs * ngpu))  # effective ~16-32

    print(f"GPUs: {ngpu}  per-device bs: {per_device_bs}  grad_accum: {grad_accum}  "
          f"effective bs: {per_device_bs * grad_accum * ngpu}", flush=True)

    ta = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="steps" if not args.no_eval else "no",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=not args.no_eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=42,
        remove_unused_columns=False,
        ddp_find_unused_parameters=ngpu > 1,  # LoRA needs this
    )
    if args.max_steps:
        ta["max_steps"] = args.max_steps
        ta.pop("num_train_epochs", None)
    training_args = TrainingArguments(**ta)

    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if not args.no_eval else None,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("Starting training...", flush=True)
    train_result = trainer.train()

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    print(f"Saved adapter to {best_dir}", flush=True)

    metrics = train_result.metrics
    metrics["train_examples"] = len(train_ds)
    metrics["val_examples"] = len(val_ds)
    (output_dir / "train_results.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Train metrics: {metrics}", flush=True)


if __name__ == "__main__":
    main()
