# ASMR Pilot Study — Project Context for Claude Code Agent

You are running on a **vast.ai 4× RTX 4090 instance in Oslo, Norway** to execute a pilot study: fine-tune translation models on Japanese ASMR audio → Chinese subtitle data, then compare against off-the-shelf baselines.

This is a **research pilot**, not production. Decisions to make: should we scale up to 3,000-7,000h full data and a real training run? Pilot answers: "is FT signal big enough?"

---

## Critical context

### Data
- `Raymxnd/asmr-pilot-50h` (HF Hub, private, you have read access via env `HF_TOKEN`)
- Already cached at `/workspace/.hf_home/hub/datasets--Raymxnd--asmr-pilot-50h/`
- 50h Japanese ASMR audio + Chinese fan-translated subtitles (NSFW)
- 138 works → 41,490 segments → train 33,479 / val 3,772 / test 4,239
- Splits BY work_id (no speaker leakage); `splits.json` in HF repo

### Pre-computed assets (DO NOT redo)
- ✅ `out/phase1_baselines/cascade_anime_results.json` — B' (Anime+Qwen) full eval, BLEU 21 / chrF ~15 / nsfw_recall TBD
- ✅ `/workspace/ASMR-Data/hf_cache/train_with_ja.jsonl` — 33k (ja_pseudo, zh) pairs from Anime-Whisper, ready for Qwen LoRA training

### Models already in HF cache
Located in `/workspace/.hf_home/hub/`:
- `models--openai--whisper-large-v3` (3GB)
- `models--litagin--anime-whisper` (1.6GB)
- `models--facebook--nllb-200-distilled-600M` (1.2GB)
- `models--facebook--seamless-m4t-v2-large` (5GB)
- `models--Qwen--Qwen2.5-7B-Instruct` (14GB)

### Models still to download
- `SakuraLLM/Sakura-14B-Qwen2.5-v1.0-AWQ` (~8GB) — for B'' Sakura baseline
- `SakuraLLM/Sakura-14B-Qwen2.5-v1.0` (28GB) — for QLoRA training base

---

## The 6-way comparison goal

| ID | System | Status |
|----|--------|--------|
| **A** | SeamlessM4T-v2-large zero-shot | TODO (rerun, prev crashed at BERTScore) |
| **B** | Whisper-large-v3 + NLLB-200 | TODO (rerun, prev crashed at BERTScore) |
| **B'** | Anime-Whisper + Qwen2.5-7B-Instruct | ✅ DONE |
| **B''** | Anime-Whisper + **Sakura-14B-AWQ** | TODO |
| **C-Cascade** | Anime-Whisper + Sakura-14B + **QLoRA** | TODO (use generate_ja_pseudo output) |
| **C-E2E** | SeamlessM4T-v2-large + LoRA | TODO |

Final output: `out/phase4_compare.md` — 6-way table with chrF, BLEU, NSFW vocab recall.

---

## Critical findings already validated (do not waste time re-validating)

1. **Audio is Japanese**, subtitles are Chinese fan-translation (not Chinese dub).
2. **Qwen2.5-7B-Instruct softens NSFW content** but doesn't refuse: まんこ → "下面"/"那个地方" instead of "小穴". Also has knowledge gaps: キツイ → "辣" (mistranslation).
3. **Abliterated Qwen does NOT solve this** — removing refusal direction ≠ filling knowledge gap. Don't waste time on abliterated variants.
4. **BERTScore is unreliable for this domain** — bert-base-chinese can't distinguish ASMR vocabulary. We dropped it. Primary metrics: chrF + BLEU + NSFW vocab hit-rate (custom).
5. **bert_score 0.3.13 is incompatible with transformers 5.7** — `BertTokenizer.build_inputs_with_special_tokens` removed. We catch and skip.

---

## Code architecture

```
/workspace/AgenticBench/
├── train/
│   ├── config.yaml              # all hyperparams
│   ├── data_utils.py            # auto-loads dataset from HF Hub
│   ├── compute_metrics.py       # BLEU + chrF + nsfw_vocab_recall
│   ├── eval_e2e.py              # SeamlessM4T eval (--zero-shot or --ckpt)
│   ├── eval_cascade.py          # Cascade eval (--variant generic|anime, --use-sakura, --mt-lora-ckpt)
│   ├── generate_ja_pseudo.py    # done already, output cached
│   ├── train_lora.py            # SeamlessM4T LoRA training (DDP via accelerate)
│   └── train_qwen_lora.py       # Qwen/Sakura LoRA training (--use-sakura → QLoRA)
├── scripts/
│   ├── setup_vast.sh            # already ran, env is configured
│   ├── run_baselines.sh         # parallel baselines runner
│   ├── run_full_pilot.sh        # full orchestrator
│   └── upload_dataset.py
└── out/                         # results go here
```

### Conventions
- Audio segments stored in HF dataset's `audio` column (sampling_rate=16000, decoded by torchcodec 0.3)
- `--use-sakura` flag swaps base MT model for Sakura
- `--mt-lora-ckpt PATH` loads a LoRA adapter on top of Qwen-family models
- All training uses `accelerate launch --num_processes=4` for DDP
- Effective batch size already tuned — don't change unless OOM

### Environment config
- Python 3.12, torch 2.7.1+cu126, transformers 5.7.0, peft 0.19, datasets 3.6 (NOT 4.x)
- torchcodec 0.3.0 (NOT 0.4)
- bitsandbytes 0.49 (for QLoRA)
- autoawq 0.2.9 (for Sakura-AWQ inference)
- HF_TOKEN in `~/.bashrc`, also in `.env`
- HF_HOME=/workspace/.hf_home
- accelerate config at `~/.cache/huggingface/accelerate/default_config.yaml` (4-GPU DDP)

---

## DO NOT

- ❌ **Don't run `rm -rf` on `/workspace/.hf_home`** — that's 30GB of cached models, painful to redownload.
- ❌ **Don't `git push`** at all — vast.ai has no GitHub credentials. Just commit locally; human will pull.
- ❌ **Don't change config.yaml's eval settings without good reason** — eval batch sizes are tuned for 24GB cards.
- ❌ **Don't rerun cascade_anime baseline** — already complete.
- ❌ **Don't try abliterated Qwen** — already verified it doesn't help.
- ❌ **Don't downgrade torchcodec or transformers** — current versions are tuned to work together; the dance to get them compatible is documented in CLAUDE.md.
- ❌ **Don't skip the diversity caps in pipeline.py** — they exist on purpose.
- ❌ **Don't use BERTScore** — it's broken AND unreliable for this domain.

## DO

- ✅ **Use chrF as primary metric** (char-level F1, perfect for CN no-tokenization).
- ✅ **Save partial results aggressively** — write JSON files incrementally so a crash doesn't lose work.
- ✅ **Log all decisions to `out/agent_decisions.md`** with timestamp + rationale.
- ✅ **Keep all 4 GPUs busy** — when one job finishes, immediately schedule the next.
- ✅ **Commit + push after each successful phase** so the human can sync results.
- ✅ **If something breaks: diagnose → fix → retry → log to agent_decisions.md.** Don't bail.
- ✅ **Use `fuser -k /dev/nvidia*` to clean up GPU zombies** if `nvidia-smi` shows leaked memory but processes are gone.
- ✅ **When in doubt, prefer cheaper experiments first.** Pilot budget is small (~$10).

---

## GPU scheduling rules of thumb

- Inference jobs (1 GPU each, mostly): SeamlessM4T zero-shot, generic cascade, anime cascade, Sakura cascade
- Training jobs (4 GPU DDP): Qwen LoRA, Sakura QLoRA, SeamlessM4T LoRA
- Generation jobs (1 GPU): generate_ja_pseudo

If 1 GPU is free, you can fit ANY of: zero-shot eval, cascade eval, ja_pseudo generation, FT eval.
If 4 GPUs all idle, prefer: training (DDP). 4× DDP is much faster than 1×.

If you have a stuck training job and idle inference work, **kill the training and run inference on freed GPUs**, then resume training.

---

## Reporting / human handoff

The human user is asleep / offline / doing other things. They check back via:
- `cat out/STATUS.md` — phase-by-phase progress (you write to this)
- `cat out/agent_decisions.md` — your decision log
- `cat out/phase4_compare.md` — final report (you generate at end)
- `tail -f /tmp/pilot.log` — running log

When done, write `DONE` to first line of `out/STATUS.md`. Push everything in `out/` to git.

---

## When to ask the human

You're autonomous, but ASK (write a question into `out/AGENT_QUESTION.md` and stop) if:
- A phase has crashed 3+ times after fix attempts
- A decision would significantly increase cost ($20+ delta)
- You discover the data is corrupt / different from spec
- The pilot's main hypothesis appears refuted (no FT improvement) — confirm before stopping
