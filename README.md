# ASMR Pilot: Japanese Audio → Chinese Subtitle Translation

LoRA fine-tuning of `facebook/seamless-m4t-v2-large` on Japanese ASMR audio
paired with Chinese-translated VTT subtitles, scraped from asmr.one.

## Goal

Validate (on ~50-80h of weakly-aligned data) whether end-to-end speech
translation (ST) can be domain-adapted to ASMR via lightweight LoRA, before
committing to a 3,000h+ scale-up.

**Four-way comparison on identical test set:**

| # | System | What |
|---|--------|------|
| **A** | SeamlessM4T-v2-large zero-shot | E2E baseline (no fine-tuning) |
| **B** | Whisper-large-v3 + NLLB-200 | Generic cascade baseline |
| **B'** | Anime-Whisper + Qwen2.5-7B-Instruct | **Domain-expert cascade** (ASMR-aware ASR + ASMR-prompted LLM translator) |
| **C** | SeamlessM4T-v2-large + LoRA | The pilot model (our fine-tune) |

**Success**: C achieves chrF improvement ≥ +5 over the BEST of {A, B, B'} on the held-out test split. If C beats A but not B', it means LoRA helps but a tuned cascade is still stronger — useful insight for scale-up.

**Budget**: $50-80 on vast.ai H100.

---

## Repo layout

```
.
├── pipeline.py                  # data downloader (asmr.one → ASMR-Data/)
├── vad_model/                   # Whisper-VAD-ASMR-onnx wrapper (legacy, unused for training)
├── eval_*.py                    # standalone eval scripts (legacy ASR/VAD analysis)
├── train/                       # ── pilot training pipeline ──
│   ├── config.yaml              # all hyperparams, model ids, paths
│   ├── requirements-train.txt   # vast.ai install list
│   ├── build_dataset.py         # VTT+audio → HF Arrow dataset
│   ├── split.py                 # 80/10/10 split BY work_id
│   ├── compute_metrics.py       # BLEU, chrF, BERTScore (shared)
│   ├── eval_cascade.py          # Whisper(ja) → NLLB(ja→zh) baseline
│   ├── eval_e2e.py              # SeamlessM4T zero-shot or fine-tuned eval
│   └── train_lora.py            # LoRA fine-tune + periodic chrF eval
├── scripts/
│   ├── upload_dataset.py        # push Arrow dataset → HF Hub private repo
│   └── setup_vast.sh            # on-instance bootstrap (deps + HF login)
├── .gitignore                   # excludes ASMR-Data/, .venv/, out/, *.mp3, etc.
└── README.md
```

Data lives in the sibling `/Users/raymond/WorkSpace/ASMR-Data/` (NOT in repo).

---

## Workflow

### A. Local prep (Mac)

#### 1. Download more works (~120 new, ~64h additional)

```bash
source .venv/bin/activate
python pipeline.py
```

This resumes from `ASMR-Data/meta/pipeline_state.json`. Already-downloaded
works are preserved; new ones are added per the diversity-aware filter
(`MAX_PER_SERIES=1, MAX_PER_CIRCLE=2, MAX_PER_CV=2`, random seed 42).

Expected: ~150 works, ~80h, ~20GB on disk after a few hours.

#### 2. Build HF Arrow dataset

```bash
python train/build_dataset.py
```

Walks `ASMR-Data/chinese_asr/{work_id}/`, parses VTT lines, slices audio with
±0.2s padding, drops segments <1.5s or >30s, writes Arrow shards to
`ASMR-Data/hf_cache/seg_dataset/`. Expected output: ~10-15k examples.

#### 3. Build train/val/test splits

```bash
python train/split.py
```

Deterministic 80/10/10 split **by `work_id`**, written to
`ASMR-Data/hf_cache/splits.json`. Verifies no work appears in two splits.

#### 4. Upload to HF Hub (private dataset)

```bash
huggingface-cli login   # paste your write token
python scripts/upload_dataset.py --repo-id YOURNAME/asmr-pilot-50h
```

Then edit `train/config.yaml`:

```yaml
dataset:
  hub_repo: YOURNAME/asmr-pilot-50h
```

Commit and `git push`.

---

### B. Train + eval on vast.ai

#### Recommended hardware (sweet spot)

| Setup | Wall time | Cost | When to pick |
|---|---|---|---|
| **4× RTX 5090 32GB** | ~2h | ~$3-4 | Best perf/$, plenty of VRAM |
| **4× RTX 4090 24GB** | ~2.5h | ~$3.5 | Cheapest if 5090 unavailable |
| 1× A100 80GB | ~4h | ~$5-8 | Single-GPU simplicity |
| 1× H100 80GB | ~3h | ~$8-12 | Speed-critical (overkill for pilot) |

PCIe 4.0 x16 strongly recommended for multi-GPU. Avoid x4/x8 hosts (DDP gradient sync becomes the bottleneck).

#### 5. Bootstrap

```bash
git clone <your-private-repo> asmr-pilot && cd asmr-pilot
export HF_TOKEN="hf_..."
export WANDB_API_KEY="..."         # optional, for loss curves
bash scripts/setup_vast.sh         # auto-detects GPU count → generates accelerate config
```

Cache the dataset to local disk (first run takes 1-3 min):

```bash
python -c "from datasets import load_dataset; load_dataset('Raymxnd/asmr-pilot-50h')"
```

#### 6. Smoke test (~5 min, ~$0.10-0.20)

Validate data + all 3 baselines + training fwd/bwd:

```bash
bash scripts/run_baselines.sh --max-samples 20  # parallel baselines on 20 samples
accelerate launch train/train_lora.py --max-steps 10 --no-eval
```

#### 7. Full baselines

```bash
bash scripts/run_baselines.sh
```

Auto-parallelizes across GPUs:
- **3+ GPUs**: each baseline runs on its own GPU (parallel) — ~30 min wall time
- **2 GPUs**: A+B parallel, then B' alone — ~50 min
- **1 GPU**: sequential — ~75 min

Outputs `out/{zeroshot,cascade_generic,cascade_anime}_results.json` with BLEU + chrF + BERTScore.

#### 8. LoRA training (DDP)

```bash
accelerate launch train/train_lora.py
```

`accelerate` auto-uses the config generated by `setup_vast.sh`.

**⚠ Effective batch size scaling**: with N GPUs DDP, effective batch = `batch_size × grad_accum × N`.
Default config has `batch_size=4, grad_accum=4` → effective bs=16 (single GPU). With 4 GPUs, this becomes 64.
Two options to compensate:

```yaml
# Option 1: keep effective bs=16 (most conservative — preserves convergence behavior)
train:
  grad_accum_steps: 1   # was 4

# Option 2: keep grad_accum=4, scale LR (faster convergence, higher throughput)
train:
  lr: 4.0e-4   # was 2e-4 (linear scaling: × N=2 for sqrt(4))
```

For a first pilot run, option 1 is safer. Wall time per setup:

| GPUs | Steps/epoch | Total steps (3 epochs) | Wall time |
|---|---|---|---|
| 1 | 2090 | ~6300 | ~5h |
| 2 | 1045 | ~3140 | ~2.7h |
| 4 | 522 | ~1570 | ~1.4h |

Saves best checkpoint by val chrF to `out/seamless-lora-pilot/best/`. Logs to wandb if configured.

#### 9. Evaluate fine-tuned model (~20 min)

```bash
python train/eval_e2e.py --ckpt out/seamless-lora-pilot/best \
                          --output out/ft_results.json
```

#### 10. Compare (manual review)

All four result JSONs have a `samples` field with per-example refs +
hypotheses for spot-checking. Decision tree:

| Outcome | Next step |
|---|---|
| FT chrF ≥ max(zero-shot, cascade-generic, cascade-anime) + 5 | Scale to 3,000h |
| FT > zero-shot + cascade-generic, but loses to cascade-anime | LoRA helps, but tuned cascade is stronger — useful insight |
| FT chrF improves marginally over zero-shot | Try wider LoRA, more epochs, full FT decoder |
| No improvement | Reconsider — VTT translations may be too noisy for direct supervision |

---

## Key design decisions

- **Split BY work_id** — same speaker/circle never appears in both train and
  test, so eval metrics measure generalization, not memorization.
- **Diversity filter** — at most 1 work per series, 2 per circle, 2 per CV.
  Prevents acoustic homogeneity from same-studio recordings.
- **Drop annotation-only VTT lines** (e.g. `[喘息]`) — these are subtitle
  cues, not translatable speech.
- **chrF over BLEU** as primary metric — VTT translations are heavily
  localized (not literal), so BLEU under-rewards good translations.
- **HF Hub private dataset** for sync — free, robust, resumable. rsync
  fallback if upload becomes flaky.

---

## Files NOT in repo

- `ASMR-Data/` — raw audio, VTT, Arrow shards (15-30 GB)
- `out/` — checkpoints, eval JSONs (regenerate per run)
- `vad_model/model.onnx` — re-download from
  `TransWithAI/Whisper-Vad-EncDec-ASMR-onnx` if needed
- HF token, wandb key — set as env vars

See `.gitignore` for full list.
