#!/usr/bin/env bash
# Full pilot study orchestrator. Idempotent — each phase checks if its
# output already exists before running, so re-runs cheap.
#
# Usage:
#   bash scripts/run_full_pilot.sh
#
# Output structure:
#   out/
#     phase1_baselines/
#       zeroshot_results.json
#       cascade_generic_results.json
#       cascade_anime_results.json
#     phase1.5_sakura/
#       cascade_sakura_results.json     (B'')
#     phase2a_qwen_lora/best/           (or sakura_lora)
#     phase2b_sm4t_lora/best/
#     phase3_ft_eval/
#       ft_cascade_results.json
#       ft_e2e_results.json
#     phase4_compare.md                 (final report)
#     STATUS.md                          (live progress)

set -uo pipefail   # don't 'e' — we want to continue past failures

cd "$(dirname "$0")/.."   # repo root

REPO_ROOT="$(pwd)"
LOG_DIR="$REPO_ROOT/logs"
OUT_DIR="$REPO_ROOT/out"
mkdir -p "$LOG_DIR" "$OUT_DIR"

# -------- Helpers --------------------------------------------------

NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')

phase_status() {
    local phase=$1
    local status=$2
    local detail="${3:-}"
    local ts=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    echo "[$ts] $phase: $status $detail" | tee -a "$OUT_DIR/STATUS.md"
}

run_or_skip() {
    local sentinel=$1
    local label=$2
    shift 2
    if [ -f "$sentinel" ]; then
        phase_status "$label" "SKIP (sentinel: $sentinel)"
        return 0
    fi
    phase_status "$label" "START"
    "$@"
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        touch "$sentinel"
        phase_status "$label" "OK"
    else
        phase_status "$label" "FAILED" "(exit $rc)"
    fi
    return $rc
}

# -------- Phase 1: 3 baselines + ja_pseudo (parallel on 4 GPUs) ----

phase1() {
    mkdir -p "$OUT_DIR/phase1_baselines"

    # A: SeamlessM4T zero-shot
    if [ ! -f "$OUT_DIR/phase1_baselines/zeroshot_results.json" ]; then
        CUDA_VISIBLE_DEVICES=0 nohup python3 -u train/eval_e2e.py --zero-shot \
            --output "$OUT_DIR/phase1_baselines/zeroshot_results.json" \
            > "$LOG_DIR/p1_zeroshot.log" 2>&1 &
        local pid_a=$!
    else
        local pid_a=""
    fi

    # B: Whisper-v3 + NLLB
    if [ ! -f "$OUT_DIR/phase1_baselines/cascade_generic_results.json" ]; then
        CUDA_VISIBLE_DEVICES=1 nohup python3 -u train/eval_cascade.py --variant generic \
            --output "$OUT_DIR/phase1_baselines/cascade_generic_results.json" \
            > "$LOG_DIR/p1_cascade_generic.log" 2>&1 &
        local pid_b=$!
    else
        local pid_b=""
    fi

    # B': Anime + Qwen
    if [ ! -f "$OUT_DIR/phase1_baselines/cascade_anime_results.json" ]; then
        CUDA_VISIBLE_DEVICES=2 nohup python3 -u train/eval_cascade.py --variant anime \
            --output "$OUT_DIR/phase1_baselines/cascade_anime_results.json" \
            > "$LOG_DIR/p1_cascade_anime.log" 2>&1 &
        local pid_bp=$!
    else
        local pid_bp=""
    fi

    # ja_pseudo for training set
    local jsonl_path="${HF_DATA_ROOT:-/workspace/ASMR-Data/hf_cache}/train_with_ja.jsonl"
    if [ ! -f "$jsonl_path" ]; then
        CUDA_VISIBLE_DEVICES=3 nohup python3 -u train/generate_ja_pseudo.py --split train \
            > "$LOG_DIR/p1_ja_pseudo.log" 2>&1 &
        local pid_ja=$!
    else
        local pid_ja=""
    fi

    [ -n "$pid_a" ] && wait $pid_a; phase_status "Phase1.A" "DONE"
    [ -n "$pid_b" ] && wait $pid_b; phase_status "Phase1.B" "DONE"
    [ -n "$pid_bp" ] && wait $pid_bp; phase_status "Phase1.B'" "DONE"
    [ -n "$pid_ja" ] && wait $pid_ja; phase_status "Phase1.ja_pseudo" "DONE"
}

# -------- Phase 1.5: Sakura B'' baseline ---------------------------

phase15_sakura() {
    mkdir -p "$OUT_DIR/phase1.5_sakura"
    if [ -f "$OUT_DIR/phase1.5_sakura/cascade_sakura_results.json" ]; then
        phase_status "Phase1.5" "SKIP"
        return 0
    fi

    # Use AWQ-quantized Sakura on a single GPU
    # Override config via env var; actual eval_cascade.py uses anime variant
    # but we override the mt_model via Python -c
    CUDA_VISIBLE_DEVICES=0 python3 -u train/eval_cascade.py \
        --variant anime \
        --output "$OUT_DIR/phase1.5_sakura/cascade_sakura_results.json" \
        --use-sakura \
        > "$LOG_DIR/p15_sakura.log" 2>&1
}

# -------- Phase 2a: QLoRA on Sakura --------------------------------

phase2a_sakura_lora() {
    if [ -d "$OUT_DIR/phase2a_sakura_lora/best" ]; then
        phase_status "Phase2a" "SKIP"
        return 0
    fi
    accelerate launch --num_processes=$NGPU train/train_qwen_lora.py \
        --output-dir "$OUT_DIR/phase2a_sakura_lora" \
        --use-sakura \
        > "$LOG_DIR/p2a_sakura_lora.log" 2>&1
}

# -------- Phase 2b: LoRA on SeamlessM4T ----------------------------

phase2b_sm4t_lora() {
    if [ -d "$OUT_DIR/phase2b_sm4t_lora/best" ]; then
        phase_status "Phase2b" "SKIP"
        return 0
    fi
    accelerate launch --num_processes=$NGPU train/train_lora.py \
        --config train/config.yaml \
        > "$LOG_DIR/p2b_sm4t_lora.log" 2>&1

    # train_lora.py defaults output to out/seamless-lora-pilot;
    # symlink so downstream finds it
    if [ -d "out/seamless-lora-pilot/best" ] && [ ! -d "$OUT_DIR/phase2b_sm4t_lora" ]; then
        mkdir -p "$OUT_DIR/phase2b_sm4t_lora"
        ln -s "$REPO_ROOT/out/seamless-lora-pilot/best" "$OUT_DIR/phase2b_sm4t_lora/best"
    fi
}

# -------- Phase 3: Eval FT models (parallel) -----------------------

phase3_ft_eval() {
    mkdir -p "$OUT_DIR/phase3_ft_eval"
    local pids=()

    if [ -d "$OUT_DIR/phase2a_sakura_lora/best" ] && \
       [ ! -f "$OUT_DIR/phase3_ft_eval/ft_cascade_results.json" ]; then
        CUDA_VISIBLE_DEVICES=0 nohup python3 -u train/eval_cascade.py \
            --variant anime \
            --use-sakura \
            --mt-lora-ckpt "$OUT_DIR/phase2a_sakura_lora/best" \
            --output "$OUT_DIR/phase3_ft_eval/ft_cascade_results.json" \
            > "$LOG_DIR/p3_ft_cascade.log" 2>&1 &
        pids+=($!)
    fi

    if [ -d "$OUT_DIR/phase2b_sm4t_lora/best" ] && \
       [ ! -f "$OUT_DIR/phase3_ft_eval/ft_e2e_results.json" ]; then
        CUDA_VISIBLE_DEVICES=1 nohup python3 -u train/eval_e2e.py \
            --ckpt "$OUT_DIR/phase2b_sm4t_lora/best" \
            --output "$OUT_DIR/phase3_ft_eval/ft_e2e_results.json" \
            > "$LOG_DIR/p3_ft_e2e.log" 2>&1 &
        pids+=($!)
    fi

    for pid in "${pids[@]}"; do
        wait $pid
    done
}

# -------- Phase 4: Aggregate report --------------------------------

phase4_report() {
    python3 -c "
import json, os
from pathlib import Path

base = Path('$OUT_DIR')
results = {
    'A. SeamlessM4T zero-shot':           base / 'phase1_baselines/zeroshot_results.json',
    'B. Whisper-v3 + NLLB':                base / 'phase1_baselines/cascade_generic_results.json',
    \"B'. Anime-Whisper + Qwen\":           base / 'phase1_baselines/cascade_anime_results.json',
    \"B''. Anime-Whisper + Sakura-14B\":    base / 'phase1.5_sakura/cascade_sakura_results.json',
    'C-Cascade. Anime + Sakura+QLoRA':     base / 'phase3_ft_eval/ft_cascade_results.json',
    'C-E2E. SeamlessM4T+LoRA':            base / 'phase3_ft_eval/ft_e2e_results.json',
}

lines = ['# Pilot study results', '', '## 6-way comparison', '']
lines.append(f'| {\"system\":<40} | BLEU | chrF | BERTScore F1 | n |')
lines.append('|' + '-' * 42 + '|------|------|--------------|---|')
for name, path in results.items():
    if not path.exists():
        lines.append(f'| {name:<40} | - | - | - | (missing) |')
        continue
    d = json.load(open(path))
    m = d.get('metrics', {})
    bleu = m.get('bleu', '-')
    chrf = m.get('chrf', '-')
    bs = m.get('bertscore_f1') or '-'
    n = m.get('n_examples', '-')
    lines.append(f'| {name:<40} | {bleu} | {chrf} | {bs} | {n} |')

(base / 'phase4_compare.md').write_text('\n'.join(lines))
print('\n'.join(lines))
"
}

# -------- Main -----------------------------------------------------

phase_status "Pilot" "START" "(NGPU=$NGPU)"

run_or_skip "$OUT_DIR/.phase1_done"  Phase1   phase1
run_or_skip "$OUT_DIR/.phase15_done" Phase1.5 phase15_sakura
run_or_skip "$OUT_DIR/.phase2a_done" Phase2a  phase2a_sakura_lora
run_or_skip "$OUT_DIR/.phase2b_done" Phase2b  phase2b_sm4t_lora
run_or_skip "$OUT_DIR/.phase3_done"  Phase3   phase3_ft_eval
phase4_report | tee "$OUT_DIR/phase4_compare.md"

phase_status "Pilot" "FINISHED"
echo "===== PILOT FINISHED ====="
cat "$OUT_DIR/STATUS.md"
