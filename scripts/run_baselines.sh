#!/usr/bin/env bash
# Run all 3 baselines in parallel, one per GPU.
#
# Auto-detects available GPUs:
#   - 3+ GPUs: each baseline on its own GPU (parallel)
#   - 2 GPUs: SeamlessM4T zero-shot + cascade-generic on GPUs, then anime
#   - 1 GPU: all sequential on GPU 0
#
# Each baseline writes to out/{baseline}_results.json
#
# Usage:
#   bash scripts/run_baselines.sh [--max-samples N]

set -euo pipefail

MAX_SAMPLES_ARG=""
if [ "${1:-}" = "--max-samples" ]; then
    MAX_SAMPLES_ARG="--max-samples $2"
    shift 2
fi

# Detect GPU count
NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
echo "Detected ${NGPU} GPU(s)"
echo "Max samples arg: '${MAX_SAMPLES_ARG:-(none)}'"

mkdir -p out logs

if [ "$NGPU" -ge 3 ]; then
    echo "→ Running 3 baselines IN PARALLEL on GPUs 0, 1, 2"
    echo ""

    CUDA_VISIBLE_DEVICES=0 python3 -u train/eval_e2e.py --zero-shot \
        $MAX_SAMPLES_ARG \
        --output out/zeroshot_results.json \
        > logs/zeroshot.log 2>&1 &
    PID_A=$!
    echo "  GPU 0: SeamlessM4T zero-shot     PID=$PID_A  log=logs/zeroshot.log"

    CUDA_VISIBLE_DEVICES=1 python3 -u train/eval_cascade.py --variant generic \
        $MAX_SAMPLES_ARG \
        --output out/cascade_generic_results.json \
        > logs/cascade_generic.log 2>&1 &
    PID_B=$!
    echo "  GPU 1: Whisper-v3 + NLLB         PID=$PID_B  log=logs/cascade_generic.log"

    CUDA_VISIBLE_DEVICES=2 python3 -u train/eval_cascade.py --variant anime \
        $MAX_SAMPLES_ARG \
        --output out/cascade_anime_results.json \
        > logs/cascade_anime.log 2>&1 &
    PID_C=$!
    echo "  GPU 2: Anime-Whisper + Qwen-7B   PID=$PID_C  log=logs/cascade_anime.log"

    echo ""
    echo "Waiting for all 3 baselines to finish..."
    wait $PID_A && echo "  ✓ A (zero-shot) done"     || echo "  ✗ A failed"
    wait $PID_B && echo "  ✓ B (generic) done"       || echo "  ✗ B failed"
    wait $PID_C && echo "  ✓ B' (anime) done"        || echo "  ✗ B' failed"

elif [ "$NGPU" -eq 2 ]; then
    echo "→ 2 GPUs: running A+B in parallel, then B' alone"
    echo ""
    CUDA_VISIBLE_DEVICES=0 python3 -u train/eval_e2e.py --zero-shot $MAX_SAMPLES_ARG \
        --output out/zeroshot_results.json > logs/zeroshot.log 2>&1 &
    PID_A=$!
    CUDA_VISIBLE_DEVICES=1 python3 -u train/eval_cascade.py --variant generic $MAX_SAMPLES_ARG \
        --output out/cascade_generic_results.json > logs/cascade_generic.log 2>&1 &
    PID_B=$!
    wait $PID_A $PID_B
    echo "  ✓ A and B done; running B' on GPU 0..."

    CUDA_VISIBLE_DEVICES=0 python3 -u train/eval_cascade.py --variant anime $MAX_SAMPLES_ARG \
        --output out/cascade_anime_results.json > logs/cascade_anime.log 2>&1
    echo "  ✓ B' done"

else
    echo "→ 1 GPU: running all 3 baselines sequentially on GPU 0"
    echo ""
    python3 -u train/eval_e2e.py --zero-shot $MAX_SAMPLES_ARG \
        --output out/zeroshot_results.json | tee logs/zeroshot.log
    python3 -u train/eval_cascade.py --variant generic $MAX_SAMPLES_ARG \
        --output out/cascade_generic_results.json | tee logs/cascade_generic.log
    python3 -u train/eval_cascade.py --variant anime $MAX_SAMPLES_ARG \
        --output out/cascade_anime_results.json | tee logs/cascade_anime.log
fi

echo ""
echo "==============================================="
echo "  ALL BASELINES COMPLETE"
echo "==============================================="
echo "Results:"
ls -la out/*.json 2>/dev/null

# Summary table
python3 -c "
import json
from pathlib import Path
files = {
    'A. SeamlessM4T zero-shot': 'out/zeroshot_results.json',
    'B. Whisper-v3 + NLLB':     'out/cascade_generic_results.json',
    \"B'. Anime-Whisper + Qwen\": 'out/cascade_anime_results.json',
}
print()
print(f'{\"baseline\":<32s} {\"BLEU\":>8s} {\"chrF\":>8s} {\"BERTScore\":>11s}')
print('-' * 65)
for name, path in files.items():
    if Path(path).exists():
        m = json.load(open(path))['metrics']
        print(f'{name:<32s} {m.get(\"bleu\",\"-\"):>8} {m.get(\"chrf\",\"-\"):>8} {m.get(\"bertscore_f1\",\"-\"):>11}')
"
