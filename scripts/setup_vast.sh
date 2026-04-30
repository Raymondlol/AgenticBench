#!/usr/bin/env bash
# Setup script for vast.ai instance (single or multi-GPU).
# Run after `git clone <repo> && cd asmr-pilot`.
#
# Usage:
#   bash scripts/setup_vast.sh

set -euo pipefail

echo "=============================================="
echo "  ASMR Pilot — vast.ai setup"
echo "=============================================="

PYTHON="${PYTHON:-python3}"
echo "Python: $($PYTHON --version)"

# 1. Install training requirements
echo ""
echo "--- Installing requirements ---"
# pip upgrade is best-effort (Debian-managed pip can't be uninstalled)
$PYTHON -m pip install --upgrade pip --break-system-packages 2>/dev/null || \
    echo "  (pip upgrade skipped — using system pip)"
$PYTHON -m pip install --break-system-packages -r train/requirements-train.txt

# 2. Verify CUDA + count GPUs
echo ""
echo "--- CUDA / GPU check ---"
$PYTHON -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
echo "Detected ${NGPU} GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /'

# 3. Generate accelerate config based on detected GPU count
echo ""
echo "--- Generating accelerate config ---"
ACC_CFG="$HOME/.cache/huggingface/accelerate/default_config.yaml"
mkdir -p "$(dirname $ACC_CFG)"

if [ "$NGPU" -gt 1 ]; then
    cat > "$ACC_CFG" <<EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: ${NGPU}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    echo "Generated DDP config for ${NGPU} GPUs at $ACC_CFG"
else
    cat > "$ACC_CFG" <<EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    echo "Generated single-GPU config at $ACC_CFG"
fi

# 4. HuggingFace login
echo ""
echo "--- HuggingFace login ---"
if [ -n "${HF_TOKEN:-}" ]; then
    echo "Using HF_TOKEN from env"
    $PYTHON -c "from huggingface_hub import login; login('${HF_TOKEN}')"
else
    echo "HF_TOKEN env var not set."
    echo "Run: huggingface-cli login   (paste your token)"
fi

# 5. wandb login (optional)
echo ""
echo "--- wandb login (optional) ---"
if [ -n "${WANDB_API_KEY:-}" ]; then
    $PYTHON -c "import wandb; wandb.login(key='${WANDB_API_KEY}')"
else
    echo "WANDB_API_KEY env var not set."
    echo "If you want loss curves: export WANDB_API_KEY=... or run wandb login"
    echo "Or set in train/config.yaml: report_to: 'none' to disable"
fi

# 6. Summary + recommended commands
echo ""
echo "--- Environment summary ---"
$PYTHON -c "
import sys, torch, transformers, peft, datasets
print(f'Python:       {sys.version.split()[0]}')
print(f'torch:        {torch.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'peft:         {peft.__version__}')
print(f'datasets:     {datasets.__version__}')
print(f'CUDA:         {torch.version.cuda} (devices: {torch.cuda.device_count()})')
"

echo ""
echo "=============================================="
echo "  Setup complete! Recommended workflow:"
echo "=============================================="
echo ""
echo "1. Cache dataset locally (first time only):"
echo "   python -c 'from datasets import load_dataset; load_dataset(\"Raymxnd/asmr-pilot-50h\")'"
echo ""
echo "2. Smoke test (quick validation):"
echo "   bash scripts/run_baselines.sh --max-samples 20"
echo "   accelerate launch train/train_lora.py --max-steps 10 --no-eval"
echo ""
echo "3. Full baselines (parallel across GPUs):"
echo "   bash scripts/run_baselines.sh"
echo ""
if [ "$NGPU" -gt 1 ]; then
echo "4. LoRA training (DDP across ${NGPU} GPUs):"
echo "   accelerate launch train/train_lora.py"
echo ""
echo "   ⚠ With ${NGPU} GPUs, effective batch size = 4 × 4 × ${NGPU} = $((4*4*NGPU))"
echo "      To preserve original effective bs=16, edit train/config.yaml:"
echo "      grad_accum_steps: $((4 / NGPU > 0 ? 4 / NGPU : 1))"
else
echo "4. LoRA training (single GPU):"
echo "   accelerate launch train/train_lora.py"
fi
echo ""
echo "5. Eval fine-tuned model:"
echo "   python train/eval_e2e.py --ckpt out/seamless-lora-pilot/best \\"
echo "       --output out/ft_results.json"
