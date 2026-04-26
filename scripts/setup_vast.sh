#!/usr/bin/env bash
# Setup script for vast.ai H100 instance.
# Run after `git clone <repo> && cd asmr-pilot`.
#
# Usage:
#   bash scripts/setup_vast.sh

set -euo pipefail

echo "=== ASMR Pilot — vast.ai setup ==="

# Default Python on vast.ai PyTorch images is /usr/bin/python3 (3.10/3.11)
PYTHON="${PYTHON:-python3}"
echo "Python: $($PYTHON --version)"

# 1. Install training requirements
echo ""
echo "--- Installing requirements ---"
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r train/requirements-train.txt

# 2. Verify CUDA is visible to torch
echo ""
echo "--- CUDA check ---"
$PYTHON -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, devices={torch.cuda.device_count()}')"
$PYTHON -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'NO GPU')"

# 3. HuggingFace login (interactive — needs HF_TOKEN env or manual paste)
echo ""
echo "--- HuggingFace login ---"
if [ -n "${HF_TOKEN:-}" ]; then
    echo "Using HF_TOKEN from env"
    $PYTHON -c "from huggingface_hub import login; login('${HF_TOKEN}')"
else
    echo "HF_TOKEN env var not set."
    echo "Run: huggingface-cli login   (paste your token)"
fi

# 4. (optional) wandb login
echo ""
echo "--- wandb login (optional) ---"
if [ -n "${WANDB_API_KEY:-}" ]; then
    $PYTHON -c "import wandb; wandb.login(key='${WANDB_API_KEY}')"
else
    echo "WANDB_API_KEY env var not set. Set or run: wandb login"
fi

# 5. Show env summary
echo ""
echo "--- Environment summary ---"
$PYTHON -c "
import sys, torch, transformers, peft, datasets
print(f'Python:       {sys.version.split()[0]}')
print(f'torch:        {torch.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'peft:         {peft.__version__}')
print(f'datasets:     {datasets.__version__}')
print(f'CUDA:         {torch.version.cuda} (available: {torch.cuda.is_available()})')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Sync dataset:  python -c \"from datasets import load_dataset; load_dataset('YOUR_REPO/asmr-pilot-50h', token='YOUR_TOKEN')\""
echo "  2. Smoke test:    python train/eval_e2e.py --zero-shot --max-samples 20 --output out/smoke.json"
echo "  3. Train:         accelerate launch train/train_lora.py"
