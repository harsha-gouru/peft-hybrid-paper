#!/bin/bash
# Setup script for Prime Intellect pods
# Usage: bash setup_pod.sh [qwen|jamba|nemotron]

set -e
MODEL=${1:-qwen}
echo "=== Setting up pod for $MODEL ==="

# Base deps (all models)
pip3 install --quiet peft transformers accelerate datasets scipy lm-eval 2>&1 | tail -3

# Qwen3.5 needs transformers >= 5.x (for qwen3_5 model type)
pip3 install --quiet "transformers>=5.0" 2>&1 | tail -1

# Nemotron needs mamba-ssm (requires CUDA toolkit)
if [ "$MODEL" = "nemotron" ]; then
    echo "Installing CUDA toolkit for mamba-ssm..."
    sudo apt-get update -qq && sudo apt-get install -y -qq cuda-toolkit-12-6 2>&1 | tail -2
    export PATH=/usr/local/cuda/bin:$PATH
    pip3 install mamba-ssm causal-conv1d 2>&1 | tail -3
fi

# Jamba works without mamba-ssm (use_mamba_kernels=False in config)
# No extra deps needed

# Verify
echo "=== Verifying ==="
python3 -c "
import torch, transformers, peft, datasets
print(f'torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__}')
print(f'CUDA: {torch.cuda.is_available()} GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')
"

echo "=== Setup complete for $MODEL ==="
