#!/bin/bash

#SBATCH --output=logs/%x-%j_%A_%a.out
#SBATCH --error=logs/%x-%j_%A_%a.err
#SBATCH --mail-type=ALL

#SBATCH --job-name=training
#SBATCH --time=24:00:00
#SBATCH --mem=450G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2

module purge
module load miniconda
conda activate ag

# # Debug: Check what's actually installed in circuit environment
# echo "=== ENVIRONMENT DEBUG ==="
# echo "Python path: $(which python)"
# echo "Python executable: $(python -c 'import sys; print(sys.executable)')"
# echo "Checking installed packages..."
# python -m pip list | grep -E "(torch|numpy|sklearn|geometric)" || echo "No ML packages found"

# # Test imports one by one
# echo "Testing individual imports..."
# python -c "import torch; print('torch OK')" || echo "torch FAILED"
# python -c "import numpy; print('numpy OK')" || echo "numpy FAILED" 
# python -c "import sklearn; print('sklearn OK')" || echo "sklearn FAILED"

# # Try installing requirements if torch fails
# if ! python -c "import torch" 2>/dev/null; then
#     echo "torch not found, installing requirements..."
#     pip install -r requirements.txt
# fi

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TOKENIZERS_PARALLELISM=false

echo "=== STARTING TRAINING ==="
# Run training with the local dataset - small model, reasonable resources
python train_attribution_graphs.py \
	--data-dir /home/sk2959/scratch_pi_rp476/sk2959/ag_data \
	--cache-dir /home/sk2959/scratch_pi_rp476/sk2959/
	--max-files 900 \
	--epochs 250 \
	--gpu
