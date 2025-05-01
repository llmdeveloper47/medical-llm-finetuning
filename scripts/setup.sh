#!/bin/bash
set -e

echo "Setting up environment for LLaMA 3.1 8B medical fine-tuning..."

# Create conda environment
echo "Creating conda environment..."
conda create -n medical_llm python=3.10 -y
conda activate medical_llm

# Install PyTorch with CUDA 11.8
echo "Installing PyTorch with CUDA support..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Setup Git LFS
echo "Setting up Git LFS..."
apt-get update && apt-get install -y git-lfs
git lfs install

# Configure AWS CLI
echo "Configuring AWS CLI..."
aws configure get region || aws configure

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p logs
mkdir -p outputs/evaluation
mkdir -p outputs/inference

# Make scripts executable
chmod +x scripts/*.sh

# Set up Hugging Face credentials for accessing LLaMA 3.1
echo "Setting up Hugging Face credentials..."
python -c "from huggingface_hub import login; login()"

# Set up WandB for experiment tracking (optional)
echo "Setting up Weights & Biases for experiment tracking (optional)..."
read -p "Would you like to set up WandB for experiment tracking? (y/n) " wandb_setup
if [[ $wandb_setup == "y" || $wandb_setup == "Y" ]]; then
    wandb login
fi

echo "Setting up SageMaker-specific configurations..."

# Verify NVIDIA drivers and CUDA
echo "Verifying NVIDIA drivers and CUDA..."
nvidia-smi
nvcc --version

# Check if EFA is enabled (for distributed training)
echo "Checking EFA configuration..."
if [[ -e /opt/amazon/efa ]]; then
    echo "EFA is installed. Setting up EFA-specific configurations..."
    export FI_PROVIDER="efa"
    export FI_EFA_USE_DEVICE_RDMA=1
    export NCCL_DEBUG=INFO
else
    echo "EFA not detected. Skipping EFA-specific configurations."
fi

# Configure PyTorch distributed training settings
echo "Configuring PyTorch distributed training settings..."
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2

# Set up environment variables for optimal performance
echo "Setting up environment variables for optimal performance..."
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Setup complete!"
echo "You can now run: bash scripts/download_data.sh to prepare the dataset."
