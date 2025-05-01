#!/bin/bash
set -e

echo "Starting LLaMA 3.1 8B medical QA fine-tuning..."

# Check for CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA drivers not detected. Please ensure CUDA is properly installed."
    exit 1
fi

# Create required directories
mkdir -p configs
mkdir -p logs
mkdir -p outputs/medical-llama-3.1-8B

# Generate DeepSpeed config if it doesn't exist
if [ ! -f "configs/deepspeed_config.json" ]; then
    echo "Generating DeepSpeed config..."
    python -c "from src.training.config import save_deepspeed_config; save_deepspeed_config('configs/deepspeed_config.json')"
fi

# Generate training config if it doesn't exist
if [ ! -f "configs/train_config.json" ]; then
    echo "Generating training configuration..."
    python -c "from src.training.config import Config; Config().save('configs/train_config.json')"
    echo "Generated default training config. Please check and modify configs/train_config.json if needed."
    echo "You can press Ctrl+C now to edit the config before proceeding, or continue with default settings."
    sleep 10
fi

# Check if dataset exists
if [ ! -f "data/processed/merged_medical_instructions.jsonl" ]; then
    echo "Dataset not found. Please run 'bash scripts/download_data.sh' first."
    exit 1
fi

# Set environment variables for optimal performance
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export OMP_NUM_THREADS=8

# Set the master address for distributed training
MASTER_ADDR=$(hostname -i)
echo "Setting MASTER_ADDR to $MASTER_ADDR"
export MASTER_ADDR

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Set up logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_${TIMESTAMP}.log"

echo "Starting training..."
echo "Log file: $LOG_FILE"
echo "Training output directory: outputs/medical-llama-3.1-8B"

# Run training
python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    src/training/train.py \
    --config=configs/train_config.json \
    $RESUME_ARG \
    $WANDB_ARG \
    2>&1 | tee $LOG_FILE

echo "Training completed!"
echo "Model outputs are saved in outputs/medical-llama-3.1-8B"
echo "Next steps:"
echo "1. Run evaluation: bash scripts/evaluate.sh"
echo "2. Measure inference latency: python src/inference/latency_test.py"
echo "3. Perform error analysis: python src/evaluation/error_analysis.py"
