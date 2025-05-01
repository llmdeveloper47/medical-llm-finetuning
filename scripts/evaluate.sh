#!/bin/bash
set -e

echo "Starting evaluation of fine-tuned LLaMA 3.1 8B on medical QA..."

# Create required directories
mkdir -p outputs/evaluation
mkdir -p outputs/error_analysis
mkdir -p outputs/latency

# Check for the latest model checkpoint
if [ -d "outputs/medical-llama-3.1-8B/final" ]; then
    MODEL_PATH="outputs/medical-llama-3.1-8B/final"
    echo "Using final model checkpoint from: $MODEL_PATH"
elif [ -d "outputs/medical-llama-3.1-8B" ]; then
    # Find the latest checkpoint
    LATEST_CHECKPOINT=$(find outputs/medical-llama-3.1-8B -maxdepth 1 -name "checkpoint-*" -type d | sort -V | tail -n 1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        MODEL_PATH=$LATEST_CHECKPOINT
        echo "Using latest checkpoint from: $MODEL_PATH"
    else
        echo "No checkpoint found in outputs/medical-llama-3.1-8B."
        echo "Please specify the model path manually with --model_path argument."
        exit 1
    fi
else
    echo "No model output directory found."
    echo "Please specify the model path manually with --model_path argument."
    exit 1
fi

# Parse command line arguments
MODEL_PATH_ARG=""
COMPARE_BASELINE=""
NUM_SAMPLES="100"  # Default to 100 samples for faster evaluation

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model_path)
            MODEL_PATH="$2"
            MODEL_PATH_ARG="--model_path $MODEL_PATH"
            shift
            shift
            ;;
        --compare_baseline)
            COMPARE_BASELINE="--compare_baseline"
            shift
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Set up logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/evaluate_${TIMESTAMP}.log"
mkdir -p logs

echo "Model path: $MODEL_PATH"
echo "Log file: $LOG_FILE"

# Run the evaluation
echo "Running model evaluation..."
python src/evaluation/evaluate.py \
    $MODEL_PATH_ARG \
    --data_path data/processed/test.jsonl \
    --output_dir outputs/evaluation \
    --num_samples $NUM_SAMPLES \
    $COMPARE_BASELINE \
    2>&1 | tee $LOG_FILE

# Run error analysis if evaluation was successful
if [ $? -eq 0 ]; then
    echo "Running error analysis..."
    python src/evaluation/error_analysis.py \
        --predictions_file outputs/evaluation/predictions.jsonl \
        --output_dir outputs/error_analysis \
        2>&1 | tee -a $LOG_FILE
    
    # Run latency tests
    echo "Running latency tests..."
    python src/inference/latency_test.py \
        $MODEL_PATH_ARG \
        --output_dir outputs/latency \
        --batch_sizes 1,4,8 \
        --sequence_lengths 512,1024,2048 \
        --num_iterations 5 \
        2>&1 | tee -a $LOG_FILE
fi

echo "Evaluation completed!"
echo "Results are available in:"
echo "- Model evaluation: outputs/evaluation/"
echo "- Error analysis: outputs/error_analysis/"
echo "- Latency measurements: outputs/latency/"
