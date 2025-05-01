# Medical LLM Fine-tuning on AWS SageMaker

This repository contains the complete pipeline for fine-tuning LLaMA 3.1 8B on a medical question-answering dataset using AWS SageMaker with ml.p4d.24xlarge instances (8x A100 GPUs).

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [Training Pipeline](#training-pipeline)
- [Experiment Tracking with Weights & Biases](#experiment-tracking-with-weights--biases)
- [Evaluation](#evaluation)
- [Inference & Latency Testing](#inference--latency-testing)
- [Error Analysis](#error-analysis)
- [Monitoring & Logging](#monitoring--logging)
- [Troubleshooting](#troubleshooting)

## Overview

This project fine-tunes the LLaMA 3.1 8B model on the MedQA dataset to create a specialized medical question-answering system. The pipeline includes dataset preparation, model fine-tuning using LitGPT, comprehensive evaluation, error analysis, and latency testing.

## Prerequisites

- AWS account with SageMaker access
- IAM role with sufficient permissions for SageMaker and S3
- AWS CLI configured
- Python 3.10+
- Git LFS (for model weights)
- (Optional) Weights & Biases account for experiment tracking

## Environment Setup

1. **Create and start a SageMaker Notebook Instance**:
   - Navigate to the SageMaker console
   - Select "Notebook instances" and click "Create notebook instance"
   - Choose `ml.p4d.24xlarge` as the instance type
   - Attach an IAM role with necessary permissions

2. **Clone the repository and set up environment**:

```bash
git clone https://github.com/yourusername/medical-llm-finetuning.git
cd medical-llm-finetuning
bash scripts/setup.sh
```

3. **Initialize and activate Conda environment**:

After running setup.sh, you'll need to initialize Conda before activating the environment:

```bash
# Initialize Conda for your shell
conda init bash

# Restart your shell or source your shell configuration
source ~/.bashrc

# Now you can activate the environment
conda activate medical_llm
```

4. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## Dataset

## Dataset

We use the **HealthCareMagic-100K** dataset, which contains 100,000 real doctor-patient consultations with detailed patient information including symptoms, age, gender, and medical history, followed by doctor diagnoses and treatment recommendations. This dataset is ideal for training a model to provide treatment plans based on patient information.

### Dataset Download and Preparation

1. **Download and prepare the dataset**:

```bash
bash scripts/download_data.sh
```

2. **Explore the dataset**:
   - Open `notebooks/dataset_exploration.ipynb` to analyze dataset characteristics

3. **Dataset Format**:
   The dataset is converted to the following instruction format:

```json
{
  "instruction": "Given the following patient information and symptoms, provide the most appropriate diagnosis and treatment plan.",
  "input": "Patient: 45-year-old male presenting with the following: \n\nI have been experiencing chest pain, shortness of breath, and sweating for the past 2 hours. I have a history of high blood pressure and diabetes. My blood pressure is currently 160/95 and my heart rate is around 100.",
  "output": "Diagnosis: Based on your symptoms, you are likely experiencing an ST-elevation myocardial infarction (STEMI), commonly known as a heart attack. The chest pain, shortness of breath, and sweating (diaphoresis) along with your risk factors of hypertension and diabetes are highly suggestive of this diagnosis.\n\nTreatment Plan: This is a medical emergency requiring immediate attention. You should call emergency services (911) immediately if you haven't already. Treatment typically includes:\n1. Immediate reperfusion therapy with primary percutaneous coronary intervention (PCI) within 90 minutes\n2. Administration of aspirin 325mg, a P2Y12 inhibitor, and anticoagulation\n3. Beta-blockers if there are no contraindications\n4. Statin therapy\n5. ACE inhibitor for patients with anterior MI, heart failure, or reduced ejection fraction"
}

```

## Training Pipeline

### Model Setup

1. **Download LLaMA 3.1 8B model**:

```bash
python src/data/download_model.py --model_name meta-llama/Meta-Llama-3.1-8B
```

2. **Prepare configuration**:
   - Edit `src/training/config.py` to adjust hyperparameters

### Fine-tuning Process

1. **Launch the fine-tuning job**:

```bash
bash scripts/train.sh
```

This script:
- Sets up distributed training across 8 A100 GPUs
- Uses QLoRA for efficient fine-tuning
- Configures gradient accumulation, batch size, and learning rate
- Enables model checkpointing and logging
- Integrates with Weights & Biases for experiment tracking

2. **Training Parameters**:
   - Learning rate: 2e-5
   - Batch size: 16 per GPU (128 global batch size)
   - Gradient accumulation steps: 4
   - Max sequence length: 4096
   - LoRA rank: 16
   - LoRA alpha: 32
   - Training epochs: 3
   - Optimizer: AdamW with weight decay 0.01
   - LR scheduler: Cosine with warmup (10% of steps)

3. **Advanced Training Options**:

You can pass additional training parameters to customize the process:

```bash
bash scripts/train.sh --resume /path/to/checkpoint --wandb_project "custom-project" --wandb_run_name "experiment-1"
```

## Experiment Tracking with Weights & Biases

This project includes comprehensive integration with Weights & Biases (wandb) for experiment tracking, model versioning, and visualization.

### Setup Options

You can configure wandb in multiple ways:

1. **During initial setup**:
   - The `setup.sh` script will prompt you to configure wandb
   - Enter your API key, project name, and team/entity

2. **Environment variables**:
   - Set variables in your environment before running training:
   ```bash
   export WANDB_API_KEY="your-api-key"
   export WANDB_PROJECT="medical-llm-finetuning"
   export WANDB_ENTITY="your-username-or-team"
   ```

3. **Command line parameters**:
   - Pass parameters directly to the training script:
   ```bash
   bash scripts/train.sh --wandb_project "my-project" --wandb_entity "my-username" --wandb_run_name "experiment-1"
   ```

### Tracked Metrics and Artifacts

Weights & Biases will track:

- **Training metrics**: Loss, learning rate, samples per second
- **Validation metrics**: Validation loss
- **Hardware usage**: GPU memory, utilization
- **Example outputs**: Sample model generations during training
- **Model checkpoints**: Saved as versioned artifacts with metadata
- **System metrics**: CPU, GPU, memory usage

### Viewing Results

Access your experiment results:

1. Log in to your wandb account: https://wandb.ai/
2. Navigate to your project
3. View runs, compare experiments, and analyze performance

### Disabling Wandb

If you don't want to use wandb, you can:
- Skip the wandb configuration in `setup.sh`
- Add `--disable_wandb` flag when running `train.sh`
- Set `WANDB_DISABLED=true` in your environment

## Evaluation

We evaluate the fine-tuned model using various metrics to assess its performance on medical question answering.

1. **Run evaluation**:

```bash
bash scripts/evaluate.sh
```

2. **Evaluation Metrics**:
   - Accuracy: Correctness of answers against reference responses
   - BLEU/ROUGE: Measuring similarity to reference answers
   - Domain-specific metrics: Medical knowledge accuracy, treatment appropriateness
   - Expert evaluation: Optional manual review by medical professionals

3. **Baseline Comparison**:
   - Compare against base LLaMA 3.1 8B model
   - Compare against other medical LLMs (if available)

## Inference & Latency Testing

1. **Measure inference latency**:

```bash
python src/inference/latency_test.py --model_path /path/to/model --batch_sizes 1,4,8,16 --sequence_lengths 512,1024,2048,4096
```

2. **Generate latency report**:
   - Average latency per token
   - Throughput (tokens/second)
   - Memory usage
   - Scaling analysis across batch sizes and sequence lengths

## Error Analysis

1. **Identify error patterns**:

```bash
python src/evaluation/error_analysis.py
```

This script:
- Categorizes errors by type (diagnosis, treatment, understanding)
- Identifies challenging medical cases
- Analyzes performance across different medical specialties
- Generates confusion matrices for multi-class diagnoses

2. **Generate error analysis report** in `reports/error_analysis.pdf`

## Monitoring & Logging

- Training logs are saved to CloudWatch and local `logs/` directory
- Metrics are tracked in SageMaker experiments and/or Weights & Biases
- GPU utilization and memory usage are monitored throughout training

## Troubleshooting

### Common Issues

1. **Conda Environment Problems**:
   - If you see `CondaError: Run 'conda init' before 'conda activate'`, follow the initialization instructions in the [Environment Setup](#environment-setup) section.

2. **Weights & Biases Authentication**:
   - If wandb authentication fails, check your API key and internet connectivity
   - You can manually log in with `wandb login` before running training

3. **Out-of-Memory Errors**:
   - Reduce batch size or gradient accumulation steps
   - Adjust sequence length in `config.py`
   - Enable gradient checkpointing

4. **Dataset Issues**:
   - If the dataset download fails, check the MedQA repository or try using the cached version

5. **Multi-GPU Training**:
   - Ensure NCCL is properly configured
   - Check that DeepSpeed configuration is appropriate for your hardware

For more details, refer to the documentation or open an issue on the GitHub repository.
