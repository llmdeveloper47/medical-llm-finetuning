# Medical LLM Fine-tuning on AWS SageMaker

This repository contains the complete pipeline for fine-tuning LLaMA 3.1 8B on a medical question-answering dataset using AWS SageMaker with ml.p4d.24xlarge instances (8x A100 GPUs).

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [Training Pipeline](#training-pipeline)
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

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## Dataset

We use the **MedQA** dataset, which contains medical exam questions with detailed patient cases, including symptoms, age, medical history, and lab results. This dataset is ideal for training a model to provide treatment recommendations based on patient information.

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
  "instruction": "Given the following patient case, provide the most appropriate diagnosis and treatment plan.",
  "input": "A 45-year-old male presents with chest pain, shortness of breath, and diaphoresis for the past 2 hours. He has a history of hypertension and diabetes. His vital signs show BP 160/95, HR 100, RR 20. ECG shows ST elevation in leads II, III, and aVF.",
  "output": "Diagnosis: ST-elevation myocardial infarction (STEMI) of the inferior wall. Treatment: 1. Immediate reperfusion therapy with primary percutaneous coronary intervention (PCI) within 90 minutes. 2. Administer aspirin 325mg, P2Y12 inhibitor, and anticoagulation. 3. Beta-blocker if no contraindications. 4. Statin therapy. 5. ACE inhibitor for patients with anterior MI, heart failure, or ejection fraction < 40%."
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

3. **Monitor Training**:
   - Logs are saved to `logs/` directory
   - Training metrics are viewable in SageMaker Studio

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

- Training logs are saved to CloudWatch
- Model metrics are tracked in SageMaker experiments
- Weights & Biases integration for visualization (optional)

## Troubleshooting

See `docs/troubleshooting.md` for common issues and solutions, including:
- CUDA out-of-memory errors
- Distributed training issues
- SageMaker permission problems
- Monitoring GPU utilization
