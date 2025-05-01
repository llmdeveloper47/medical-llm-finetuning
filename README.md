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

3. **Monitoring Training**:
   - Logs are saved to `logs/` directory
   - Training metrics are viewable in SageMaker Studio
   - (Optional) Weights & Biases integration for advanced experiment tracking

### Weights & Biases Integration

The project supports Weights & Biases (wandb) for experiment tracking. You can configure wandb in several ways:

1. **During setup**: The `setup.sh` script will prompt you to configure wandb
2. **Environment variable**: Set `WANDB_API_KEY` in your environment
3. **Command line**: Pass wandb parameters to the training script:

```bash
bash scripts/train.sh --wandb_project "my-project" --wandb_entity "my-username" --wandb_run_name "experiment-1"
```

Wandb will track:
- Training and validation metrics
- Model checkpoints
- System resources (GPU, CPU, memory)
- Configuration parameters
