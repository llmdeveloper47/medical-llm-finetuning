#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for fine-tuning LLaMA 3.1 8B on medical QA dataset using LitGPT.
"""

import argparse
import os
import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
import transformers
from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy
from litgpt import GPT, Config as LitConfig
from litgpt.lora import GPTLora, lora_filter
from litgpt.utils import lazy_load, check_valid_checkpoint_dir, quantization

# Add project root to path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import MedicalInstructionDataset
from src.training.config import Config, save_deepspeed_config
from src.utils.aws_utils import upload_to_s3
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


def setup_wandb(config: Config, args: argparse.Namespace):
    """
    Set up Weights & Biases logging.
    
    Args:
        config: Training configuration
        args: Command-line arguments
    
    Returns:
        True if wandb was initialized successfully, False otherwise
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed. Install with 'pip install wandb' to use.")
        return False
    
    # Check if WANDB_API_KEY is set
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set. WandB logging disabled.")
        return False
    
    # Set up default values
    wandb_project = args.wandb_project
    wandb_entity = args.wandb_entity
    wandb_run_name = args.wandb_run_name or f"medical-llama-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Set up config for wandb
    wandb_config = {
        "model": {
            "name": config.model.model_name,
            "hidden_size": config.model.hidden_size,
            "num_hidden_layers": config.model.num_hidden_layers,
            "num_attention_heads": config.model.num_attention_heads,
        },
        "lora": {
            "enabled": config.lora.use_lora,
            "r": config.lora.r,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
            "target_modules": config.lora.target_modules,
        },
        "training": {
            "learning_rate": config.train.learning_rate,
            "weight_decay": config.train.weight_decay,
            "batch_size": config.data.per_device_train_batch_size * torch.cuda.device_count(),
            "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
            "epochs": config.train.num_train_epochs,
            "warmup_ratio": config.train.warmup_ratio,
            "max_seq_length": config.data.max_seq_length,
        },
        "hardware": {
            "num_gpus": torch.cuda.device_count(),
            "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        }
    }
    
    # Initialize wandb
    try:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=wandb_config,
            job_type="training",
            tags=["llama-3.1", "medical", "qlora"],
            notes=f"Fine-tuning LLaMA 3.1 8B on medical QA dataset with QLoRA",
        )
        logger.info(f"WandB initialized: project={wandb_project}, run={wandb_run_name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize WandB: {e}")
        return False


def setup_fabric(config: Config) -> L.Fabric:
    """
    Set up Lightning Fabric for distributed training.
    
    Args:
        config: Training configuration
        
    Returns:
        Configured Lightning Fabric object
    """
    devices = torch.cuda.device_count()
    logger.info(f"Using {devices} devices")
    
    if config.train.deepspeed:
        if not os.path.exists(config.train.deepspeed):
            logger.info(f"DeepSpeed config not found at {config.train.deepspeed}, generating one...")
            os.makedirs(os.path.dirname(config.train.deepspeed), exist_ok=True)
            save_deepspeed_config(config.train.deepspeed)
        
        logger.info(f"Using DeepSpeed strategy with config at {config.train.deepspeed}")
        strategy = DeepSpeedStrategy(
            config=config.train.deepspeed,
            process_group_backend="nccl",
        )
    else:
        logger.info("Using default strategy")
        strategy = "auto"
    
    # Create fabric object
    fabric = L.Fabric(
        accelerator="cuda",
        devices=devices,
        strategy=strategy,
        precision=config.model.precision,
    )
    
    fabric.launch()
    return fabric


def load_tokenizer(config: Config) -> transformers.PreTrainedTokenizer:
    """
    Load tokenizer for the model.
    
    Args:
        config: Training configuration
        
    Returns:
        Loaded tokenizer
    """
    logger.info(f"Loading tokenizer: {config.model.tokenizer_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.tokenizer_name,
        cache_dir=config.model.cache_dir,
        padding_side="right",
        trust_remote_code=config.model.trust_remote_code,
    )
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def load_model(
    fabric: L.Fabric,
    config: Config,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[Union[GPT, GPTLora], Dict]:
    """
    Load the model for training.
    
    Args:
        fabric: Lightning Fabric object
        config: Training configuration
        checkpoint_dir: Optional path to checkpoint directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if checkpoint_dir is not None:
        # Resume from checkpoint
        logger.info(f"Loading model from checkpoint: {checkpoint_dir}")
        check_valid_checkpoint_dir(checkpoint_dir)
        
        # Load config and model weights
        with open(os.path.join(checkpoint_dir, "lit_config.json"), "r") as f:
            lit_config = LitConfig.from_json(f.read())
        checkpoint_path = os.path.join(checkpoint_dir, "lit_model.pth")
        
        # Create model
        model_args = {"config": lit_config}
        
        # Load checkpoint
        checkpoint = lazy_load(checkpoint_path)
        
        # Check if it's a lora checkpoint
        if "model" in checkpoint and "lora_A" in checkpoint:
            logger.info("Loading LoRA checkpoint")
            model = GPTLora(**model_args)
        else:
            logger.info("Loading full model checkpoint")
            model = GPT(**model_args)
    else:
        # Load from HF Hub
        logger.info(f"Loading model from HF Hub: {config.model.model_name}")
        
        # Convert from HF config to LitGPT config
        lit_config = LitConfig.from_huggingface(
            config.model.model_name,
            cache_dir=config.model.cache_dir,
            trust_remote_code=config.model.trust_remote_code,
        )
        
        # Create model
        model_args = {"config": lit_config}
        
        if config.lora.use_lora:
            logger.info("Using LoRA fine-tuning")
            model = GPTLora(**model_args, r=config.lora.r, alpha=config.lora.alpha, dropout=config.lora.dropout, target_modules=config.lora.target_modules)
            # If continuing from existing LoRA weights
            checkpoint = None
            if config.lora.lora_path:
                logger.info(f"Loading initial LoRA weights from: {config.lora.lora_path}")
                checkpoint = lazy_load(config.lora.lora_path)
        else:
            logger.info("Using full-parameter fine-tuning")
            model = GPT(**model_args)
            checkpoint = None
    
    # Set up precision
    if config.model.precision.startswith("int8"):
        logger.info("Using int8 quantization")
        model = quantization.quantize(model, quantization.WeightOnlyInt8QuantHandler)
    
    # Setup the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        eps=config.train.adam_epsilon,
    )
    
    if config.lora.use_lora:
        # Filter only LoRA parameters for optimization
        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if lora_filter(n, p)],
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            eps=config.train.adam_epsilon,
        )
    
    # Setup model and optimizer with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    
    # If resuming, load the optimizer state as well
    if checkpoint is not None:
        model.load_state_dict(checkpoint)
    
    return model, optimizer


def train(
    fabric: L.Fabric,
    model: Union[GPT, GPTLora],
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    config: Config = Config(),
    tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
    wandb_enabled: bool = False,
) -> None:
    """
    Main training loop.
    
    Args:
        fabric: Lightning Fabric object
        model: Model to train
        optimizer: Optimizer
        train_dataloader: Training dataloader
        val_dataloader: Optional validation dataloader
        config: Training configuration
        tokenizer: Tokenizer for decoding examples
        wandb_enabled: Whether wandb logging is enabled
    """
    # Create gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=(config.model.precision == "fp16"))
    
    # Training state
    step_count = 0
    total_t0 = time.time()
    
    # Setup learning rate scheduler
    num_training_steps = len(train_dataloader) * config.train.num_train_epochs // config.train.gradient_accumulation_steps
    warmup_steps = int(config.train.warmup_ratio * num_training_steps)
    
    logger.info(f"Total training steps: {num_training_steps}, warmup steps: {warmup_steps}")
    
    lr_scheduler = transformers.get_scheduler(
        name=config.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Log training parameters
    if wandb_enabled and fabric.global_rank == 0:
        try:
            import wandb
            wandb.log({
                "train/total_steps": num_training_steps,
                "train/warmup_steps": warmup_steps,
                "train/batch_size_per_device": config.data.per_device_train_batch_size,
                "train/global_batch_size": config.data.per_device_train_batch_size * fabric.world_size * config.train.gradient_accumulation_steps,
                "train/num_devices": fabric.world_size,
                "train/gradient_accumulation_steps": config.train.gradient_accumulation_steps,
            })
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")
    
    # Train for specified number of epochs
    for epoch in range(config.train.num_train_epochs):
        model.train()
        total_loss = 0
        epoch_t0 = time.time()
        
        logger.info(f"Starting epoch {epoch+1}/{config.train.num_train_epochs}")
        progress = fabric.progress(train_dataloader) if fabric.global_rank == 0 else train_dataloader
        
        # Training loop
        for batch_idx, batch in enumerate(progress):
            with fabric.no_backward_sync(model, enabled=(batch_idx + 1) % config.train.gradient_accumulation_steps != 0):
                # Forward pass
                input_ids = batch["input_ids"].to(fabric.device)
                attention_mask = batch["attention_mask"].to(fabric.device)
                labels = batch["labels"].to(fabric.device)
                
                # Model forward
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / config.train.gradient_accumulation_steps
                
                # Backward pass
                fabric.backward(loss)
                
                # Update running loss
                total_loss += loss.detach().float()
            
            # Optimize step after accumulation
            if (batch_idx + 1) % config.train.gradient_accumulation_steps == 0:
                # Clip gradients
                if config.train.max_grad_norm > 0:
                    fabric.clip_gradients(model, optimizer, max_norm=config.train.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                step_count += 1
                
                # Logging
                if step_count % config.train.logging_steps == 0 and fabric.global_rank == 0:
                    avg_loss = total_loss / config.train.logging_steps
                    elapsed = time.time() - total_t0
                    samples_per_second = config.train.gradient_accumulation_steps * config.data.per_device_train_batch_size * fabric.world_size / (time.time() - total_t0)
                    
                    fabric.log("train/loss", avg_loss, step=step_count)
                    fabric.log("train/lr", lr_scheduler.get_last_lr()[0], step=step_count)
                    
                    # Log to wandb if available
                    if wandb_enabled:
                        try:
                            import wandb
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/lr": lr_scheduler.get_last_lr()[0],
                                "train/step": step_count,
                                "train/epoch": epoch + 1,
                                "train/global_step": step_count,
                                "train/samples_per_second": samples_per_second,
                                "train/elapsed_time": elapsed,
                            })
                        except Exception as e:
                            logger.warning(f"Failed to log to wandb: {e}")
                    
                    logger.info(
                        f"Epoch: {epoch+1}/{config.train.num_train_epochs} | "
                        f"Step: {step_count}/{num_training_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.7f} | "
                        f"Samples/sec: {samples_per_second:.2f} | "
                        f"Elapsed: {elapsed:.2f}s"
                    )
                    
                    # Example of current outputs if tokenizer provided
                    if tokenizer is not None and batch_idx % (config.train.logging_steps * 10) == 0:
                        # Sample one item from the batch
                        idx = 0
                        
                        # Get input and predicted output
                        input_text = tokenizer.decode(input_ids[idx], skip_special_tokens=False)
                        logger.info(f"Sample input: {input_text[:100]}...")
                        
                        # Generate a short sample if small enough model
                        if hasattr(model, "generate") and config.model.hidden_size <= 4096:
                            with torch.no_grad():
                                sample_output = model.generate(
                                    input_ids[:1], max_new_tokens=100, temperature=0.7, top_p=0.9
                                )
                                sample_output_text = tokenizer.decode(sample_output[0], skip_special_tokens=False)
                                logger.info(f"Sample generation: {sample_output_text}")
                                
                                # Log sample to wandb
                                if wandb_enabled:
                                    try:
                                        import wandb
                                        wandb.log({
                                            "examples/input": wandb.Html(input_text),
                                            "examples/output": wandb.Html(sample_output_text),
                                        })
                                    except Exception as e:
                                        logger.warning(f"Failed to log examples to wandb: {e}")
                    
                    total_loss = 0.0
                
                # Evaluation
                if val_dataloader is not None and step_count % config.train.eval_steps == 0:
                    val_loss = evaluate(fabric, model, val_dataloader, config, wandb_enabled, step_count)
                    fabric.log("val/loss", val_loss, step=step_count)
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    model.train()
                
                # Save checkpoint
                if step_count % config.train.save_steps == 0 and fabric.global_rank == 0:
                    save_checkpoint(
                        fabric, model, optimizer, lr_scheduler, step_count, config, wandb_enabled=wandb_enabled
                    )
        
        # End of epoch
        epoch_t1 = time.time()
        epoch_time = epoch_t1 - epoch_t0
        logger.info(f"Epoch {epoch+1} finished in {epoch_time:.2f} seconds")
        
        # Log epoch stats to wandb
        if wandb_enabled and fabric.global_rank == 0:
            try:
                import wandb
                wandb.log({
                    "train/epoch": epoch + 1,
                    "train/epoch_time": epoch_time,
                })
            except Exception as e:
                logger.warning(f"Failed to log epoch stats to wandb: {e}")
        
        # Save checkpoint at end of epoch
        if fabric.global_rank == 0:
            save_checkpoint(
                fabric, model, optimizer, lr_scheduler, step_count, config, 
                is_final=(epoch == config.train.num_train_epochs - 1),
                wandb_enabled=wandb_enabled
            )
    
    # Final evaluation
    if val_dataloader is not None and fabric.global_rank == 0:
        val_loss = evaluate(fabric, model, val_dataloader, config, wandb_enabled, step_count)
        fabric.log("val/loss", val_loss, step=step_count)
        logger.info(f"Final validation loss: {val_loss:.4f}")
    
    logger.info(f"Training completed in {time.time() - total_t0:.2f} seconds")
    
    # Close wandb
    if wandb_enabled and fabric.global_rank == 0:
        try:
            import wandb
            wandb.finish()
        except Exception as e:
            logger.warning(f"Failed to close wandb: {e}")


def evaluate(
    fabric: L.Fabric,
    model: Union[GPT, GPTLora],
    val_dataloader: torch.utils.data.DataLoader,
    config: Config,
    wandb_enabled: bool = False,
    step_count: int = 0,
) -> float:
    """
    Evaluate the model on validation data.
    
    Args:
        fabric: Lightning Fabric object
        model: Model to evaluate
        val_dataloader: Validation dataloader
        config: Training configuration
        wandb_enabled: Whether wandb logging is enabled
        step_count: Current training step
        
    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in fabric.progress(val_dataloader):
            # Forward pass
            input_ids = batch["input_ids"].to(fabric.device)
            attention_mask = batch["attention_mask"].to(fabric.device)
            labels = batch["labels"].to(fabric.device)
            
            # Model forward
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            val_loss += loss.detach().float()
    
    # Average loss
    val_loss /= len(val_dataloader)
    
    # Log to wandb
    if wandb_enabled and fabric.global_rank == 0:
        try:
            import wandb
            wandb.log({
                "val/loss": val_loss,
                "val/step": step_count,
            })
        except Exception as e:
            logger.warning(f"Failed to log validation loss to wandb: {e}")
    
    return val_loss


def save_checkpoint(
    fabric: L.Fabric,
    model: Union[GPT, GPTLora],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    step_count: int,
    config: Config,
    is_final: bool = False,
    wandb_enabled: bool = False,
) -> None:
    """
    Save model checkpoint.
    
    Args:
        fabric: Lightning Fabric object
        model: Model to save
        optimizer: Optimizer state
        lr_scheduler: Learning rate scheduler
        step_count: Current training step
        config: Training configuration
        is_final: Whether this is the final checkpoint
        wandb_enabled: Whether wandb logging is enabled
    """
    checkpoint_dir = (
        os.path.join(config.train.output_dir, f"checkpoint-{step_count}")
        if not is_final
        else os.path.join(config.train.output_dir, "final")
    )
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Saving checkpoint to {checkpoint_dir}")
    
    # Save model state
    checkpoint = model.state_dict()
    
    # Save optimizer and scheduler state if requested
    if not config.train.save_only_model:
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
        checkpoint["step"] = step_count
    
    # Save config
    lit_config = model.config
    with open(os.path.join(checkpoint_dir, "lit_config.json"), "w") as f:
        f.write(lit_config.to_json())
    
    # Save checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, "lit_model.pth"))
    
    # Save training config
    config.save(os.path.join(checkpoint_dir, "train_config.json"))
    
    # Upload to S3 if using SageMaker
    if config.train.use_sagemaker and config.train.sagemaker_bucket:
        s3_path = f"s3://{config.train.sagemaker_bucket}/medical-llm-finetuning/checkpoints/checkpoint-{step_count}"
        logger.info(f"Uploading checkpoint to {s3_path}")
        upload_to_s3(checkpoint_dir, s3_path)
    
    # Log to wandb if available
    if wandb_enabled:
        try:
            import wandb
            
            # Create a metadata file for the checkpoint
            metadata = {
                "step": step_count,
                "is_final": is_final,
                "date": datetime.now().isoformat(),
                "config": config.__dict__,
            }
            
            with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Log the checkpoint as an artifact
            if wandb.run is not None:
                artifact = wandb.Artifact(
                    name=f"model-checkpoint-{step_count}",
                    type="model",
                    description=f"Model checkpoint at step {step_count}" + (" (final)" if is_final else ""),
                )
                artifact.add_dir(checkpoint_dir)
                wandb.log_artifact(artifact)
                logger.info(f"Logged checkpoint to wandb: {artifact.name}")
        except Exception as e:
            logger.warning(f"Failed to log checkpoint to wandb: {e}")


def log_gpu_memory_stats(wandb_enabled: bool = False):
    """
    Log GPU memory statistics.
    
    Args:
        wandb_enabled: Whether wandb logging is enabled
    """
    if not torch.cuda.is_available():
        return
    
    try:
        gpu_stats = []
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
            max_memory_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)  # GB
            
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Memory Allocated: {memory_allocated:.2f} GB")
            logger.info(f"  Memory Reserved: {memory_reserved:.2f} GB")
            logger.info(f"  Max Memory Allocated: {max_memory_allocated:.2f} GB")
            
            gpu_stats.append({
                "device": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
                "max_memory_allocated_gb": max_memory_allocated,
            })
        
        # Log to wandb
        if wandb_enabled:
            try:
                import wandb
                for stats in gpu_stats:
                    wandb.log({
                        f"gpu/{stats['device']}/memory_allocated_gb": stats["memory_allocated_gb"],
                        f"gpu/{stats['device']}/memory_reserved_gb": stats["memory_reserved_gb"],
                        f"gpu/{stats['device']}/max_memory_allocated_gb": stats["max_memory_allocated_gb"],
                    })
            except Exception as e:
                logger.warning(f"Failed to log GPU stats to wandb: {e}")
    except Exception as e:
        logger.warning(f"Failed to get GPU memory stats: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train LLaMA 3.1 8B on medical QA data")
    parser.add_argument("--config", type=str, default="configs/train_config.json", help="Path to training configuration")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory for resuming training")
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "medical-llm-finetuning"), 
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", None), 
                        help="WandB entity (username or team)")
    parser.add_argument("--wandb_run_name", type=str, default=None, 
                        help="WandB run name (defaults to timestamp if not provided)")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable WandB logging")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        config = Config()
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        config.save(args.config)
    
    # Override resume path if provided
    if args.resume:
        config.train.resume_from_checkpoint = args.resume
    
    # Setup WandB if enabled
    wandb_enabled = not args.disable_wandb and os.environ.get("WANDB_API_KEY") is not None
    if wandb_enabled:
        wandb_enabled = setup_wandb(config, args)
    
    # Setup Lightning Fabric
    fabric = setup_fabric(config)
    
    # Log system information
    if fabric.global_rank == 0:
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Log GPU memory stats
        log_gpu_memory_stats(wandb_enabled)
    
    # Load tokenizer
    tokenizer = load_tokenizer(config)
    
    # Create datasets and dataloaders
    logger.info("Creating datasets and dataloaders")
    train_dataset = MedicalInstructionDataset(
        data_path=config.data.data_dir,
        tokenizer_name_or_path=config.model.tokenizer_name,
        max_length=config.data.max_seq_length,
        split="train",
    )
    
    val_dataset = MedicalInstructionDataset(
        data_path=config.data.data_dir,
        tokenizer_name_or_path=config.model.tokenizer_name,
        max_length=config.data.max_seq_length,
        split="validation",
    )
    
    train_dataloader = train_dataset.get_dataloader(
        batch_size=config.data.per_device_train_batch_size,
        shuffle=True,
    )
    
    val_dataloader = val_dataset.get_dataloader(
        batch_size=config.data.per_device_eval_batch_size,
        shuffle=False,
    )
    
    # Prepare train and validation dataloaders
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    val_dataloader = fabric.setup_dataloaders(val_dataloader)
    
    # Load model and optimizer
    model, optimizer = load_model(
        fabric,
        config,
        checkpoint_dir=config.train.resume_from_checkpoint,
    )
    
    # Train model
    train(
        fabric=fabric,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        tokenizer=tokenizer,
        wandb_enabled=wandb_enabled,
    )
    
    # Final memory stats
    if fabric.global_rank == 0:
        logger.info("Final GPU memory stats:")
        log_gpu_memory_stats(wandb_enabled)


if __name__ == "__main__":
    main()
