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
                    fabric.log("train/loss", avg_loss, step=step_count)
                    fabric.log("train/lr", lr_scheduler.get_last_lr()[0], step=step_count)
                    
                    logger.info(
                        f"Epoch: {epoch+1}/{config.train.num_train_epochs} | "
                        f"Step: {step_count}/{num_training_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.7f} | "
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
                    
                    total_loss = 0.0
                
                # Evaluation
                if val_dataloader is not None and step_count % config.train.eval_steps == 0:
                    val_loss = evaluate(fabric, model, val_dataloader, config)
                    fabric.log("val/loss", val_loss, step=step_count)
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    model.train()
                
                # Save checkpoint
                if step_count % config.train.save_steps == 0 and fabric.global_rank == 0:
                    save_checkpoint(
                        fabric, model, optimizer, lr_scheduler, step_count, config
                    )
        
        # End of epoch
        epoch_t1 = time.time()
        logger.info(f"Epoch {epoch+1} finished in {epoch_t1 - epoch_t0:.2f} seconds")
        
        # Save checkpoint at end of epoch
        if fabric.global_rank == 0:
            save_checkpoint(
                fabric, model, optimizer, lr_scheduler, step_count, config, is_final=(epoch == config.train.num_train_epochs - 1)
            )
    
    # Final evaluation
    if val_dataloader is not None and fabric.global_rank == 0:
        val_loss = evaluate(fabric, model, val_dataloader, config)
        fabric.log("val/loss", val_loss, step=step_count)
        logger.info(f"Final validation loss: {val_loss:.4f}")
    
    logger.info(f"Training completed in {time.time() - total_t0:.2f} seconds")


def evaluate(
    fabric: L.Fabric,
    model: Union[GPT, GPTLora],
    val_dataloader: torch.utils.data.DataLoader,
    config: Config,
) -> float:
    """
    Evaluate the model on validation data.
    
    Args:
        fabric: Lightning Fabric object
        model: Model to evaluate
        val_dataloader: Validation dataloader
        config: Training configuration
        
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
    
    return val_loss


def save_checkpoint(
    fabric: L.Fabric,
    model: Union[GPT, GPTLora],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    step_count: int,
    config: Config,
    is_final: bool = False,
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


def main():
    parser = argparse.ArgumentParser(description="Train LLaMA 3.1 8B on medical QA data")
    parser.add_argument("--config", type=str, default="configs/train_config.json", help="Path to training configuration")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory for resuming training")
    
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
    
    # Setup Lightning Fabric
    fabric = setup_fabric(config)
    
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
    )


if __name__ == "__main__":
    main()
