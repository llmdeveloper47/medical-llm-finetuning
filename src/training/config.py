#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration settings for fine-tuning LLaMA 3.1 8B model on medical QA data.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from transformers import TrainingArguments


@dataclass
class ModelConfig:
    """Model configuration."""
    
    # Base model settings
    model_name: str = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer_name: Optional[str] = None  # Uses model_name if None
    cache_dir: Optional[str] = None
    trust_remote_code: bool = True
    
    # Architecture settings
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None  # Same as num_attention_heads if None
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Precision settings
    precision: str = "bf16"  # Options: "fp32", "fp16", "bf16", "int8"
    
    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name
        
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class LoraConfig:
    """Configuration for LoRA fine-tuning."""
    
    use_lora: bool = True
    r: int = 16  # LoRA rank
    alpha: int = 32  # LoRA alpha
    dropout: float = 0.05
    bias: str = "none"  # Options: "none", "all", "lora_only"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    task_type: str = "CAUSAL_LM"
    lora_path: Optional[str] = None  # Path to existing LoRA weights for continued training


@dataclass
class DataConfig:
    """Data configuration."""
    
    # Data paths
    data_dir: str = "data/processed"
    train_file: str = "train.jsonl"
    validation_file: str = "validation.jsonl"
    test_file: str = "test.jsonl"
    
    # Data processing
    max_seq_length: int = 4096
    pad_to_max_length: bool = False
    
    # DataLoader settings
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    dataloader_num_workers: int = 4
    prefetch_factor: int = 2
    
    def get_train_path(self) -> str:
        """Get path to training data."""
        return os.path.join(self.data_dir, self.train_file)
    
    def get_validation_path(self) -> str:
        """Get path to validation data."""
        return os.path.join(self.data_dir, self.validation_file)
    
    def get_test_path(self) -> str:
        """Get path to test data."""
        return os.path.join(self.data_dir, self.test_file)


@dataclass
class TrainConfig:
    """Training configuration."""
    
    # Output directories
    output_dir: str = "outputs/medical-llama-3.1-8B"
    logging_dir: str = "logs"
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Training schedule
    num_train_epochs: int = 3
    max_steps: int = -1  # Overrides num_train_epochs if > 0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"  # Options: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    
    # Regularization
    gradient_accumulation_steps: int = 4
    
    # Logging & Evaluation
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Distributed training
    local_rank: int = -1
    deepspeed: Optional[str] = "configs/deepspeed_config.json"
    
    # Checkpointing
    resume_from_checkpoint: Optional[str] = None
    save_only_model: bool = True  # If True, only save model weights, not optimizer state
    
    # Miscellaneous
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    
    # SageMaker specific
    aws_region: str = "us-west-2"
    use_sagemaker: bool = True
    sagemaker_bucket: Optional[str] = None
    sagemaker_role: Optional[str] = None
    
    def to_transformers_training_args(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=self.logging_dir,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            max_grad_norm=self.max_grad_norm,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            per_device_train_batch_size=16,  # Set from DataConfig
            per_device_eval_batch_size=16,   # Set from DataConfig
            logging_steps=self.logging_steps,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            fp16=self.fp16,
            bf16=self.bf16,
            local_rank=self.local_rank,
            deepspeed=self.deepspeed,
            save_strategy="steps",
            evaluation_strategy="steps",
            report_to=["tensorboard", "wandb"],
            remove_unused_columns=False,
            label_names=["labels"],
        )


@dataclass
class Config:
    """Main configuration that combines all other configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    def save(self, output_path: str) -> None:
        """Save configuration to a JSON file."""
        import json
        from dataclasses import asdict
        
        config_dict = asdict(self)
        
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, input_path: str) -> "Config":
        """Load configuration from a JSON file."""
        import json
        
        with open(input_path, "r") as f:
            config_dict = json.load(f)
        
        # Recreate nested dataclasses
        model_config = ModelConfig(**config_dict.pop("model"))
        lora_config = LoraConfig(**config_dict.pop("lora"))
        data_config = DataConfig(**config_dict.pop("data"))
        train_config = TrainConfig(**config_dict.pop("train"))
        
        return cls(
            model=model_config,
            lora=lora_config,
            data=data_config,
            train=train_config,
        )


# Export default configuration
config = Config()


# DeepSpeed configuration for ml.p4d.24xlarge (8x A100 GPUs)
def generate_deepspeed_config() -> Dict:
    """Generate DeepSpeed configuration for ml.p4d.24xlarge."""
    return {
        "train_micro_batch_size_per_gpu": 16,
        "gradient_accumulation_steps": 4,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
            "stage3_prefetch_bucket_size": 2e8,
            "stage3_param_persistence_threshold": 1e6
        },
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 2e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 1e-6,
                "warmup_max_lr": 2e-5,
                "warmup_num_steps": 100,
                "total_num_steps": 5000
            }
        },
        "steps_per_print": 10,
        "wall_clock_breakdown": False
    }


def save_deepspeed_config(output_path: str = "configs/deepspeed_config.json") -> None:
    """Save DeepSpeed configuration to a JSON file."""
    import json
    import os
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(generate_deepspeed_config(), f, indent=2)
