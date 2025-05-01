#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset handling utilities for fine-tuning LLaMA 3.1 8B model on medical QA data.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


class MedicalInstructionDataset:
    """
    Dataset class for handling medical instruction-tuning data.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 4096,
        split: str = "train",
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the processed dataset
            tokenizer_name_or_path: HF tokenizer name or path
            max_length: Maximum sequence length
            split: Dataset split ("train", "validation", or "test")
        """
        self.data_path = data_path
        self.max_length = max_length
        self.split = split
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        self.dataset = self._load_dataset()
    
    def _load_dataset(self) -> Dataset:
        """
        Load the processed dataset.
        
        Returns:
            HF Dataset object
        """
        # Check if we have split files
        split_file = os.path.join(self.data_path, f"{self.split}.jsonl")
        
        if os.path.exists(split_file):
            # Load specific split
            dataset = load_dataset("json", data_files=split_file)["train"]
        else:
            # Load merged dataset and filter for the desired split
            merged_file = os.path.join(self.data_path, "merged_medical_instructions.jsonl")
            if not os.path.exists(merged_file):
                raise FileNotFoundError(f"Dataset file not found: {merged_file}")
            
            dataset = load_dataset("json", data_files=merged_file)["train"]
        
        return dataset
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """
        Tokenize and format examples for training.
        
        Args:
            examples: Batch of examples
            
        Returns:
            Processed batch with tokenized inputs
        """
        # Format each example as instruction + input + output
        formatted_prompts = []
        
        for instruction, input_text, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            # Format as per LitGPT's preferred format
            if input_text:
                formatted_prompt = f"<|user|>\n{instruction}\n\n{input_text}<|endofuser|>\n<|assistant|>\n{output}<|endofassistant|>"
            else:
                formatted_prompt = f"<|user|>\n{instruction}<|endofuser|>\n<|assistant|>\n{output}<|endofassistant|>"
            
            formatted_prompts.append(formatted_prompt)
        
        # Tokenize
        tokenized_inputs = self.tokenizer(
            formatted_prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Prepare labels (same as input_ids, but -100 for prompt tokens to ignore them in loss)
        labels = tokenized_inputs["input_ids"].clone()
        
        # Find positions of <|assistant|> in each sequence
        for i, prompt in enumerate(formatted_prompts):
            assistant_pos = prompt.find("<|assistant|>")
            if assistant_pos != -1:
                # Convert character position to token position (approximate)
                assistant_token_pos = len(self.tokenizer.encode(prompt[:assistant_pos])) - 1
                # Set all tokens before assistant's response to -100
                labels[i, :assistant_token_pos] = -100
        
        tokenized_inputs["labels"] = labels
        
        return tokenized_inputs
    
    def prepare_dataset(self) -> Dataset:
        """
        Prepare dataset for training.
        
        Returns:
            Processed dataset ready for training
        """
        # Map preprocessing function to dataset
        processed_dataset = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset.column_names,
            desc=f"Processing {self.split} dataset",
        )
        
        return processed_dataset
    
    def get_dataloader(self, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
        """
        Get DataLoader for the dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            PyTorch DataLoader
        """
        processed_dataset = self.prepare_dataset()
        
        dataloader = DataLoader(
            processed_dataset,
            batch_size=batch_size,
            shuffle=shuffle and self.split == "train",
            pin_memory=True,
            drop_last=self.split == "train",
        )
        
        return dataloader


def create_dataset_splits(
    data_path: str,
    output_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Create train/validation/test splits from the processed dataset.
    
    Args:
        data_path: Path to the processed dataset
        output_path: Path to save the split datasets
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Check ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1")
    
    # Load the merged dataset
    input_file = os.path.join(data_path, "merged_medical_instructions.jsonl")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Merged dataset not found: {input_file}")
    
    # Read examples
    examples = []
    with open(input_file, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    
    # Shuffle examples
    random.shuffle(examples)
    
    # Calculate split sizes
    n_examples = len(examples)
    n_train = int(n_examples * train_ratio)
    n_val = int(n_examples * val_ratio)
    
    # Create splits
    train_examples = examples[:n_train]
    val_examples = examples[n_train:n_train + n_val]
    test_examples = examples[n_train + n_val:]
    
    # Save splits
    os.makedirs(output_path, exist_ok=True)
    
    # Train split
    train_file = os.path.join(output_path, "train.jsonl")
    with open(train_file, "w") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")
    
    # Validation split
    val_file = os.path.join(output_path, "validation.jsonl")
    with open(val_file, "w") as f:
        for example in val_examples:
            f.write(json.dumps(example) + "\n")
    
    # Test split
    test_file = os.path.join(output_path, "test.jsonl")
    with open(test_file, "w") as f:
        for example in test_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Created dataset splits:")
    print(f"  - Train: {len(train_examples)} examples ({train_ratio*100:.1f}%)")
    print(f"  - Validation: {len(val_examples)} examples ({val_ratio*100:.1f}%)")
    print(f"  - Test: {len(test_examples)} examples ({test_ratio*100:.1f}%)")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create dataset splits for fine-tuning")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save split datasets")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    create_dataset_splits(
        data_path=args.data_dir,
        output_path=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
