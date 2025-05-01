#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for fine-tuned LLaMA 3.1 8B model on medical QA dataset.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Add project root to path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training.config import Config
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


class MedicalQAEvaluator:
    """
    Evaluator for medical QA models.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        data_path: str = "data/processed/test.jsonl",
        output_dir: str = "outputs/evaluation",
        device: str = "cuda",
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the model or model name
            tokenizer_path: Path to the tokenizer (uses model_path if None)
            data_path: Path to the test dataset
            output_dir: Directory to save evaluation results
            device: Device to use for evaluation
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Load metrics
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("sacrebleu")
        self.bertscore = evaluate.load("bertscore")
        
        # Load dataset
        self.test_dataset = self._load_dataset()
    
    def _load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            padding_side="right",
            trust_remote_code=True,
        )
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loading model from {self.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        
        # Move model to device
        model.to(self.device)
        model.eval()
        
        return model, tokenizer
    
    def _load_dataset(self) -> Dataset:
        """
        Load and prepare the test dataset.
        
        Returns:
            Test dataset
        """
        logger.info(f"Loading test dataset from {self.data_path}")
        
        if self.data_path.endswith(".jsonl"):
            # Load from JSONL file
            test_dataset = load_dataset("json", data_files=self.data_path)["train"]
        else:
            # Try loading as a Hugging Face dataset
            test_dataset = load_dataset(self.data_path)["test"]
        
        logger.info(f"Loaded {len(test_dataset)} test examples")
        
        return test_dataset
    
    def generate_predictions(
        self,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
        num_samples: Optional[int] = None,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Generate predictions for the test dataset.
        
        Args:
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum number of new tokens to generate
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Tuple of (instructions, inputs, references, predictions)
        """
        instructions = []
        inputs = []
        references = []
        predictions = []
        
        # Limit number of samples if specified
        if num_samples is not None and num_samples < len(self.test_dataset):
            indices = np.random.choice(
                len(self.test_dataset), num_samples, replace=False
            )
            test_dataset = self.test_dataset.select(indices)
        else:
            test_dataset = self.test_dataset
        
        logger.info(f"Generating predictions for {len(test_dataset)} examples")
        
        for idx, example in enumerate(tqdm(test_dataset)):
            instruction = example["instruction"]
            input_text = example["input"]
            reference = example["output"]
            
            # Format prompt as per model's training format
            if input_text:
                prompt = f"<|user|>\n{instruction}\n\n{input_text}<|endofuser|>\n<|assistant|>"
            else:
                prompt = f"<|user|>\n{instruction}<|endofuser|>\n<|assistant|>"
            
            # Tokenize
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=(temperature > 0),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode the output, extract only the assistant's reply
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
            
            # Extract only the assistant's reply
            assistant_part = generated_text.split("<|assistant|>")[1].split("<|endofassistant|>")[0].strip()
            
            instructions.append(instruction)
            inputs.append(input_text)
            references.append(reference)
            predictions.append(assistant_part)
        
        return instructions, inputs, references, predictions
    
    def compute_metrics(
        self,
        references: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            references: Reference outputs
            predictions: Model predictions
            
        Returns:
            Dictionary of metrics
        """
        # Initialize metrics dictionary
        metrics = {}
        
        # ROUGE scores
        logger.info("Computing ROUGE scores")
        rouge_output = self.rouge.compute(predictions=predictions, references=references)
        metrics.update({f"rouge_{k}": v for k, v in rouge_output.items()})
        
        # BLEU score
        logger.info("Computing BLEU score")
        bleu_output = self.bleu.compute(predictions=predictions, references=[[r] for r in references])
        metrics["bleu"] = bleu_output["score"]
        
        # BERTScore (on a subset to avoid OOM)
        try:
            logger.info("Computing BERTScore")
            max_bertscore_samples = min(100, len(predictions))
            bertscore_output = self.bertscore.compute(
                predictions=predictions[:max_bertscore_samples],
                references=references[:max_bertscore_samples],
                lang="en",
            )
            metrics["bertscore_precision"] = np.mean(bertscore_output["precision"])
            metrics["bertscore_recall"] = np.mean(bertscore_output["recall"])
            metrics["bertscore_f1"] = np.mean(bertscore_output["f1"])
        except Exception as e:
            logger.warning(f"Failed to compute BERTScore: {e}")
        
        # Medical domain-specific metrics
        logger.info("Computing medical domain-specific metrics")
        medical_metrics = self._compute_medical_metrics(references, predictions)
        metrics.update(medical_metrics)
        
        return metrics
    
    def _compute_medical_metrics(
        self,
        references: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        """
        Compute medical domain-specific metrics.
        
        Args:
            references: Reference outputs
            predictions: Model predictions
            
        Returns:
            Dictionary of medical metrics
        """
        medical_metrics = {}
        
        # Diagnosis accuracy: Check if diagnoses match between reference and prediction
        diagnosis_pattern = r"(?i)diagnosis:\s*(.*?)(?=treatment|plan|$)"
        
        diagnosis_matches = 0
        total_diagnoses = 0
        
        for ref, pred in zip(references, predictions):
            ref_match = re.search(diagnosis_pattern, ref)
            pred_match = re.search(diagnosis_pattern, pred)
            
            if ref_match and pred_match:
                total_diagnoses += 1
                ref_diagnosis = ref_match.group(1).strip().lower()
                pred_diagnosis = pred_match.group(1).strip().lower()
                
                # Check if diagnoses match
                if self._text_similarity(ref_diagnosis, pred_diagnosis) > 0.7:
                    diagnosis_matches += 1
        
        if total_diagnoses > 0:
            medical_metrics["diagnosis_accuracy"] = diagnosis_matches / total_diagnoses
        else:
            medical_metrics["diagnosis_accuracy"] = 0.0
        
        # Treatment plan presence
        treatment_pattern = r"(?i)treatment|plan|therapy|management"
        
        treatment_present = 0
        for pred in predictions:
            if re.search(treatment_pattern, pred):
                treatment_present += 1
        
        medical_metrics["treatment_plan_presence"] = treatment_present / len(predictions)
        
        return medical_metrics
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using Jaccard similarity of words.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def evaluate(
        self,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on the test dataset.
        
        Args:
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum number of new tokens to generate
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting evaluation")
        start_time = time.time()
        
        # Generate predictions
        instructions, inputs, references, predictions = self.generate_predictions(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
        )
        
        # Compute metrics
        metrics = self.compute_metrics(references, predictions)
        
        # Save predictions and metrics
        self._save_results(instructions, inputs, references, predictions, metrics)
        
        logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Metrics: {metrics}")
        
        return metrics
    
    def _save_results(
        self,
        instructions: List[str],
        inputs: List[str],
        references: List[str],
        predictions: List[str],
        metrics: Dict[str, float],
    ) -> None:
        """
        Save evaluation results.
        
        Args:
            instructions: Instructions
            inputs: Input texts
            references: Reference outputs
            predictions: Model predictions
            metrics: Evaluation metrics
        """
        # Save predictions
        results = []
        for instruction, input_text, reference, prediction in zip(
            instructions, inputs, references, predictions
        ):
            results.append({
                "instruction": instruction,
                "input": input_text,
                "reference": reference,
                "prediction": prediction,
            })
        
        predictions_file = os.path.join(self.output_dir, "predictions.jsonl")
        with open(predictions_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        # Save metrics
        metrics_file = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved predictions to {predictions_file}")
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Generate a human-readable report
        report_file = os.path.join(self.output_dir, "evaluation_report.md")
        self._generate_report(results, metrics, report_file)
    
    def _generate_report(
        self,
        results: List[Dict[str, str]],
        metrics: Dict[str, float],
        output_file: str,
    ) -> None:
        """
        Generate a human-readable evaluation report.
        
        Args:
            results: Evaluation results
            metrics: Evaluation metrics
            output_file: Output file path
        """
        with open(output_file, "w") as f:
            f.write("# Medical QA Model Evaluation Report\n\n")
            
            # Add metrics
            f.write("## Evaluation Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for metric, value in metrics.items():
                f.write(f"| {metric} | {value:.4f} |\n")
            
            # Add sample predictions
            f.write("\n## Sample Predictions\n\n")
            
            # Select a few examples (max 10)
            num_examples = min(10, len(results))
            sample_indices = np.random.choice(len(results), num_examples, replace=False)
            
            for i, idx in enumerate(sample_indices):
                result = results[idx]
                
                f.write(f"### Example {i+1}\n\n")
                f.write(f"**Instruction:** {result['instruction']}\n\n")
                f.write(f"**Input:** {result['input']}\n\n")
                f.write(f"**Reference:** {result['reference']}\n\n")
                f.write(f"**Prediction:** {result['prediction']}\n\n")
                f.write("---\n\n")
        
        logger.info(f"Generated evaluation report at {output_file}")


def compare_with_baseline(
    finetuned_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    output_file: str,
) -> None:
    """
    Compare fine-tuned model with baseline model.
    
    Args:
        finetuned_metrics: Fine-tuned model metrics
        baseline_metrics: Baseline model metrics
        output_file: Output file path
    """
    with open(output_file, "w") as f:
        f.write("# Model Comparison: Fine-tuned vs. Baseline\n\n")
        
        f.write("| Metric | Baseline | Fine-tuned | Improvement |\n")
        f.write("|--------|----------|------------|-------------|\n")
        
        for metric in finetuned_metrics:
            if metric in baseline_metrics:
                baseline_value = baseline_metrics[metric]
                finetuned_value = finetuned_metrics[metric]
                improvement = finetuned_value - baseline_value
                improvement_pct = (improvement / baseline_value) * 100 if baseline_value != 0 else float('inf')
                
                f.write(f"| {metric} | {baseline_value:.4f} | {finetuned_value:.4f} | {improvement_pct:+.2f}% |\n")
    
    logger.info(f"Generated model comparison report at {output_file}")


def evaluate_baseline(config: Config, output_dir: str) -> Dict[str, float]:
    """
    Evaluate the baseline model (LLaMA 3.1 8B without fine-tuning).
    
    Args:
        config: Training configuration
        output_dir: Output directory
        
    Returns:
        Baseline model metrics
    """
    logger.info("Evaluating baseline model")
    
    baseline_evaluator = MedicalQAEvaluator(
        model_path=config.model.model_name,
        data_path=os.path.join(config.data.data_dir, "test.jsonl"),
        output_dir=os.path.join(output_dir, "baseline"),
    )
    
    baseline_metrics = baseline_evaluator.evaluate(num_samples=100)
    
    return baseline_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLaMA 3.1 8B model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer (uses model_path if None)")
    parser.add_argument("--data_path", type=str, default="data/processed/test.jsonl", help="Path to the test dataset")
    parser.add_argument("--config", type=str, default="configs/train_config.json", help="Path to training configuration")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation", help="Directory to save evaluation results")
    parser.add_argument("--compare_baseline", action="store_true", help="Compare with baseline model")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (None for all)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    if os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        logger.warning(f"Config file not found: {args.config}, using default config")
        config = Config()
    
    # Evaluate fine-tuned model
    evaluator = MedicalQAEvaluator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
    
    finetuned_metrics = evaluator.evaluate(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
    )
    
    # Compare with baseline if requested
    if args.compare_baseline:
        baseline_metrics = evaluate_baseline(config, args.output_dir)
        
        # Generate comparison report
        comparison_file = os.path.join(args.output_dir, "model_comparison.md")
        compare_with_baseline(finetuned_metrics, baseline_metrics, comparison_file)


if __name__ == "__main__":
    main()
