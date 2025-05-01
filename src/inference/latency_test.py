#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Latency measurement script for fine-tuned LLaMA 3.1 8B model on medical QA tasks.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Add project root to path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


class LatencyTester:
    """
    Measure inference latency for language models.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        output_dir: str = "outputs/latency",
        device: str = "cuda",
        precision: str = "bf16",
    ):
        """
        Initialize the latency tester.
        
        Args:
            model_path: Path to the model or model name
            tokenizer_path: Path to the tokenizer (uses model_path if None)
            output_dir: Directory to save results
            device: Device to use for inference ("cuda" or "cpu")
            precision: Model precision ("fp32", "fp16", or "bf16")
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.output_dir = output_dir
        self.device = device
        self.precision = precision
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Sample prompts for testing
        self.sample_prompts = [
            "A 67-year-old man presents with chest pain radiating to the left arm, shortness of breath, and diaphoresis that began 2 hours ago. He has a history of hypertension and diabetes. What is the most likely diagnosis and appropriate treatment plan?",
            "A 35-year-old woman presents with fever, productive cough, and pleuritic chest pain for the past 3 days. Her temperature is 102.3°F, heart rate 110, respiratory rate 22, and oxygen saturation 94% on room air. What is the most likely diagnosis and appropriate treatment plan?",
            "A 42-year-old man presents with sudden onset severe headache described as 'the worst headache of my life.' He has neck stiffness and photophobia. BP is 170/100, HR 90. What is the most likely diagnosis and appropriate treatment plan?",
            "A 55-year-old woman with a history of COPD presents with worsening shortness of breath over the past week, increased sputum production, and wheezing. She uses an albuterol inhaler at home which is no longer providing relief. What is the most likely diagnosis and appropriate treatment plan?",
            "A 28-year-old previously healthy woman presents with right lower quadrant abdominal pain that began periumbilically 12 hours ago. She has nausea, vomiting, and a temperature of 100.2°F. What is the most likely diagnosis and appropriate treatment plan?",
        ]
    
    def _load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
        )
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loading model from {self.model_path}")
        
        # Set dtype based on precision
        if self.precision == "fp16":
            dtype = torch.float16
        elif self.precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        # Move model to device if not using device_map
        if self.device == "cpu":
            model = model.to(self.device)
        
        model.eval()
        
        return model, tokenizer
    
    def measure_latency(
        self,
        batch_sizes: List[int] = [1, 4, 8, 16],
        sequence_lengths: List[int] = [512, 1024, 2048, 4096],
        num_iterations: int = 10,
        warmup_iterations: int = 3,
    ) -> Dict:
        """
        Measure model inference latency across different batch sizes and sequence lengths.
        
        Args:
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            num_iterations: Number of iterations for each configuration
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary of latency results
        """
        logger.info("Starting latency measurements")
        
        results = {
            "model": self.model_path,
            "device": self.device,
            "precision": self.precision,
            "batch_sizes": batch_sizes,
            "sequence_lengths": sequence_lengths,
            "measurements": [],
        }
        
        # For each batch size and sequence length combination
        for batch_size in batch_sizes:
            for seq_length in sequence_lengths:
                logger.info(f"Testing batch_size={batch_size}, seq_length={seq_length}")
                
                # Prepare prompts
                prompts = []
                for i in range(batch_size):
                    prompt_idx = i % len(self.sample_prompts)
                    prompt = f"<|user|>\n{self.sample_prompts[prompt_idx]}<|endofuser|>\n<|assistant|>"
                    prompts.append(prompt)
                
                # Tokenize
                inputs = self.tokenizer(
                    prompts,
                    padding="max_length",
                    truncation=True,
                    max_length=seq_length,
                    return_tensors="pt",
                ).to(self.device)
                
                # Warmup
                logger.info(f"Running {warmup_iterations} warmup iterations")
                for _ in range(warmup_iterations):
                    with torch.no_grad():
                        self.model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=32,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                
                # Measure latency
                logger.info(f"Running {num_iterations} measurement iterations")
                
                generation_times = []
                tokens_generated = []
                
                # Measure prefill latency (first token)
                prefill_times = []
                
                for _ in range(num_iterations):
                    # Measure prefill (time to first token)
                    torch.cuda.synchronize() if self.device == "cuda" else None
                    prefill_start = time.time()
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=32,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                    
                    torch.cuda.synchronize() if self.device == "cuda" else None
                    prefill_end = time.time()
                    
                    prefill_times.append(prefill_end - prefill_start)
                    
                    # Measure full generation
                    torch.cuda.synchronize() if self.device == "cuda" else None
                    gen_start = time.time()
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=128,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    
                    torch.cuda.synchronize() if self.device == "cuda" else None
                    gen_end = time.time()
                    
                    generation_times.append(gen_end - gen_start)
                    
                    # Count generated tokens
                    num_input_tokens = inputs["input_ids"].shape[1]
                    num_output_tokens = outputs.shape[1]
                    num_new_tokens = num_output_tokens - num_input_tokens
                    tokens_generated.append(num_new_tokens)
                
                # Calculate statistics
                avg_gen_time = np.mean(generation_times)
                avg_tokens_generated = np.mean(tokens_generated)
                avg_time_per_token = avg_gen_time / avg_tokens_generated
                avg_prefill_time = np.mean(prefill_times)
                
                throughput = (avg_tokens_generated * batch_size) / avg_gen_time
                
                # Measure memory usage
                if self.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=128,
                            do_sample=False,
                        )
                    memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
                else:
                    memory_used = None
                
                # Record results
                results["measurements"].append({
                    "batch_size": batch_size,
                    "sequence_length": seq_length,
                    "avg_generation_time": avg_gen_time,
                    "avg_time_per_token": avg_time_per_token,
                    "avg_prefill_time": avg_prefill_time,
                    "throughput": throughput,
                    "memory_used_gb": memory_used,
                })
                
                logger.info(f"Results for batch_size={batch_size}, seq_length={seq_length}:")
                logger.info(f"  Average generation time: {avg_gen_time:.4f} s")
                logger.info(f"  Average time per token: {avg_time_per_token:.4f} s")
                logger.info(f"  Throughput: {throughput:.2f} tokens/s")
                logger.info(f"  Average prefill time: {avg_prefill_time:.4f} s")
                logger.info(f"  Memory used: {memory_used:.2f} GB" if memory_used else "  Memory used: N/A")
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict) -> None:
        """
        Save latency results to file and generate plots.
        
        Args:
            results: Latency measurement results
        """
        # Save raw results
        output_file = os.path.join(self.output_dir, "latency_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved latency results to {output_file}")
        
        # Generate plots
        self._generate_plots(results)
        
        # Generate report
        self._generate_report(results)
    
    def _generate_plots(self, results: Dict) -> None:
        """
        Generate plots from latency results.
        
        Args:
            results: Latency measurement results
        """
        # Convert results to DataFrame for easier plotting
        measurements = results["measurements"]
        df = pd.DataFrame(measurements)
        
        # Set style
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 10))
        
        # 1. Plot latency vs batch size for different sequence lengths
        plt.subplot(2, 2, 1)
        for seq_length in results["sequence_lengths"]:
            subset = df[df["sequence_length"] == seq_length]
            plt.plot(subset["batch_size"], subset["avg_generation_time"], marker="o", label=f"Seq Len {seq_length}")
        
        plt.xlabel("Batch Size")
        plt.ylabel("Generation Time (s)")
        plt.title("Generation Time vs Batch Size")
        plt.legend()
        plt.grid(True)
        
        # 2. Plot throughput vs batch size for different sequence lengths
        plt.subplot(2, 2, 2)
        for seq_length in results["sequence_lengths"]:
            subset = df[df["sequence_length"] == seq_length]
            plt.plot(subset["batch_size"], subset["throughput"], marker="o", label=f"Seq Len {seq_length}")
        
        plt.xlabel("Batch Size")
        plt.ylabel("Throughput (tokens/s)")
        plt.title("Throughput vs Batch Size")
        plt.legend()
        plt.grid(True)
        
        # 3. Plot memory usage vs batch size for different sequence lengths
        if "memory_used_gb" in df.columns and not df["memory_used_gb"].isna().all():
            plt.subplot(2, 2, 3)
            for seq_length in results["sequence_lengths"]:
                subset = df[df["sequence_length"] == seq_length]
                plt.plot(subset["batch_size"], subset["memory_used_gb"], marker="o", label=f"Seq Len {seq_length}")
            
            plt.xlabel("Batch Size")
            plt.ylabel("Memory Usage (GB)")
            plt.title("Memory Usage vs Batch Size")
            plt.legend()
            plt.grid(True)
        
        # 4. Plot time per token vs sequence length for different batch sizes
        plt.subplot(2, 2, 4)
        for batch_size in results["batch_sizes"]:
            subset = df[df["batch_size"] == batch_size]
            plt.plot(subset["sequence_length"], subset["avg_time_per_token"], marker="o", label=f"Batch {batch_size}")
        
        plt.xlabel("Sequence Length")
        plt.ylabel("Time per Token (s)")
        plt.title("Time per Token vs Sequence Length")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.output_dir, "latency_plots.png")
        plt.savefig(plot_file, dpi=300)
        logger.info(f"Saved latency plots to {plot_file}")
        
        plt.close()
    
    def _generate_report(self, results: Dict) -> None:
        """
        Generate a human-readable report from latency results.
        
        Args:
            results: Latency measurement results
        """
        report_file = os.path.join(self.output_dir, "latency_report.md")
        
        with open(report_file, "w") as f:
            f.write("# LLaMA 3.1 8B Medical QA Model Latency Report\n\n")
            
            f.write(f"## Model Information\n\n")
            f.write(f"- **Model**: {results['model']}\n")
            f.write(f"- **Device**: {results['device']}\n")
            f.write(f"- **Precision**: {results['precision']}\n\n")
            
            f.write("## Latency Measurements\n\n")
            
            # Summary table
            f.write("### Summary Table\n\n")
            f.write("| Batch Size | Sequence Length | Generation Time (s) | Time per Token (s) | Throughput (tokens/s) | Memory (GB) |\n")
            f.write("|------------|-----------------|---------------------|---------------------|------------------------|------------|\n")
            
            for measurement in results["measurements"]:
                memory = f"{measurement['memory_used_gb']:.2f}" if measurement.get("memory_used_gb") is not None else "N/A"
                f.write(f"| {measurement['batch_size']} | {measurement['sequence_length']} | "
                       f"{measurement['avg_generation_time']:.4f} | {measurement['avg_time_per_token']:.4f} | "
                       f"{measurement['throughput']:.2f} | {memory} |\n")
            
            # Key findings
            f.write("\n## Key Findings\n\n")
            
            # Find optimal batch size for throughput
            df = pd.DataFrame(results["measurements"])
            max_throughput_idx = df["throughput"].idxmax()
            optimal_config = df.iloc[max_throughput_idx]
            
            f.write(f"### Optimal Configuration\n\n")
            f.write(f"- Highest throughput achieved: **{optimal_config['throughput']:.2f} tokens/s**\n")
            f.write(f"- Batch size: **{optimal_config['batch_size']}**\n")
            f.write(f"- Sequence length: **{optimal_config['sequence_length']}**\n\n")
            
            f.write("### Analysis\n\n")
            f.write("1. **Batch Size Impact**: ")
            if len(results["batch_sizes"]) > 1:
                for seq_length in results["sequence_lengths"]:
                    subset = df[df["sequence_length"] == seq_length]
                    throughput_increase = subset["throughput"].max() / subset["throughput"].min()
                    f.write(f"For sequence length {seq_length}, increasing batch size from {subset['batch_size'].min()} "
                           f"to {subset['batch_size'].max()} resulted in a {throughput_increase:.2f}x throughput increase. ")
            f.write("\n\n")
            
            f.write("2. **Sequence Length Impact**: ")
            if len(results["sequence_lengths"]) > 1:
                for batch_size in results["batch_sizes"]:
                    subset = df[df["batch_size"] == batch_size]
                    time_per_token_increase = subset["avg_time_per_token"].max() / subset["avg_time_per_token"].min()
                    f.write(f"For batch size {batch_size}, increasing sequence length from {subset['sequence_length'].min()} "
                           f"to {subset['sequence_length'].max()} resulted in a {time_per_token_increase:.2f}x increase in time per token. ")
            f.write("\n\n")
            
            if "memory_used_gb" in df.columns and not df["memory_used_gb"].isna().all():
                f.write("3. **Memory Usage**: ")
                max_memory_idx = df["memory_used_gb"].idxmax()
                max_memory_config = df.iloc[max_memory_idx]
                f.write(f"Maximum memory usage was {max_memory_config['memory_used_gb']:.2f} GB with batch size {max_memory_config['batch_size']} "
                       f"and sequence length {max_memory_config['sequence_length']}. ")
                f.write("\n\n")
            
            f.write("## Conclusion\n\n")
            f.write(f"Based on these measurements, the recommended configuration for optimal throughput is batch size **{optimal_config['batch_size']}** "
                   f"with a context length of **{optimal_config['sequence_length']}** tokens, which achieves **{optimal_config['throughput']:.2f} tokens/s**.\n\n")
            
            f.write("For latency-sensitive applications where response time is critical, smaller batch sizes may be preferred despite lower throughput.\n\n")
            
            f.write("![Latency Plots](./latency_plots.png)\n")
        
        logger.info(f"Generated latency report at {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Measure inference latency of LLaMA 3.1 8B model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer (uses model_path if None)")
    parser.add_argument("--output_dir", type=str, default="outputs/latency", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for inference")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Model precision")
    parser.add_argument("--batch_sizes", type=str, default="1,4,8,16", help="Comma-separated list of batch sizes to test")
    parser.add_argument("--sequence_lengths", type=str, default="512,1024,2048,4096", help="Comma-separated list of sequence lengths to test")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of iterations for each configuration")
    parser.add_argument("--warmup_iterations", type=int, default=3, help="Number of warmup iterations")
    
    args = parser.parse_args()
    
    # Parse batch sizes and sequence lengths
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    sequence_lengths = [int(sl) for sl in args.sequence_lengths.split(",")]
    
    # Check if GPU is available
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    
    # Create latency tester
    tester = LatencyTester(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        device=args.device,
        precision=args.precision,
    )
    
    # Measure latency
    tester.measure_latency(
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths,
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations,
    )


if __name__ == "__main__":
    main()
