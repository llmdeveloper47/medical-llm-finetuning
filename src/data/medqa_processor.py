#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MedQA dataset processor for fine-tuning LLaMA 3.1 8B model.
Converts the MedQA dataset to an instruction-following format.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from datasets import Dataset, load_from_disk
from tqdm import tqdm


INSTRUCTION_TEMPLATES = [
    "Given the following patient case, provide the most appropriate diagnosis and treatment plan.",
    "Based on the patient information provided, determine the diagnosis and recommend a treatment plan.",
    "Review the following clinical scenario and provide your diagnosis and treatment recommendations.",
    "As a medical professional, analyze this patient case and provide the most likely diagnosis and appropriate treatment plan.",
    "Evaluate the following patient presentation and provide clinical assessment and management recommendations."
]

def create_instruction_example(
    question: str,
    answer: str,
    options: Optional[List[str]] = None,
    instruction: Optional[str] = None,
) -> Dict[str, str]:
    """
    Create a single instruction example for fine-tuning.
    
    Args:
        question: The patient case/question
        answer: The correct answer
        options: Multiple choice options if available
        instruction: Specific instruction to use (random one chosen if None)
        
    Returns:
        Dict with instruction, input, and output fields
    """
    # Clean up the question text
    question = question.strip()
    
    # Extract patient demographics if present
    patient_age = None
    patient_gender = None
    
    age_indicators = ["year-old", "yo", "year old", "years old"]
    gender_indicators = ["male", "female", "man", "woman", "boy", "girl"]
    
    # Simple extraction of age and gender (can be improved with NLP techniques)
    for age_ind in age_indicators:
        if age_ind in question.lower():
            try:
                # Find text before age indicator
                idx = question.lower().find(age_ind)
                age_text = question[max(0, idx-3):idx].strip()
                # Extract digits
                patient_age = ''.join(filter(str.isdigit, age_text))
            except:
                pass
    
    for gender_ind in gender_indicators:
        if gender_ind in question.lower():
            patient_gender = gender_ind
            break
    
    # Format the answer to include diagnosis and treatment
    if "diagnosis" not in answer.lower():
        expanded_answer = f"Diagnosis: {answer}"
        
        # Add a generic treatment plan if not provided
        if "treatment" not in answer.lower():
            expanded_answer += "\nTreatment plan: Based on this diagnosis, I recommend the following:\n"
            expanded_answer += "1. Further diagnostic tests to confirm the diagnosis.\n"
            expanded_answer += "2. Appropriate medication based on current clinical guidelines.\n"
            expanded_answer += "3. Patient education about the condition and management.\n"
            expanded_answer += "4. Follow-up to monitor response to treatment."
    else:
        expanded_answer = answer
    
    # Select a random instruction if not provided
    if instruction is None:
        instruction = random.choice(INSTRUCTION_TEMPLATES)
    
    # Create the example
    example = {
        "instruction": instruction,
        "input": question,
        "output": expanded_answer
    }
    
    return example


def process_medqa_usmle(
    dataset_path: str,
    output_path: str,
    instruction: Optional[str] = None,
) -> None:
    """
    Process the MedQA-USMLE dataset into instruction format.
    
    Args:
        dataset_path: Path to the raw MedQA dataset
        output_path: Path to save the processed dataset
        instruction: Specific instruction to use (random one chosen if None)
    """
    print(f"Loading MedQA dataset from {dataset_path}...")
    medqa_dataset = load_from_disk(dataset_path)
    
    print("Converting to instruction format...")
    instruction_examples = []
    
    for split in medqa_dataset:
        for example in tqdm(medqa_dataset[split]):
            question = example["question"]
            
            # Create detailed answer
            answer_idx = ord(example["answer"]) - ord("A")
            if 0 <= answer_idx < len(example["options"]):
                answer = example["options"][answer_idx]
            else:
                answer = "Unable to determine from the provided information."
            
            # Create and add the instruction example
            instruction_example = create_instruction_example(
                question=question,
                answer=answer,
                options=example["options"],
                instruction=instruction,
            )
            instruction_examples.append(instruction_example)
    
    # Save as JSONL
    output_file = os.path.join(output_path, "medqa_instructions.jsonl")
    with open(output_file, "w") as f:
        for example in instruction_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Saved {len(instruction_examples)} instruction examples to {output_file}")


def process_pubmedqa(
    dataset_path: str,
    output_path: str,
    instruction: Optional[str] = None,
) -> None:
    """
    Process the PubMedQA dataset into instruction format.
    
    Args:
        dataset_path: Path to the raw PubMedQA dataset
        output_path: Path to save the processed dataset
        instruction: Specific instruction to use (random one chosen if None)
    """
    print(f"Loading PubMedQA dataset from {dataset_path}...")
    pubmedqa_dataset = load_from_disk(dataset_path)
    
    print("Converting to instruction format...")
    instruction_examples = []
    
    # Process only examples with definitive yes/no answers
    for example in tqdm(pubmedqa_dataset["train"]):
        if example["final_decision"] in ["yes", "no"]:
            context = example["context"]["contexts"][0] if example["context"]["contexts"] else ""
            
            # Construct a medical case from the question and context
            question = f"Patient case: {context}\n\nMedical question: {example['question']}"
            
            # Create an answer based on the yes/no decision and explanation
            if example["final_decision"] == "yes":
                answer = f"Yes. {example['long_answer']}"
            else:
                answer = f"No. {example['long_answer']}"
            
            # Create and add the instruction example
            instruction_example = create_instruction_example(
                question=question, 
                answer=answer,
                instruction=instruction,
            )
            instruction_examples.append(instruction_example)
    
    # Save as JSONL
    output_file = os.path.join(output_path, "pubmedqa_instructions.jsonl")
    with open(output_file, "w") as f:
        for example in instruction_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Saved {len(instruction_examples)} instruction examples to {output_file}")


def merge_datasets(output_path: str) -> None:
    """
    Merge all processed datasets into a single one.
    
    Args:
        output_path: Directory containing processed datasets
    """
    print("Merging all processed datasets...")
    all_examples = []
    
    # Load MedQA examples
    medqa_file = os.path.join(output_path, "medqa_instructions.jsonl")
    if os.path.exists(medqa_file):
        with open(medqa_file, "r") as f:
            for line in f:
                all_examples.append(json.loads(line))
    
    # Load PubMedQA examples
    pubmedqa_file = os.path.join(output_path, "pubmedqa_instructions.jsonl")
    if os.path.exists(pubmedqa_file):
        with open(pubmedqa_file, "r") as f:
            for line in f:
                all_examples.append(json.loads(line))
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Save merged dataset
    output_file = os.path.join(output_path, "merged_medical_instructions.jsonl")
    with open(output_file, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Saved {len(all_examples)} merged instruction examples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process medical datasets to instruction format")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw datasets")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed datasets")
    parser.add_argument("--instruction", type=str, default=None, help="Custom instruction (random if not provided)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process MedQA dataset
    medqa_path = os.path.join(args.input_dir, "medqa_usmle")
    if os.path.exists(medqa_path):
        process_medqa_usmle(medqa_path, args.output_dir, args.instruction)
    else:
        print(f"Warning: MedQA dataset not found at {medqa_path}")
    
    # Process PubMedQA dataset
    pubmedqa_path = os.path.join(args.input_dir, "pubmed_qa")
    if os.path.exists(pubmedqa_path):
        process_pubmedqa(pubmedqa_path, args.output_dir, args.instruction)
    else:
        print(f"Warning: PubMedQA dataset not found at {pubmedqa_path}")
    
    # Merge datasets
    merge_datasets(args.output_dir)
    
    print("Dataset processing complete!")


if __name__ == "__main__":
    main()
