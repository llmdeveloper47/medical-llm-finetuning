#!/bin/bash
set -e

echo "Downloading and preparing MedQA dataset for fine-tuning..."

# Create data directories if they don't exist
mkdir -p data/raw
mkdir -p data/processed

# Download MedQA dataset from Hugging Face Datasets
echo "Downloading MedQA dataset..."
python -c "
from datasets import load_dataset
import os

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Download the dataset
print('Downloading MedQA dataset from Hugging Face...')
dataset = load_dataset('GBaker/MedQA-USMLE-4-options')

# Save to disk
print('Saving raw dataset to disk...')
dataset.save_to_disk('data/raw/medqa_usmle')

print('MedQA dataset successfully downloaded!')
"

# Download PubMedQA dataset as supplementary data
echo "Downloading PubMedQA dataset as supplementary data..."
python -c "
from datasets import load_dataset
import os

# Download the dataset
print('Downloading PubMedQA dataset from Hugging Face...')
dataset = load_dataset('pubmed_qa')

# Save to disk
print('Saving raw dataset to disk...')
dataset.save_to_disk('data/raw/pubmed_qa')

print('PubMedQA dataset successfully downloaded!')
"

# Process datasets into instruction format
echo "Processing datasets into instruction format..."
python src/data/medqa_processor.py --input_dir data/raw --output_dir data/processed

# Create train/validation/test splits
echo "Creating train/validation/test splits..."
python src/data/dataset.py --data_dir data/processed --output_dir data/processed --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

# Upload processed data to S3 (if specified)
read -p "Would you like to upload the processed data to S3? (y/n) " upload_to_s3
if [[ $upload_to_s3 == "y" || $upload_to_s3 == "Y" ]]; then
    read -p "Enter S3 bucket name: " s3_bucket
    
    echo "Uploading processed data to S3..."
    aws s3 cp data/processed s3://$s3_bucket/medical-llm-finetuning/data/processed --recursive
    
    echo "Data uploaded to S3 bucket: $s3_bucket"
fi

echo "Dataset preparation complete!"
echo "Next steps:"
echo "1. Explore the dataset using notebooks/dataset_exploration.ipynb"
echo "2. Configure training parameters in src/training/config.py"
echo "3. Start training with bash scripts/train.sh"
