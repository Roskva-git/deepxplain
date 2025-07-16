# Activity 1: : # Fine-Tuning BERT and Applying LIME & SHAP for Sentiment Analysis

## Chapter 1: Preprocessing

# Written by: Røskva
# Created: 01. July 2025
# Updated: 03. July 2025

# 1. PREPROCESSING STEPS
### 1.0 Setting up environment and configuration
### 1.1 Loading model and tokenizer
### 1.2 Creating a stratified split
### 1.3 Validation for sanity
### 1.4 Text analysis
### 1.5 Creating a dataset class - tokenization and encoding
### 1.6 Creating datasets for training and validation
### 1.7 Creating dataloaders
### 1.8 Saving preprocessed data



## ---- 1.0 Setting up environment and configuration ----  

# Importing libraries and tools

import numpy as np
import pandas as pd
import torch
import sklearn
import pickle
import os
import json

from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

# ==== REMEMBER TO UNCOMMENT FOR FOX ====
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Dataset configuration - modify for hatespeech dataset
DATASET_NAME = "HateBRXplain_cleaned"           
MODEL_NAME = 'adalbertojunior/distilbert-portuguese-cased'
MAX_LENGTH = 512
BATCH_SIZE = 16                  # Change from 8
NUM_WORKERS = 0                 # Change from 0
TEST_SIZE = 0.3                 # Validation split ratio
RANDOM_STATE = 42

# Versioned output directory - FOR FOX CLUSTER
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
BASE_OUTPUT_DIR = "/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/preprocessed_data" # Change this as needed
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"{DATASET_NAME}_{MODEL_NAME.replace('/', '_')}_maxlen{MAX_LENGTH}_{timestamp}")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Created output directory: {OUTPUT_DIR}")


print("\nFOX CLUSTER CONFIGURATION:")
print(f"  Dataset: {DATASET_NAME}")
print(f"  Model: {MODEL_NAME}")
print(f"  Max length: {MAX_LENGTH}")
print(f"  Batch size: {BATCH_SIZE} (Fox optimized)")
print(f"  Workers: {NUM_WORKERS} (Fox parallel)")
print(f"  Output directory: {OUTPUT_DIR}")
print()

print("FOX CLUSTER SAVING CONFIGURATION:")
print(f"  Base directory: {BASE_OUTPUT_DIR}")
print(f"  This run will save to: {OUTPUT_DIR}")
print()


## ---- 1.1 Loading model and tokenizer ----

print('\n1.1 LOADING MODEL AND TOKENIZER')
print()
model_name = MODEL_NAME 

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

print(f"Model: {model_name}")
print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
print(f"Vocabulary size: {tokenizer.vocab_size:,} tokens")
print(f"Model loaded: {model.__class__.__name__}")
print(f"Number of labels: 2 (neutral=0, offensive=1)")
print()


# Loading dataset
print(f"Loading {DATASET_NAME} dataset...")
dataset_path = "/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/dataset/HateBRXplain_cleaned.json"

with open(dataset_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

print(f"Dataset loaded!")
print(f"Total examples: {len(raw_data)}")

# Extracting texts and labels from the dataset
texts = [item["comment"] for item in raw_data]
labels = [int(item["offensive label"]) for item in raw_data] # Converts 1.0 -> 1, 0.0 -> 0


# Look at an example
print("Sample data:")
print(f"Text: {texts[0][:300]}...")
print(f"Label: {labels[0]} (0=neutral, 1=offensive)")

print(f"Label distribution:")
print(f"  Neutral: {labels.count(0)} samples")
print(f"  Offensive: {labels.count(1)} samples")


# ==== PREPROCESSING ====

## ---- 1.2 Creating a stratified split ----

from sklearn.model_selection import train_test_split, StratifiedKFold


# Stratified split - 50/50 offensive and neutral for all splits
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, 
    labels, 
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, 
    temp_labels,
    test_size=0.5, #Splits temp into validation and test
    random_state=RANDOM_STATE,
    stratify=temp_labels
)



### ---- 1.3 Split validation for sanity ----

# Checking the split sizes
print("\n1.3 SPLIT VALIDATION")
print()

total_original = len(texts)
total_split = len(train_texts) + len(val_texts) + len(test_texts)

print("Split created:")
print(f"   Original data: {total_original:,} samples")
print(f"   Training: {len(train_texts):,} samples ({len(train_texts)/total_original*100:.1f}%)")
print(f"   Validation: {len(val_texts):,} samples ({len(val_texts)/total_original*100:.1f}%)")
print(f"   Test: {len(test_texts):,} samples ({len(test_texts)/total_original*100:.1f}%)")
print(f"   Total after split: {total_split:,} samples")
print(f"   No data lost: {total_original == total_split}")


# Checking label distributions 

print("\nCHECKING LABEL DISTRIBUTIONS")
print( )


# This function checks the distributions of labels in the dataset
# It takes labels and split_name as arguments and returns percentage
# of neutral labels

def check_label_distribution(labels, split_name):
    off_count = sum(labels)  # Count 1s (offensive)
    neu_count = len(labels) - off_count  # Count 0s (neutral)
    neu_pct = neu_count / len(labels) * 100
    print(f"   {split_name}: {neu_count:,} neutral ({neu_pct:.1f}%), {off_count:,} offensive ({100-neu_pct:.1f}%)")
    return neu_pct

# Uses the check_label_distribution for training, validation and test
original_neu_pct = check_label_distribution(labels, "Original")
train_neu_pct = check_label_distribution(train_labels, "Training")
val_neu_pct = check_label_distribution(val_labels, "Validation")
test_neu_pct = check_label_distribution(test_labels, "Test")


# Verify that the stratification worked
print("\nSTRATIFICATION VERIFICATION")
print()

# Calculating percentage differences
train_diff = abs(train_neu_pct - original_neu_pct)
val_diff = abs(val_neu_pct - original_neu_pct)
test_diff = abs(test_neu_pct - original_neu_pct)

print(f"Stratification quality:")
print(f"   Training difference: {train_diff:.2f}% (should be < 1%)")
print(f"   Validation difference: {val_diff:.2f}% (should be < 1%)")
print(f"   Test difference: {test_diff:.2f}% (should be < 1%)")
print()



## ---- 1.4 Checking the length of text ----


## Try out different max_lengths later, maybe the model can perform well with shorter lengths

print("\n1.4 HANDLING DIFFERENT TEXT LENGTHS")
print()

# Check text lengths in data
print("Analyzing token lengths (sample of 1000 reviews)...")
sample_lengths = [len(tokenizer.encode(text)) for text in train_texts[:1000]]

print(f"Token length statistics:")
print(f"  Min: {min(sample_lengths)} tokens")
print(f"  Max: {max(sample_lengths)} tokens")
print(f"  Average: {np.mean(sample_lengths):.1f} tokens")
print(f"  85th percentile: {np.percentile(sample_lengths, 85):.0f} tokens")
print(f"  95th percentile: {np.percentile(sample_lengths, 95):.0f} tokens")
print()

print(f"Coverage with max_length={MAX_LENGTH}:")
covered = sum(1 for length in sample_lengths if length <= MAX_LENGTH)
coverage_pct = covered / len(sample_lengths) * 100
print(f"  {coverage_pct:.1f}% of texts will be fully captured")
print()

print("Common max_length choices:")
print("  128 tokens: Short texts, faster training")
print("  256 tokens: Medium texts, good balance")
print("  512 tokens: Long texts, BERT's maximum")
print()


## ---- 1.5 Creating a dataset class ----

# Creating dataset class - tokenization and encoding

print("\n1.5 CREATING THE DATASET CLASS")
print()

class TextClassificationDataset(Dataset):
    """
    Generic Dataset class for text classification.
    Handles tokenization and encoding for any text classification task.
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Get the text and label for this index
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,      # Cutting off if too long
            padding='max_length', # Padding if too short
            max_length=self.max_length,
            return_tensors='pt'   # Return PyTorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print("Dataset class defined!")
print()

print("What this class does:")
print("  - Takes raw text and converts to BERT input format")
print("  - Handles padding/truncation automatically")
print("  - Returns tensors ready for training")
print("  - Creates 'input_ids', 'attention_mask', and 'labels'")
print()



## ---- 1.6 Creating datasets for training and validation ----

print("\n1.6 CREATING TRAINING, VALIDATION AND TEST DATASETS")
print( )

train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

print(f"Datasets created!")
print(f"Training dataset: {len(train_dataset)} samples")
print(f"Validation dataset: {len(val_dataset)} samples")
print(f"Test dataset: {len(test_dataset)} samples")
print()


## ---- 1.7 Creating dataloaders ----

# We do this so that only 16 objects are processed in each batch, for memory management and smoother learning.
# Shuffling the training data so the model doesn't learn the order of objects

print("\n1.7 CREATING DATALOADERS")
print()


train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,    
    shuffle=True,      
    num_workers=NUM_WORKERS 
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=False,    
    num_workers=NUM_WORKERS 
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print(f"Dataloaders created:")
print(f"  Training: {len(train_loader)} batches")
print(f"  Validation: {len(val_loader)} batches")
print(f"  Test: {len(test_loader)} batches")
print(f"  Batch size: {BATCH_SIZE}")
print()


## ---- 1.8 Saving preprocessed data ----

print(f"\n1.8 SAVING PREPROCESSED DATA TO: {OUTPUT_DIR}")
print()

# Create configuration dictionary
print("CREATING CONFIGURATION DICTIONARY")
print()
config = {
    'dataset_name': DATASET_NAME,
    'model_name': MODEL_NAME,
    'max_length': MAX_LENGTH,
    'batch_size': BATCH_SIZE,
    'test_size': TEST_SIZE,
    'random_state': RANDOM_STATE,
    'train_size': len(train_dataset),
    'val_size': len(val_dataset),
    'test_size': len(test_dataset),
    'num_classes': 2,
    'class_names': ['neutral', 'offensive'],
    'train_neu_pct': train_neu_pct,
    'val_neu_pct': val_neu_pct,
    'test_neu_pct': test_neu_pct,
    'avg_token_length': np.mean(sample_lengths),
    'max_token_length': max(sample_lengths),
    'coverage_pct': coverage_pct,
    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
print("Configuration dictionary created successfully!")
print()

def save_essential_data(train_dataset, val_dataset, test_dataset,
                       tokenizer, config, output_dir):
    """Save only essential preprocessing data (no raw data or loaders)"""
   
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
   
    try:
        # Save datasets (essential)
        print("Saving datasets...")
        torch.save(train_dataset, f"{output_dir}/train_dataset.pt")
        torch.save(val_dataset, f"{output_dir}/val_dataset.pt")
        torch.save(test_dataset, f"{output_dir}/test_dataset.pt")
       
        # Save tokenizer (essential)
        print("Saving tokenizer...")
        tokenizer.save_pretrained(f"{output_dir}/tokenizer")
       
        # Save configuration (essential)
        print("Saving configuration...")
        with open(f"{output_dir}/config.json", "w") as f:
            json.dump(config, f, indent=2)
       
        # Create a symlink to latest (optional)
        latest_link = f"{BASE_OUTPUT_DIR}/latest"
        if os.path.exists(latest_link):
            os.remove(latest_link)
        try:
            os.symlink(os.path.basename(output_dir), latest_link)
        except OSError:
            # Symlinks might not work on Windows without admin privileges
            print("Note: Could not create symlink (this is normal on Windows)")
       
        print("Essential data saved successfully!")
        print(f"  Data location: {output_dir}")
        print(f"  Latest link: {latest_link}")
        print()
       
        # Save a README
        readme_content = f"""# Preprocessing Results

## Configuration
Dataset: {config['dataset_name']}
Model: {config['model_name']}
Max Length: {config['max_length']}
Batch Size: {config['batch_size']}
Test Size: {config['test_size']}
Random State: {config['random_state']}

## Data Splits
Training: {config['train_size']:,} samples
Validation: {config['val_size']:,} samples  
Test: {config['test_size']:,} samples

## Files Saved
- train_dataset.pt, val_dataset.pt, test_dataset.pt (tokenized datasets)
- tokenizer/ (saved tokenizer)
- config.json (configuration parameters)

## Note
DataLoaders can be recreated from datasets using:
```python
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

## Usage
Load this data in Part 2 (training) by setting:
INPUT_DIR = "{output_dir}"
"""
       
        with open(f"{output_dir}/README.md", "w") as f:
            f.write(readme_content)
       
        return True
       
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

# Call the function (Save essential data only)
success = save_essential_data(
    train_dataset, val_dataset, test_dataset,
    tokenizer, config, OUTPUT_DIR
)

if success:
    print("Essential data saved successfully!")
    print("Note: DataLoaders can be recreated from saved datasets when needed")
else:
    print("Failed to save data")
