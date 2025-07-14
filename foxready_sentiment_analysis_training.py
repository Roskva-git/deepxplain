# Activity 1: Fine-Tuning BERT and Applying LIME & SHAP for Sentiment Analysis

## Chapter 2: Training and Analysis

# Written by: Røskva
# Created: 04. July 2025
# Updated: 09. July 2025

# 2. TRAINING AND FINE-TUNING STEPS
### 2.0 Device setup and training configuration
### 2.1 Loading preprocessed data and setting up training
### 2.2 Training functions and optimization
### 2.3 Training loop and model fine-tuning
### 2.4 Model evaluation and testing
### 2.5 Saving trained model and results


## ---- 2.0 Device setup and training configuration ----

# Importing additional tools for training
import numpy as np
import pandas as pd
import torch
import json
import os

from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

# ==== FOX CLUSTER OPTIMIZATIONS ====
# Disable tokenizer parallelism warnings on cluster
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ==== CONFIGURATION FOR TRAINING ====

# Data loading configuration - SET YOUR FOX CLUSTER PATH HERE
INPUT_DIR = "/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/preprocessed_data/HateBRXplain_adalbertojunior_distilbert-portuguese-cased_maxlen512_20250709_0844"

# Training hyperparameters - optimized for Fox cluster GPUs
LEARNING_RATE = 2e-05
EPOCHS = 3  # BRhatexplain has 5, runs risk of overfitting
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16        # Increased for GPU efficiency
NUM_WORKERS = 4        # Parallel data loading on Fox

# Output directory for training results
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
TRAINING_OUTPUT_DIR = f"/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/training_results/training_{timestamp}"

print("FOX CLUSTER TRAINING CONFIGURATION:")
print(f"  Input directory: {INPUT_DIR}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Warmup steps: {WARMUP_STEPS}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Batch size: {BATCH_SIZE} (optimized for GPU)")
print(f"  Num workers: {NUM_WORKERS} (parallel data loading)")
print(f"  Output directory: {TRAINING_OUTPUT_DIR}")
print()

## ---- Device setup for Fox cluster ----

print("FOX CLUSTER DEVICE SETUP")
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Fox cluster GPU optimizations
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    print("CUDA optimizations enabled for Fox cluster")
else:
    print("WARNING: No GPU available! This will be very slow.")
    print("Make sure you requested GPU resources in your SLURM job.")
print()

## ---- 2.1 Loading preprocessed data and setting up training ----

print("\n2.1 LOADING PREPROCESSED DATA AND SETTING UP TRAINING")
print()

# Dataset class definition (needed for loading saved datasets)
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

# Load configuration from preprocessing
print("Loading configuration from preprocessing...")
print(f"INPUT_DIR is: {INPUT_DIR}")
print(f"Looking for config at: {INPUT_DIR}/config.json")
with open(f"{INPUT_DIR}/config.json", 'r') as f:
    config = json.load(f)

print(f"Configuration loaded:")
print(f"  Dataset: {config['dataset_name']}")
print(f"  Model: {config['model_name']}")
print(f"  Max length: {config['max_length']}")
print(f"  Batch size: {config['batch_size']}")
print(f"  Training samples: {config['train_size']:,}")
print(f"  Validation samples: {config['val_size']:,}")
print(f"  Test samples: {config['test_size']:,}")
print()

# Load datasets
print("Loading datasets...")
train_dataset = torch.load(f"{INPUT_DIR}/train_dataset.pt", weights_only=False)
val_dataset = torch.load(f"{INPUT_DIR}/val_dataset.pt", weights_only=False)
test_dataset = torch.load(f"{INPUT_DIR}/test_dataset.pt", weights_only=False)

print(f"Datasets loaded!")
print(f"Training dataset: {len(train_dataset)} samples")
print(f"Validation dataset: {len(val_dataset)} samples")
print(f"Test dataset: {len(test_dataset)} samples")
print()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(f"{INPUT_DIR}/tokenizer")

print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
print(f"Vocabulary size: {tokenizer.vocab_size:,} tokens")
print()

# Load base model for fine-tuning
print("Loading base model for fine-tuning...")
model = BertForSequenceClassification.from_pretrained(config['model_name'])

print(f"Model loaded: {model.__class__.__name__}")
print(f"Number of labels: 2 (neutral=0, offensive=1)")
print()

# Move model to device
model.to(device)
print(f"Model moved to {device}")
print()

# Create output directory
os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)
print(f"Training results will be saved to: {TRAINING_OUTPUT_DIR}")
print()

## ---- Creating dataloaders ----

# We do this so that only X objects are processed in each batch, for memory management and smoother learning.
# Shuffling the training data so the model doesn't learn the order of objects

print("CREATING DATALOADERS")
print()

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,    # Use Fox-optimized batch size
    shuffle=True,      
    num_workers=NUM_WORKERS,  # Parallel data loading on Fox
    pin_memory=True          # Faster GPU transfer
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=False,    
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"Dataloaders created for Fox cluster:")
print(f"  Training: {len(train_loader)} batches")
print(f"  Validation: {len(val_loader)} batches")
print(f"  Test: {len(test_loader)} batches")
print(f"  Batch size: {BATCH_SIZE} (GPU optimized)")
print(f"  Workers: {NUM_WORKERS} (parallel loading)")
print(f"  Pin memory: True (faster GPU transfer)")
print()

## ---- Training setup ----

print("Setting up optimizer and scheduler...")

# Setup optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)

print(f"Training setup complete:")
print(f"  Optimizer: AdamW")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Total steps: {total_steps:,}")
print(f"  Warmup steps: {WARMUP_STEPS}")
print()

## ---- 2.2 Training functions and optimization ----

print("\n2.2 DEFINING TRAINING FUNCTIONS")
print()

def train_epoch(model, loader, optimizer, scheduler, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(loader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device with non_blocking for efficiency
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability (Fox cluster recommendation)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        # Track loss
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Clear cache periodically on GPU
        if torch.cuda.is_available() and (len(progress_bar) % 100 == 0):
            torch.cuda.empty_cache()
    
    return total_loss / len(loader)

def evaluate_model(model, loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    avg_loss = total_loss / len(loader)
    
    return avg_loss, accuracy, f1, all_predictions, all_labels

print("Training functions defined!")
print()

print("What these functions do:")
print("  - train_epoch(): Trains model for one complete epoch")
print("  - evaluate_model(): Evaluates model and returns metrics")
print("  - Both use progress bars to show training progress")
print("  - Both handle device placement automatically")
print()

## ---- 2.3 Training loop and model fine-tuning ----

print("\n2.3 STARTING TRAINING LOOP")
print()

# Track training history
training_history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'val_f1': []
}

best_val_accuracy = 0
best_model_state = None

print("Starting training...")
start_time = datetime.now()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 50)
    
    # Training
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    
    # Validation
    val_loss, val_accuracy, val_f1, val_predictions, val_labels = evaluate_model(model, val_loader, device)
    
    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict().copy()
        print(f"✓ New best model saved! Accuracy: {val_accuracy:.4f}")
    
    # Record history
    training_history['train_loss'].append(train_loss)
    training_history['val_loss'].append(val_loss)
    training_history['val_accuracy'].append(val_accuracy)
    training_history['val_f1'].append(val_f1)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy:.4f}")
    print(f"Val F1: {val_f1:.4f}")

# Load best model
model.load_state_dict(best_model_state)

end_time = datetime.now()
training_time = end_time - start_time

print(f"\nTraining completed!")
print(f"Training time: {training_time}")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print()

## ---- 2.4 Model evaluation and testing ----

print("\n2.4 FINAL MODEL EVALUATION")
print()

# Final test evaluation
test_loss, test_accuracy, test_f1, test_predictions, test_labels = evaluate_model(model, test_loader, device)

print(f"Final Test Results:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Test F1: {test_f1:.4f}")
print()

# Detailed classification report
print("Detailed Classification Report:")
print(classification_report(test_labels, test_predictions, target_names=['Neutral', 'Offensive']))
print()

## ---- 2.5 Saving trained model and results ----

print("\n2.5 SAVING TRAINED MODEL AND RESULTS")
print()

# Create configuration dictionary for saving
print("Creating results dictionary...")
results_config = {
    'training_completed_at': end_time.isoformat(),
    'training_duration': str(training_time),
    'best_val_accuracy': best_val_accuracy,
    'test_accuracy': test_accuracy,
    'test_f1': test_f1,
    'test_loss': test_loss,
    'training_config': {
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'warmup_steps': WARMUP_STEPS,
        'weight_decay': WEIGHT_DECAY
    },
    'data_config': config,
    'training_history': training_history,
    'classification_report': classification_report(test_labels, test_predictions, 
                                                 target_names=['Neutral', 'Offensive'], 
                                                 output_dict=True)
}
print("Results dictionary created successfully!")
print()

# Save the trained model
print("Saving trained model...")
model_save_path = f"{TRAINING_OUTPUT_DIR}/trained_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Save training history
print("Saving training history...")
with open(f"{TRAINING_OUTPUT_DIR}/training_history.json", 'w') as f:
    json.dump(training_history, f, indent=2)

# Save comprehensive results
print("Saving comprehensive results...")
with open(f"{TRAINING_OUTPUT_DIR}/results.json", 'w') as f:
    json.dump(results_config, f, indent=2)

print("Model and results saved successfully!")
print(f"  Model saved to: {model_save_path}")
print(f"  Training history: {TRAINING_OUTPUT_DIR}/training_history.json")
print(f"  Results: {TRAINING_OUTPUT_DIR}/results.json")
print()