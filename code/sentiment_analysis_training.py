# Activity 1: Fine-Tuning BERT and Applying LIME & SHAP for Sentiment Analysis

## Chapter 2: Training and Analysis

# Written by: Røskva
# Created: 04. July 2025
# Updated: 23. July 2025

# 2. TRAINING AND FINE-TUNING STEPS
### 2.0 Device setup and training configuration
### 2.1 Loading preprocessed data and setting up training
### 2.2 Training functions and optimization
### 2.3 Attention extraction and alignment metrics
### 2.4 Training loop and model fine-tuning
### 2.5 Model evaluation and testing
### 2.6 Saving trained model and results


## ---- 2.0 Device setup and training configuration ----

# Importing additional tools for training
import numpy as np
import pandas as pd
import torch
import json
import os
import re

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
INPUT_DIR = "/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/preprocessed_data/HateBRXplain_no_emojis.json_adalbertojunior_distilbert-portuguese-cased_maxlen512_20250722_1117_rationale_supervised"

# Training hyperparameters - optimized for Fox cluster GPUs
LEARNING_RATE = 2e-05
EPOCHS = 3  # BRhatexplain has 5, runs risk of overfitting
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16        # Increased for GPU efficiency
NUM_WORKERS = 4        # Parallel data loading on Fox

# Output directory for training results
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
TRAINING_OUTPUT_DIR = f"/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/training_results/no_emoji/training_{timestamp}"

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


def create_rationale_labels(text, rationale_1, rationale_2, tokenizer, max_length=512):
    """Map human rationales to token-level attention labels"""
    
    # Tokenize the text to get the exact tokens BERT will see
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    rationale_labels = [0] * len(tokens)  # 0 = not important
    
    # Extract human rationale words
    human_words = set()
    if rationale_1 and rationale_1 != 'N/A':
        words = re.findall(r'\b\w+\b', rationale_1.lower())
        human_words.update(words)
    if rationale_2 and rationale_2 != 'N/A':
        words = re.findall(r'\b\w+\b', rationale_2.lower())
        human_words.update(words)
    
    # Map human words to BERT tokens
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        
        # Clean token (remove ## for subwords)
        clean_token = token.replace('##', '').lower()
        
        # If this token matches any human rationale word
        if clean_token in human_words:
            rationale_labels[i] = 1
        
        # Also check if token is part of a rationale word
        for human_word in human_words:
            if clean_token in human_word or human_word in clean_token:
                rationale_labels[i] = 1
    
    return rationale_labels


# Dataset class definition (needed for loading saved datasets)

class RationaleTextDataset(Dataset):
    """
    Dataset class with rationale supervision for text classification.
    """
    def __init__(self, texts, labels, rationales_1, rationales_2, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.rationales_1 = rationales_1
        self.rationales_2 = rationales_2
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Standard tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create rationale labels
        rationale_labels = create_rationale_labels(
            text, 
            self.rationales_1[idx], 
            self.rationales_2[idx], 
            self.tokenizer, 
            self.max_length
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'rationale_labels': torch.tensor(rationale_labels, dtype=torch.float)
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

# RATIONALE WEIGHT = ALPHA

def train_epoch_with_rationales(model, loader, optimizer, scheduler, device, rationale_weight=10):
    """Train the model for one epoch with rationale supervision"""
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_rationale_loss = 0
    
    progress_bar = tqdm(loader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        rationale_labels = batch['rationale_labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True)
        
        # Classification loss
        classification_loss = outputs.loss
        
        # Rationale supervision loss
        # Get attention from last layer, first head 
        # We can experiment with this. Layer 8, head 7 has best performance in BRhatexplain.
        attention_weights = outputs.attentions[-1][:, 0, :, :]  # [batch, seq_len, seq_len]
        # Average over the sequence dimension to get token importance
        token_attention = attention_weights.mean(dim=1)  # [batch, seq_len]
        
        # MSE loss between model attention and human rationales
        rationale_loss = torch.nn.functional.mse_loss(token_attention, rationale_labels)
        
        # Combined loss
        total_batch_loss = classification_loss + rationale_weight * rationale_loss
        
        # Backward pass
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Track losses
        total_loss += total_batch_loss.item()
        total_class_loss += classification_loss.item()
        total_rationale_loss += rationale_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'total_loss': total_batch_loss.item(),
            'class_loss': classification_loss.item(),
            'rat_loss': rationale_loss.item()
        })
    
    return (total_loss / len(loader), 
            total_class_loss / len(loader), 
            total_rationale_loss / len(loader))

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


## ---- 2.3 Attention extraction and alignment metrics ----

print("\n2.3 DEFINING ATTENTION EXTRACTION FUNCTIONS")
print()

def extract_attention_and_evaluate(model, dataset, tokenizer, device, layer=8, head=7):
    """Extract attention weights and compute alignment metrics"""
    model.eval()
    
    all_attentions = []
    all_rationale_labels = []
    all_predictions = []
    all_labels = []
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Batch size 1 for easier processing
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting attention"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            rationale_labels = batch['rationale_labels']
            
            # Get model outputs with attention
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                           labels=labels, output_attentions=True)
            
            # Extract attention from specific layer and head (CLS token to all tokens)
            attention = outputs.attentions[layer-1][0, head, 0, :].cpu().numpy()  # CLS token attention
            
            # Get valid tokens (not padding)
            valid_length = attention_mask.sum().item()
            attention = attention[:valid_length]
            rationale = rationale_labels[0][:valid_length].numpy()
            
            all_attentions.append(attention)
            all_rationale_labels.append(rationale)
            all_predictions.append(torch.argmax(outputs.logits).item())
            all_labels.append(labels.item())
    
    return all_attentions, all_rationale_labels, all_predictions, all_labels

def compute_alignment_metrics(attentions, rationale_labels):
    """Compute IOU F1, Token F1, and other alignment metrics"""
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    all_attention_binary = []
    all_rationale_binary = []
    
    for attention, rationale in zip(attentions, rationale_labels):
        # Convert attention to binary (top 20% of tokens)
        if len(attention) > 0:
            threshold = np.percentile(attention, 80)
            attention_binary = (attention >= threshold).astype(int)
            
            all_attention_binary.extend(attention_binary)
            all_rationale_binary.extend(rationale[:len(attention_binary)])
    
    # Compute metrics
    if len(all_rationale_binary) > 0 and sum(all_rationale_binary) > 0:
        token_f1 = f1_score(all_rationale_binary, all_attention_binary, zero_division=0)
        token_precision = precision_score(all_rationale_binary, all_attention_binary, zero_division=0)
        token_recall = recall_score(all_rationale_binary, all_attention_binary, zero_division=0)
    else:
        token_f1 = token_precision = token_recall = 0.0
    
    # IOU F1 (intersection over union for each example)
    iou_scores = []
    for attention, rationale in zip(attentions, rationale_labels):
        if len(attention) > 0:
            # Convert to binary
            threshold = np.percentile(attention, 80)
            attention_binary = (attention >= threshold).astype(int)
            rationale_binary = rationale[:len(attention_binary)]
            
            # Compute IOU
            intersection = np.sum(attention_binary * rationale_binary)
            union = np.sum((attention_binary + rationale_binary) > 0)
            
            if union > 0:
                iou_scores.append(intersection / union)
            else:
                iou_scores.append(0.0)
    
    iou_f1 = np.mean(iou_scores) if iou_scores else 0.0
    
    return {
        'iou_f1': iou_f1,
        'token_f1': token_f1,
        'token_precision': token_precision,
        'token_recall': token_recall
    }

def evaluate_with_attention_metrics(model, test_dataset, tokenizer, device):
    """Complete evaluation including attention alignment"""
    
    # Standard classification metrics
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    avg_loss, accuracy, f1, predictions, labels = evaluate_model(model, test_loader, device)
    
    # Attention alignment metrics
    print("Extracting attention weights for alignment metrics...")
    attentions, rationale_labels, _, _ = extract_attention_and_evaluate(
        model, test_dataset, tokenizer, device
    )
    
    alignment_metrics = compute_alignment_metrics(attentions, rationale_labels)
    
    # Combine all metrics
    results = {
        'test_accuracy': accuracy,
        'test_f1': f1,
        'test_loss': avg_loss,
        'iou_f1': alignment_metrics['iou_f1'],
        'token_f1': alignment_metrics['token_f1'],
        'token_precision': alignment_metrics['token_precision'],
        'token_recall': alignment_metrics['token_recall']
    }
    
    print(f"Attention alignment metrics computed:")
    print(f"  IOU F1: {alignment_metrics['iou_f1']:.3f}")
    print(f"  Token F1: {alignment_metrics['token_f1']:.3f}")
    print(f"  Token Precision: {alignment_metrics['token_precision']:.3f}")
    print(f"  Token Recall: {alignment_metrics['token_recall']:.3f}")
    
    return results

print("Attention extraction functions defined!")
print()



## ---- 2.4 Training loop and model fine-tuning ----

print("\n2.4 STARTING TRAINING LOOP")
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
    train_loss, class_loss, rat_loss = train_epoch_with_rationales(model, train_loader, optimizer, scheduler, device)
    print(f"Train Loss: {train_loss:.4f} (Class: {class_loss:.4f}, Rationale: {rat_loss:.4f})")
    
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

## ---- 2.5 Model evaluation and testing ----

print("\n2.5 FINAL MODEL EVALUATION")
print()

# Final test evaluation with attention metrics
results = evaluate_with_attention_metrics(model, test_dataset, tokenizer, device)
test_loss, test_accuracy, test_f1 = results['test_loss'], results['test_accuracy'], results['test_f1']

# For the classification report, we still need predictions and labels
_, _, _, test_predictions, test_labels = evaluate_model(model, test_loader, device)

print(f"Final Test Results:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Test F1: {test_f1:.4f}")
print()

# Detailed classification report
print("Detailed Classification Report:")
print(classification_report(test_labels, test_predictions, target_names=['Neutral', 'Offensive']))
print()

## ---- 2.6 Saving trained model and results ----

print("\n2.6 SAVING TRAINED MODEL AND RESULTS")
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
    'iou_f1': results['iou_f1'],
    'token_f1': results['token_f1'],
    'token_precision': results['token_precision'],
    'token_recall': results['token_recall'],
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
