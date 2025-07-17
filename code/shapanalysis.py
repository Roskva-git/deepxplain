# Activity 1: Fine-Tuning BERT and Applying LIME & SHAP for Sentiment Analysis

## Chapter 4: SHAP Analysis

# Written by: Røskva
# Created: 17. July 2025
# Purpose: SHAP explainability analysis for offensive/non-offensive classification
# Dataset: HateBRXplain_no_emojis.json (emoji-free version)

# 4. SHAP ANALYSIS STEPS
### 4.0 Environment setup and configuration
### 4.1 Loading trained model and preprocessed data
### 4.2 Creating SHAP-compatible prediction function
### 4.3 Setting up SHAP explainer with memory optimization
### 4.4 Generating SHAP explanations
### 4.5 Analyzing and visualizing results
### 4.6 Saving SHAP results


## ---- 4.0 Environment setup and configuration ----

# Importing libraries and tools

import shap
import json
import torch
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt

from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# SHAP Analysis Configuration
print("SHAP ANALYSIS FOR EMOJI-FREE DATASET")
print("=" * 50)
print()

# Dataset and model configuration - MUST match training setup exactly
MODEL_NAME = 'google-bert/bert-base-uncased'  # Change every time
DATASET_NAME = "HateBRXplain_no_emojis.json"
MAX_LENGTH = 512  # Keep same as LIME analysis for comparability
BATCH_SIZE = 1    # Small batch size for SHAP memory management
RANDOM_STATE = 42

# Paths
TRAINED_MODEL_PATH = "/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/training_results/no_emoji/training_20250716_1027/trained_model"
PREPROCESSED_DATA_PATH = "/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/preprocessed_data/no_emoji/HateBRXplain_no_emojis.json_google-bert_bert-base-uncased_maxlen512_20250716_1014"

# Output directory for SHAP results
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
BASE_OUTPUT_DIR = "/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/shap_results"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"shap_{MODEL_NAME.replace('/', '_')}_no_emoji_{timestamp}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("FOX CLUSTER SHAP CONFIGURATION:")
print(f"  Model: {MODEL_NAME}")
print(f"  Dataset: {DATASET_NAME}")
print(f"  Max length: {MAX_LENGTH} tokens (same as LIME)")
print(f"  Batch size: {BATCH_SIZE} (memory optimized)")
print(f"  Trained model: {TRAINED_MODEL_PATH}")
print(f"  Preprocessed data: {PREPROCESSED_DATA_PATH}")
print(f"  Output directory: {OUTPUT_DIR}")
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
    
    # Clear GPU cache to start fresh
    torch.cuda.empty_cache()
    print("GPU cache cleared for SHAP analysis")
else:
    print("WARNING: No GPU available! SHAP will be very slow.")
    print("Make sure you requested GPU resources in your SLURM job.")
print()

## ---- 4.1 Loading trained model and preprocessed data ----

print("\n4.1 LOADING TRAINED MODEL AND PREPROCESSED DATA")
print()

# Load the trained model and tokenizer
print("Loading trained model...")
print(f"Model path: {TRAINED_MODEL_PATH}")

try:
    model = BertForSequenceClassification.from_pretrained(TRAINED_MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(TRAINED_MODEL_PATH)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check the TRAINED_MODEL_PATH in the configuration section")
    exit(1)

print(f"Model loaded: {model.__class__.__name__}")
print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
print(f"Vocabulary size: {tokenizer.vocab_size:,} tokens")
print()

# Move model to device and set to evaluation mode
model.to(device)
model.eval()
print(f"Model moved to {device} and set to evaluation mode")
print()

# Load preprocessed data configuration
print("Loading preprocessed data configuration...")
print(f"Config path: {PREPROCESSED_DATA_PATH}/config.json")

try:
    with open(f"{PREPROCESSED_DATA_PATH}/config.json", 'r') as f:
        config = json.load(f)
    print("Configuration loaded successfully!")
except Exception as e:
    print(f"Error loading config: {e}")
    print("Please check the PREPROCESSED_DATA_PATH in the configuration section")
    exit(1)

print(f"Configuration details:")
print(f"  Dataset: {config['dataset_name']}")
print(f"  Original model: {config['model_name']}")
print(f"  Max length: {config['max_length']}")
print(f"  Training samples: {config['train_size']:,}")
print(f"  Validation samples: {config['val_size']:,}")
print(f"  Test samples: {config['test_size']:,}")
print()

# Dataset class definition (needed for loading saved datasets)
class TextClassificationDataset(Dataset):
    """
    Same dataset class used in preprocessing and training
    Handles tokenization and encoding for text classification
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print("Dataset class defined (same as preprocessing and training)")
print()

# Load test dataset for SHAP analysis
print("Loading test dataset for SHAP analysis...")
try:
    test_dataset = torch.load(f"{PREPROCESSED_DATA_PATH}/test_dataset.pt", weights_only=False)
    print(f"Test dataset loaded: {len(test_dataset)} samples")
except Exception as e:
    print(f"Error loading test dataset: {e}")
    exit(1)

# Extract texts and labels from test dataset for SHAP
print("Extracting texts and labels from test dataset...")
test_texts = []
test_labels = []

for i in range(len(test_dataset)):
    item = test_dataset[i]
    # Get the original text (before tokenization)
    test_texts.append(test_dataset.texts[i])
    test_labels.append(test_dataset.labels[i])

print(f"Extracted {len(test_texts)} texts and {len(test_labels)} labels")
print()

# Display sample data
print("Sample data from test set:")
print(f"Text example: {test_texts[0][:200]}...")
print(f"Label example: {test_labels[0]} (0=neutral, 1=offensive)")
print()

# Check label distribution
neutral_count = test_labels.count(0)
offensive_count = test_labels.count(1)
print(f"Test set label distribution:")
print(f"  Neutral: {neutral_count} samples ({neutral_count/len(test_labels)*100:.1f}%)")
print(f"  Offensive: {offensive_count} samples ({offensive_count/len(test_labels)*100:.1f}%)")
print()

## ---- 4.2 Creating SHAP-compatible prediction function ----

print("\n4.2 CREATING SHAP-COMPATIBLE PREDICTION FUNCTION")
print()

def create_shap_prediction_function(model, tokenizer, device, max_length=512):
    """
    Create a robust prediction function that SHAP can use
    
    This function handles various input formats that SHAP might send
    """
    def predict_fn(texts):
        """
        Robust prediction function for SHAP
        
        Args:
            texts: Can be a single string, list of strings, or numpy array
            
        Returns:
            numpy array of shape (n_samples, n_classes) with probabilities
        """
        model.eval()
        predictions = []
        
        # Handle different input types that SHAP might send
        if isinstance(texts, str):
            # Single string input
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            # Convert numpy array to list
            texts = texts.tolist()
        elif not isinstance(texts, list):
            # Try to convert to list
            try:
                texts = list(texts)
            except:
                # If all else fails, wrap in a list
                texts = [str(texts)]
        
        # Ensure all items are strings
        texts = [str(text) for text in texts]
        
        # Process each text
        for text in texts:
            try:
                # Clean the text
                text = text.strip()
                if not text:  # Handle empty strings
                    # Return neutral prediction for empty text
                    predictions.append([0.5, 0.5])
                    continue
                
                # Tokenize exactly the same way as training
                inputs = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get model predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Convert logits to probabilities
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predictions.append(probs.cpu().numpy()[0])
                    
            except Exception as e:
                print(f"Error processing text: '{text[:50]}...'")
                print(f"Error details: {e}")
                # Return neutral prediction as fallback
                predictions.append([0.5, 0.5])
        
        return np.array(predictions)
    
    return predict_fn

# Create the prediction function
print("Creating SHAP prediction function...")
predict_fn = create_shap_prediction_function(model, tokenizer, device, MAX_LENGTH)

print("SHAP prediction function created!")
print()

print("What this function does:")
print("  - Takes raw text strings as input")
print("  - Tokenizes them using the same process as training")
print("  - Runs them through your trained model")
print("  - Returns class probabilities [P(neutral), P(offensive)]")
print("  - Processes texts one by one to manage memory")
print()

# Test the prediction function
print("Testing prediction function with sample texts...")
test_sample = test_texts[:3]  # Test with 3 examples
test_predictions = predict_fn(test_sample)

print(f"Test results:")
for i, (text, pred) in enumerate(zip(test_sample, test_predictions)):
    predicted_class = "Offensive" if pred[1] > pred[0] else "Neutral"
    confidence = max(pred[0], pred[1])
    print(f"  {i+1}. Text: {text[:50]}...")
    print(f"     Prediction: {predicted_class} (confidence: {confidence:.3f})")
    print(f"     Probabilities: [Neutral: {pred[0]:.3f}, Offensive: {pred[1]:.3f}]")

print()

## ---- 4.3 Setting up SHAP explainer with memory optimization ----

print("\n4.3 SETTING UP SHAP EXPLAINER WITH MEMORY OPTIMIZATION")
print()

# Select background dataset for SHAP
print("Selecting background dataset for SHAP...")
print()

print("Background dataset explanation:")
print("  - SHAP needs a 'background' dataset to compute feature importance")
print("  - This represents the 'typical' input the model sees")
print("  - Smaller background = faster computation but less stable explanations")
print("  - We're using a stratified sample from the test set")
print()

# Create stratified background sample
neutral_texts = [text for text, label in zip(test_texts, test_labels) if label == 0]
offensive_texts = [text for text, label in zip(test_texts, test_labels) if label == 1]

# Take equal numbers from each class for background
background_size = 20  # Total background size
per_class = background_size // 2

background_texts = (
    neutral_texts[:per_class] + 
    offensive_texts[:per_class]
)

print(f"Background dataset created:")
print(f"  Total size: {len(background_texts)} texts")
print(f"  Neutral examples: {per_class}")
print(f"  Offensive examples: {per_class}")
print()


# Create SHAP explainer with memory optimization and multiple fallback methods
print("Creating SHAP explainer with memory optimization...")
print("This may take a few minutes as SHAP initializes...")

try:
    # Method 1: Try with current background size (20 texts)
    print("Trying Method 1: Standard explainer with background...")
    explainer = shap.Explainer(predict_fn, background_texts)
    print("✓ Standard SHAP explainer created successfully!")
    
    # Test the explainer with a single example
    print("Testing explainer with single example...")
    test_text = test_texts[0]
    test_shap = explainer([test_text])
    print("✓ Explainer test successful!")
    
except Exception as e:
    print(f"Method 1 failed: {e}")
    print("Trying Method 2: Reduced background size...")
    
    try:
        # Method 2: Reduce background size for memory
        small_background = background_texts[:10]  # Reduce from 20 to 10
        explainer = shap.Explainer(predict_fn, small_background)
        print("✓ Reduced background explainer created successfully!")
        
        # Test the explainer
        print("Testing reduced background explainer...")
        test_text = test_texts[0]
        test_shap = explainer([test_text])
        print("✓ Reduced background explainer test successful!")
        
    except Exception as e2:
        print(f"Method 2 failed: {e2}")
        print("Trying Method 3: Masker-based approach...")
        
        try:
            # Method 3: Use text masker (no background needed)
            masker = shap.maskers.Text(tokenizer, mask_token="[MASK]")
            explainer = shap.Explainer(predict_fn, masker)
            print("✓ Masker-based SHAP explainer created successfully!")
            
            # Test the explainer
            print("Testing masker-based explainer...")
            test_text = test_texts[0]
            test_shap = explainer([test_text])
            print("✓ Masker-based explainer test successful!")
            
        except Exception as e3:
            print(f"Method 3 failed: {e3}")
            print("Trying Method 4: Minimal background...")
            
            try:
                # Method 4: Very small background
                minimal_background = background_texts[:5]  # Just 5 examples
                explainer = shap.Explainer(predict_fn, minimal_background)
                print("✓ Minimal background explainer created successfully!")
                
                # Test the explainer
                print("Testing minimal background explainer...")
                test_text = test_texts[0]
                test_shap = explainer([test_text])
                print("✓ Minimal background explainer test successful!")
                
            except Exception as e4:
                print(f"All methods failed. Last error: {e4}")
                print("This suggests a fundamental compatibility issue.")
                print("Check SHAP version and BERT model compatibility.")
                exit(1)

print()
print("SHAP explainer setup complete!")


# Helper function to extract word importance from SHAP values
# This function mimics LIME's word importance output

def extract_word_importance(shap_values, text, top_n=10):
    """
    Extract word-level importance from SHAP values
    Similar to LIME's word importance output
    """
    try:
        # Get the SHAP values for the offensive class (class 1)
        values = shap_values.values[0][:, 1]  # offensive class
        data = shap_values.data[0]
        
        # Create word-importance pairs
        word_importance = []
        for i, (word, importance) in enumerate(zip(data, values)):
            if word.strip() and word not in ['[CLS]', '[SEP]', '[PAD]']:
                word_importance.append({
                    'word': word,
                    'importance': float(importance),
                    'position': i
                })
        
        # Sort by absolute importance
        word_importance.sort(key=lambda x: abs(x['importance']), reverse=True)
        
        return word_importance[:top_n]
    
    except Exception as e:
        print(f"Error extracting word importance: {e}")
        return []

def create_shap_word_plot(word_importance, text, true_label, predicted_label, confidence, filename):
    """
    Create a word importance plot similar to LIME
    """
    if not word_importance:
        return False
    
    # Extract words and importance scores
    words = [item['word'] for item in word_importance]
    importances = [item['importance'] for item in word_importance]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color bars based on importance (red for offensive, blue for neutral)
    colors = ['red' if imp > 0 else 'blue' for imp in importances]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(words)), importances, color=colors, alpha=0.7)
    
    # Customize the plot
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel('SHAP Importance Score')
    ax.set_title(f'SHAP Word Importance\n'
                f'Text: {text[:100]}...\n'
                f'True: {true_label}, Predicted: {predicted_label} (conf: {confidence:.3f})')
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add legend
    ax.text(0.02, 0.98, 'Red: Pushes toward Offensive\nBlue: Pushes toward Neutral', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True


## ---- 4.4 Generating SHAP explanations ----

print("\n4.4 GENERATING SHAP EXPLANATIONS")
print()

# Select ALL OFFENSIVE examples for SHAP analysis (to match LIME analysis)
print("Selecting ALL offensive examples for SHAP analysis...")

# Get all offensive examples from test set
offensive_examples = [(text, label) for text, label in zip(test_texts, test_labels) if label == 1]

print(f"Found {len(offensive_examples)} offensive examples in test set")
print("This matches your LIME analysis scope for proper comparison")
print()

# Generate SHAP explanations for ALL offensive examples
print("Generating SHAP explanations for all offensive examples...")
print("This will take 20-30 minutes. Progress will be shown below.")
print()

shap_results = []
total_offensive = len(offensive_examples)

for i, (text, true_label) in enumerate(offensive_examples):
    print(f"Processing offensive example {i+1}/{total_offensive}...")
    print(f"  Text: {text[:80]}...")
    
    try:
        # Clear GPU cache periodically
        if i % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get SHAP values
        shap_values = explainer([text])
        
        # Get model prediction for this text
        prediction = predict_fn([text])[0]
        predicted_class = 1 if prediction[1] > prediction[0] else 0
        confidence = max(prediction[0], prediction[1])
        
        # Store results
        result = {
            'text': text,
            'true_label': true_label,
            'predicted_label': predicted_class,
            'prediction_probs': prediction.tolist(),
            'confidence': confidence,
            'shap_values': shap_values,
            'correct_prediction': true_label == predicted_class
        }
        
        shap_results.append(result)
        
        print(f"  Prediction: {'Offensive' if predicted_class == 1 else 'Neutral'} (confidence: {confidence:.3f})")
        print(f"  Correct: {true_label == predicted_class}")
        print(f"  ✓ SHAP analysis complete")
        
        # Progress indicator
        if (i + 1) % 25 == 0:
            print(f"\n--- Progress: {i+1}/{total_offensive} completed ({(i+1)/total_offensive*100:.1f}%) ---\n")
        
    except Exception as e:
        print(f"  ✗ Error processing example {i+1}: {e}")
        print(f"  Skipping this example...")
        continue

print(f"\nSHAP analysis complete! Successfully processed {len(shap_results)}/{total_offensive} offensive examples.")
print(f"Success rate: {len(shap_results)/total_offensive*100:.1f}%")
print()


## ---- 4.5 Analyzing and visualizing results ----

print("\n4.5 ANALYZING AND VISUALIZING RESULTS")
print()

# Summary statistics
correct_predictions = sum(1 for r in shap_results if r['correct_prediction'])
accuracy = correct_predictions / len(shap_results) if shap_results else 0

print(f"Analysis summary:")
print(f"  Total examples analyzed: {len(shap_results)}")
print(f"  Correct predictions: {correct_predictions}")
print(f"  Accuracy: {accuracy:.3f}")
print()

# Extract word-level importance for each example
print("Extracting word-level importance from SHAP values...")
for i, result in enumerate(shap_results):
    word_importance = extract_word_importance(result['shap_values'], result['text'])
    result['word_importance'] = word_importance

# Select 3 offensive examples for visualization
print("Selecting 3 offensive examples for visualization...")

# Find offensive examples (label 1) - prioritize correct predictions
offensive_correct = [i for i, r in enumerate(shap_results) if r['true_label'] == 1 and r['correct_prediction']]
offensive_incorrect = [i for i, r in enumerate(shap_results) if r['true_label'] == 1 and not r['correct_prediction']]

selected_indices = []

# Select 3 offensive examples (prefer correct predictions)
all_offensive = offensive_correct + offensive_incorrect
for i in range(min(3, len(all_offensive))):
    selected_indices.append(all_offensive[i])

print(f"Selected {len(selected_indices)} offensive examples for visualization")

# Create visualizations directory
viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(viz_dir, exist_ok=True)

# Generate visualizations for selected examples
print("Creating SHAP visualizations for selected examples...")
for i, idx in enumerate(selected_indices):
    result = shap_results[idx]
    
    if result['word_importance']:
        true_label = 'Offensive' if result['true_label'] == 1 else 'Neutral'
        pred_label = 'Offensive' if result['predicted_label'] == 1 else 'Neutral'
        
        filename = os.path.join(viz_dir, f"shap_example_{i+1}.png")
        
        success = create_shap_word_plot(
            result['word_importance'],
            result['text'],
            true_label,
            pred_label,
            result['confidence'],
            filename
        )
        
        if success:
            print(f"  ✓ Created visualization {i+1}: {filename}")
            print(f"    Text: {result['text'][:80]}...")
            print(f"    True: {true_label}, Predicted: {pred_label}")
        else:
            print(f"  ✗ Failed to create visualization {i+1}")

print(f"Visualizations saved to: {viz_dir}")
print()

# Display the 3 selected examples
print("SELECTED EXAMPLES FOR ANALYSIS:")
print("=" * 60)

for i, idx in enumerate(selected_indices):
    result = shap_results[idx]
    print(f"\nExample {i+1}:")
    print(f"Text: {result['text']}")
    print(f"True Label: {'Offensive' if result['true_label'] == 1 else 'Neutral'}")
    print(f"Predicted: {'Offensive' if result['predicted_label'] == 1 else 'Neutral'}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Correct: {result['correct_prediction']}")
    
    print(f"Top 8 important words:")
    for j, word_info in enumerate(result['word_importance'][:8]):
        direction = "OFFENSIVE" if word_info['importance'] > 0 else "NEUTRAL"
        print(f"  {j+1}. '{word_info['word']}': {word_info['importance']:.3f} → {direction}")
    print("-" * 60)



## ---- 4.6 Computing evaluation metrics ----

print("\n4.6 COMPUTING EVALUATION METRICS")
print()

# Load human rationales (keep the same function as before)
def load_human_rationales():
    """Load human rationales from the HateBRXplain dataset"""
    try:
        dataset_file = "/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/HateBRXplain_no_emojis.json"
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        human_rationales = {}
        
        for item in data:
            text = item['text']
            if item['label'] == 1:  # Only offensive samples have rationales
                rationale_words = []
                
                if 'human_rationale_1' in item and item['human_rationale_1']:
                    rationale_1 = item['human_rationale_1'].split(';')
                    rationale_words.extend([r.strip() for r in rationale_1 if r.strip()])
                
                if 'human_rationale_2' in item and item['human_rationale_2']:
                    rationale_2 = item['human_rationale_2'].split(';')
                    rationale_words.extend([r.strip() for r in rationale_2 if r.strip()])
                
                all_rationale_words = set()
                for rationale in rationale_words:
                    words = rationale.lower().split()
                    all_rationale_words.update(words)
                
                if all_rationale_words:
                    human_rationales[text] = list(all_rationale_words)
        
        return human_rationales
    except Exception as e:
        print(f"Error loading human rationales: {e}")
        return {}

print("Loading human rationales...")
human_rationales = load_human_rationales()
print(f"Loaded human rationales for {len(human_rationales)} offensive examples")

print("Computing evaluation metrics on all offensive examples...")

# Compute metrics (simplified versions)
def compute_all_metrics(shap_results, human_rationales, predict_fn):
    """Compute all metrics in one efficient pass"""
    
    # Initialize metric lists
    comprehensiveness_scores = []
    sufficiency_scores = []
    precisions = []
    recalls = []
    f1_scores = []
    iou_f1_scores = []
    overlap_percentages = []
    agreement_count = 0
    coverage_count = 0
    
    print(f"Computing metrics for {len(shap_results)} examples...")
    
    for i, result in enumerate(shap_results):
        text = result['text']
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(shap_results)} examples...")
        
        try:
            # --- Faithfulness Metrics ---
            original_prob = result['prediction_probs'][1]
            important_words = [w['word'] for w in result['word_importance'][:5] if w['importance'] > 0]
            
            if important_words:
                # Comprehensiveness
                text_without_important = text
                for word in important_words:
                    text_without_important = text_without_important.replace(word, '')
                text_without_important = ' '.join(text_without_important.split())
                
                if text_without_important.strip():
                    new_prob = predict_fn([text_without_important])[0][1]
                    comprehensiveness_scores.append(original_prob - new_prob)
                
                # Sufficiency
                important_text = ' '.join(important_words)
                if important_text.strip():
                    sufficient_prob = predict_fn([important_text])[0][1]
                    sufficiency = sufficient_prob / original_prob if original_prob > 0 else 0
                    sufficiency_scores.append(sufficiency)
            
            # --- Plausibility Metrics ---
            if text in human_rationales:
                coverage_count += 1
                
                shap_words = set(w['word'].lower() for w in result['word_importance'][:10] if w['importance'] > 0)
                human_words = set(human_rationales[text])
                
                if len(shap_words) > 0 and len(human_words) > 0:
                    intersection = shap_words & human_words
                    union = shap_words | human_words
                    
                    precision = len(intersection) / len(shap_words)
                    recall = len(intersection) / len(human_words)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    iou_f1 = len(intersection) / len(union)
                    overlap_percentage = len(intersection) / len(shap_words) * 100
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1)
                    iou_f1_scores.append(iou_f1)
                    overlap_percentages.append(overlap_percentage)
                    
                    if len(intersection) > 0:
                        agreement_count += 1
                        
        except Exception as e:
            print(f"Error processing example {i+1}: {e}")
            continue
    
    return {
        'faithfulness': {
            'comprehensiveness': {'mean': np.mean(comprehensiveness_scores), 'std': np.std(comprehensiveness_scores)},
            'sufficiency': {'mean': np.mean(sufficiency_scores), 'std': np.std(sufficiency_scores)}
        },
        'plausibility': {
            'precision': {'mean': np.mean(precisions), 'std': np.std(precisions)},
            'recall': {'mean': np.mean(recalls), 'std': np.std(recalls)},
            'f1': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)},
            'iou_f1': {'mean': np.mean(iou_f1_scores), 'std': np.std(iou_f1_scores)}
        },
        'coverage': {
            'agreement_rate': agreement_count / coverage_count if coverage_count > 0 else 0,
            'coverage': coverage_count / len(shap_results),
            'average_overlap': {'mean': np.mean(overlap_percentages), 'std': np.std(overlap_percentages)}
        }
    }

# Compute all metrics
all_metrics = compute_all_metrics(shap_results, human_rationales, predict_fn)

# Calculate basic statistics
total_samples = len(shap_results)
accuracy = sum(1 for r in shap_results if r['correct_prediction']) / total_samples

# Print results
print("\nSHAP EVALUATION RESULTS")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_NAME}")
print(f"Total offensive samples analyzed: {total_samples}")
print(f"Accuracy: {accuracy:.3f}")
print()

print("FAITHFULNESS METRICS:")
print(f"  Comprehensiveness: {all_metrics['faithfulness']['comprehensiveness']['mean']:.3f} ± {all_metrics['faithfulness']['comprehensiveness']['std']:.3f}")
print(f"  Sufficiency: {all_metrics['faithfulness']['sufficiency']['mean']:.3f} ± {all_metrics['faithfulness']['sufficiency']['std']:.3f}")
print()

print("PLAUSIBILITY METRICS:")
print(f"  Precision: {all_metrics['plausibility']['precision']['mean']:.3f} ± {all_metrics['plausibility']['precision']['std']:.3f}")
print(f"  Recall: {all_metrics['plausibility']['recall']['mean']:.3f} ± {all_metrics['plausibility']['recall']['std']:.3f}")
print(f"  F1: {all_metrics['plausibility']['f1']['mean']:.3f} ± {all_metrics['plausibility']['f1']['std']:.3f}")
print(f"  IOU F1: {all_metrics['plausibility']['iou_f1']['mean']:.3f} ± {all_metrics['plausibility']['iou_f1']['std']:.3f}")
print()

print("COVERAGE METRICS:")
print(f"  Agreement rate: {all_metrics['coverage']['agreement_rate']:.1%}")
print(f"  Coverage: {all_metrics['coverage']['coverage']:.1%}")
print(f"  Average overlap: {all_metrics['coverage']['average_overlap']['mean']:.1f}% ± {all_metrics['coverage']['average_overlap']['std']:.1f}%")
print()

# Comparison table format
print("COMPARISON TABLE FORMAT:")
print("-" * 100)
print("Model | Comprehensiveness | Sufficiency | Precision | Recall | F1 | IOU F1 | Agreement | Coverage | Overlap")
print("-" * 100)
print(f"SHAP-bert-base-uncased | "
      f"{all_metrics['faithfulness']['comprehensiveness']['mean']:.3f}±{all_metrics['faithfulness']['comprehensiveness']['std']:.3f} | "
      f"{all_metrics['faithfulness']['sufficiency']['mean']:.3f}±{all_metrics['faithfulness']['sufficiency']['std']:.3f} | "
      f"{all_metrics['plausibility']['precision']['mean']:.3f}±{all_metrics['plausibility']['precision']['std']:.3f} | "
      f"{all_metrics['plausibility']['recall']['mean']:.3f}±{all_metrics['plausibility']['recall']['std']:.3f} | "
      f"{all_metrics['plausibility']['f1']['mean']:.3f}±{all_metrics['plausibility']['f1']['std']:.3f} | "
      f"{all_metrics['plausibility']['iou_f1']['mean']:.3f}±{all_metrics['plausibility']['iou_f1']['std']:.3f} | "
      f"{all_metrics['coverage']['agreement_rate']:.1%} | "
      f"{all_metrics['coverage']['coverage']:.1%} | "
      f"{all_metrics['coverage']['average_overlap']['mean']:.1f}%±{all_metrics['coverage']['average_overlap']['std']:.1f}%")
print("-" * 100)

# Save results
results_filename = f"{OUTPUT_DIR}/shap_evaluation_results.json"
final_results = {
    'model_info': {
        'model_name': MODEL_NAME,
        'total_samples': total_samples,
        'accuracy': accuracy
    },
    'metrics': all_metrics
}

with open(results_filename, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"✓ Evaluation results saved to: {results_filename}")
print("\nSHAP EVALUATION COMPLETE!")
print("These metrics can now be directly compared with your LIME results!")
