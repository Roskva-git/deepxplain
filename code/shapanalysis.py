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

# Select examples for SHAP analysis
print("Selecting examples for SHAP analysis...")

# Choose a diverse set of examples
analysis_examples = []
analysis_labels = []

# Get examples from different categories
neutral_examples = [(text, label) for text, label in zip(test_texts, test_labels) if label == 0]
offensive_examples = [(text, label) for text, label in zip(test_texts, test_labels) if label == 1]

# Select balanced examples
examples_per_class = 5  # Analyze 5 examples per class
analysis_examples.extend([text for text, label in neutral_examples[:examples_per_class]])
analysis_labels.extend([label for text, label in neutral_examples[:examples_per_class]])
analysis_examples.extend([text for text, label in offensive_examples[:examples_per_class]])
analysis_labels.extend([label for text, label in offensive_examples[:examples_per_class]])

print(f"Selected {len(analysis_examples)} examples for analysis:")
print(f"  Neutral examples: {examples_per_class}")
print(f"  Offensive examples: {examples_per_class}")
print()

# Generate SHAP explanations
print("Generating SHAP explanations...")
print("This will take several minutes. Progress will be shown below.")
print()

shap_results = []

for i, (text, true_label) in enumerate(zip(analysis_examples, analysis_labels)):
    print(f"Processing example {i+1}/{len(analysis_examples)}...")
    print(f"  Text: {text[:100]}...")
    print(f"  True label: {'Offensive' if true_label == 1 else 'Neutral'}")
    
    try:
        # Clear GPU cache before each example
        if torch.cuda.is_available():
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
        print()
        
    except Exception as e:
        print(f"  ✗ Error processing example {i+1}: {e}")
        print(f"  Skipping this example...")
        print()
        continue

print(f"SHAP analysis complete! Successfully processed {len(shap_results)} examples.")
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
    
    print(f"Example {i+1} - Top important words:")
    for j, word_info in enumerate(word_importance[:5]):  # Show top 5
        direction = "→ Offensive" if word_info['importance'] > 0 else "→ Neutral"
        print(f"  {j+1}. '{word_info['word']}': {word_info['importance']:.3f} {direction}")
    print()

# Create visualizations directory
viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(viz_dir, exist_ok=True)

# Generate word importance plots
print("Creating SHAP word importance visualizations...")
for i, result in enumerate(shap_results):
    if result['word_importance']:
        true_label = 'Offensive' if result['true_label'] == 1 else 'Neutral'
        pred_label = 'Offensive' if result['predicted_label'] == 1 else 'Neutral'
        
        filename = os.path.join(viz_dir, f"shap_word_importance_example_{i+1}.png")
        
        success = create_shap_word_plot(
            result['word_importance'],
            result['text'],
            true_label,
            pred_label,
            result['confidence'],
            filename
        )
        
        if success:
            print(f"  ✓ Created word importance plot for example {i+1}")
        else:
            print(f"  ✗ Failed to create plot for example {i+1}")

print(f"Word importance visualizations saved to: {viz_dir}")
print()

# SHAP RESULTS IN LIME-COMPARABLE FORMAT
print("SHAP RESULTS IN LIME-COMPARABLE FORMAT:")
print("=" * 60)

for i, result in enumerate(shap_results):
    print(f"\nExample {i+1}:")
    print(f"Text: {result['text']}")
    print(f"True Label: {'Offensive' if result['true_label'] == 1 else 'Neutral'}")
    print(f"Predicted: {'Offensive' if result['predicted_label'] == 1 else 'Neutral'}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Correct: {result['correct_prediction']}")
    
    print(f"Top words influencing decision:")
    for j, word_info in enumerate(result['word_importance'][:8]):
        direction = "OFFENSIVE" if word_info['importance'] > 0 else "NEUTRAL"
        print(f"  {j+1}. '{word_info['word']}': {word_info['importance']:.3f} → {direction}")
    print("-" * 60)


## ---- 4.6 Saving SHAP results ----

print("\n4.6 SAVING SHAP RESULTS")
print()

# Prepare results for saving
print("Preparing results for saving...")

# Create comprehensive results dictionary
results_dict = {
    'analysis_info': {
        'timestamp': datetime.now().isoformat(),
        'model_name': MODEL_NAME,
        'dataset_name': DATASET_NAME,
        'max_length': MAX_LENGTH,
        'batch_size': BATCH_SIZE,
        'background_size': len(background_texts),
        'examples_analyzed': len(shap_results),
        'accuracy': accuracy,
        'device': str(device)
    },
    'configuration': {
        'trained_model_path': TRAINED_MODEL_PATH,
        'preprocessed_data_path': PREPROCESSED_DATA_PATH,
        'output_directory': OUTPUT_DIR,
        'examples_per_class': examples_per_class,
        'background_per_class': per_class
    },
    'results': []
}


# Process and save results with proper type conversion
for i, result in enumerate(shap_results):
    result_dict = {
        'example_id': i + 1,
        'text': result['text'],
        'true_label': int(result['true_label']),
        'predicted_label': int(result['predicted_label']),
        'prediction_probs': [float(p) for p in result['prediction_probs']],
        'confidence': float(result['confidence']),
        'correct_prediction': bool(result['correct_prediction']),
        'word_importance': [
            {
                'word': word_info['word'],
                'importance': float(word_info['importance']),
                'position': int(word_info['position'])
            }
            for word_info in result['word_importance']
        ],
        'shap_info': {
            'values_shape': str(result['shap_values'].values.shape),
            'base_values': [float(x) for x in result['shap_values'].base_values],
            'data_preview': str(result['shap_values'].data)[:200]
        }
    }
    results_dict['results'].append(result_dict)


# Save results as JSON
json_filename = f"{OUTPUT_DIR}/shap_results.json"
try:
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    print(f"✓ Results saved to: {json_filename}")
except Exception as e:
    print(f"✗ Error saving JSON results: {e}")

# Save raw SHAP values using pickle
pickle_filename = f"{OUTPUT_DIR}/shap_values.pkl"
try:
    with open(pickle_filename, 'wb') as f:
        pickle.dump(shap_results, f)
    print(f"✓ Raw SHAP values saved to: {pickle_filename}")
except Exception as e:
    print(f"✗ Error saving pickle results: {e}")

# Create summary report
print("Creating summary report...")
report_filename = f"{OUTPUT_DIR}/shap_analysis_report.txt"

try:
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("SHAP ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Max Length: {MAX_LENGTH} tokens\n")
        f.write(f"Device: {device}\n\n")
        
        f.write("ANALYSIS RESULTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Examples Analyzed: {len(shap_results)}\n")
        f.write(f"Correct Predictions: {correct_predictions}\n")
        f.write(f"Accuracy: {accuracy:.3f}\n\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(shap_results):
            f.write(f"Example {i+1}:\n")
            f.write(f"  Text: {result['text'][:100]}...\n")
            f.write(f"  True: {'Offensive' if result['true_label'] == 1 else 'Neutral'}\n")
            f.write(f"  Predicted: {'Offensive' if result['predicted_label'] == 1 else 'Neutral'}\n")
            f.write(f"  Confidence: {result['confidence']:.3f}\n")
            f.write(f"  Correct: {result['correct_prediction']}\n\n")
    
    print(f"✓ Summary report saved to: {report_filename}")
    
except Exception as e:
    print(f"✗ Error creating summary report: {e}")

print()
print("SHAP ANALYSIS COMPLETE!")
print("=" * 50)
print()
print("Files created:")
print(f"  - JSON results: {json_filename}")
print(f"  - Raw SHAP values: {pickle_filename}")
print(f"  - Summary report: {report_filename}")
print(f"  - Visualizations: {viz_dir}")
print()
print("Next steps:")
print("  1. Review the summary report for key findings")
print("  2. Examine visualizations to understand feature importance")
print("  3. Compare with LIME results for comprehensive analysis")
print("  4. Consider analyzing more examples if needed")
print()
print("For questions or issues, check the Fox cluster logs and GPU memory usage.")