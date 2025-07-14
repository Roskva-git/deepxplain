# ============================================================================
# 3. ANALYSIS WITH LIME & SHAP
# ============================================================================

## Chapter 3: Analysis with LIME and SHAP

# Written by: Røskva
# Created: 09. July 2025
# Updated: 14. July 2025


# 3. ANALYSIS WITH LIME & SHAP
### 3.0 Setting up explainability functions
### 3.1 Combined LIME and SHAP Analysis
### 3.2 Save comprehensive analysis to file
### 3.3 Overall summary
### 3.4 


print("\n")
print("3. ANALYSIS WITH LIME & SHAP")
print()

## ---- 3.0 Setting up explainability functions (Francielle's original functions) ----

print("\n3.0 SETTING UP EXPLAINABILITY FUNCTIONS")
print()


# Importing libraries
import torch
import json
import shap
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, BertTokenizer
from lime.lime_text import LimeTextExplainer
from sklearn.model_selection import train_test_split

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Loading model 
print("Loading saved model...")
TRAINING_OUTPUT_DIR = "/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/training_results/training_20250707_1206"
model_path = f"{TRAINING_OUTPUT_DIR}/trained_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.to(device)
print(f"Model loaded from: {model_path}")
print()


# Load the dataset
dataset_path = "/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/dataset/HateBRXplain.json"
with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


# Extract texts and labels
texts = [item['comment'] for item in data]
labels = [int(item['offensive label']) for item in data]

# Splitting dataset into train, validation 
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

# Track which original indices ended up in test set
all_indices = list(range(len(texts)))
train_indices, temp_indices, _, _ = train_test_split(
    all_indices, labels, test_size=0.3, random_state=42, stratify=labels
)
val_indices, test_indices, _, _ = train_test_split(
    temp_indices, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

class SimpleDataset:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)

test_dataset = SimpleDataset(test_texts, test_labels)
print(f"Dataset loaded: {len(test_dataset)} test samples")
print()


print("Setting up prediction function...")

def predict_fn(texts):
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        # Move encodings to device
        encodings = {k: v.to(device) for k, v in encodings.items()}
        outputs = model(**encodings)
    return torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

# Initialize explainer
explainer = LimeTextExplainer(class_names=['Neutral', 'Offensive'])  # Ajuste as classes conforme necessário

def explain_instance(text, model, num_features=10):
    explanation = explainer.explain_instance(
        text, 
        predict_fn, 
        num_features=num_features
    )
    return explanation

print("Functions set up successfully!")
print("predict_fn() - handles tokenization and prediction")
print("explainer - LimeTextExplainer with class names")
print("explain_instance() - generates explanations")
print()

print("What these functions do:")
print("  - predict_fn(): Tokenizes texts and returns prediction probabilities")
print("  - explainer: LIME text explainer configured for sentiment analysis")
print("  - explain_instance(): Generates feature importance explanations")
print()


## ---- 3.1 Combined LIME and SHAP Analysis ----

print("\n3.2 COMBINED LIME AND SHAP ANALYSIS WITH RATIONALES")
print()


# Filter for offensive examples
offensive_indices = [i for i, label in enumerate(test_dataset.labels) if label == 1]
print(f"Found {len(offensive_indices)} offensive examples in test set")

# Pick 3 random offensive examples
np.random.seed(42)
if len(offensive_indices) >= 3:
    sample_indices = np.random.choice(offensive_indices, 3, replace=False)
else:
    sample_indices = offensive_indices  # Use all if less than 3


print(f"Generating both LIME and SHAP explanations for 3 random test samples...")
print("Comparing model explanations with human rationales...")
print()

# Set up SHAP
print("Setting up SHAP explainer...")

# SHAP explainer for transformers
shap_explainer = shap.Explainer(predict_fn, shap.maskers.Text(tokenizer))
print("SHAP explainer ready")
print()

for i, idx in enumerate(sample_indices):
    print(f"{'='*70}")
    print(f"SAMPLE {i+1}/3 (Test Index: {idx})")
    print(f"{'='*70}")
    
    # Get the original data item (not just from test_dataset)
    original_item = data[test_indices[idx]]  # Get original item with rationales
    
    # Get the text and true label
    text = test_dataset.texts[idx]
    true_label = test_dataset.labels[idx]
    
    # Get human rationales
    rationale_1 = original_item.get('rationales annotator 1', 'N/A')
    rationale_2 = original_item.get('rationales annotator 2', 'N/A')
    
    print(f"Full Text:")
    print(f"   \"{text}\"")
    print()
    
    print(f"Human Rationales (what humans think is offensive):")
    print(f"   Annotator 1: \"{rationale_1}\"")
    print(f"   Annotator 2: \"{rationale_2}\"")
    print()
    
    print(f"Ground Truth: {true_label} ({'Offensive' if true_label == 1 else 'Neutral'})")
    print()
    
    # Get model prediction
    pred_probs = predict_fn([text])[0]
    pred_label = np.argmax(pred_probs)
    
    print(f"Model Prediction:")
    print(f"   Predicted: {pred_label} ({'Offensive' if pred_label == 1 else 'Neutral'})")
    print(f"   Confidence: {pred_probs[pred_label]:.3f}")
    print(f"   Probabilities: [Neutral: {pred_probs[0]:.3f}, Offensive: {pred_probs[1]:.3f}]")
    print(f"   Result: {'Correct' if pred_label == true_label else 'Wrong'}")
    print()
    
    # ============ LIME ANALYSIS ============
    print(f"LIME EXPLANATION:")
    print("-" * 30)
    
    # Generate LIME explanation
    lime_explanation = explain_instance(text, model, num_features=8)
    lime_features = lime_explanation.as_list()
    
    print(f"   Top {min(6, len(lime_features))} influential words:")
    for j, (feature, weight) in enumerate(lime_features[:6], 1):
        if weight > 0:
            direction = f"→ Offensive (+{weight:.3f})"
        else:
            direction = f"→ Neutral ({weight:.3f})"
        print(f"   {j:2d}. '{feature}': {direction}")
    
    # Extract LIME offensive words
    lime_offensive_words = [word for word, weight in lime_features if weight > 0]
    print()
    
# ============ SHAP ANALYSIS ============
    print(f"SHAP EXPLANATION:")
    print("-" * 30)
    
    try:        
        # Get explanation with limited evaluations to avoid timeout
        shap_values = shap_explainer([text], max_evals=50)
        
        # Extract the values for the offensive class
        if len(shap_values) > 0:
            explanation = shap_values[0]
            
            # Get the text tokens (simple word split)
            words = text.split()
            
            # Try to get the values for offensive class
            if hasattr(explanation, 'values') and explanation.values is not None:
                values = explanation.values
                
                # If 2D array, get offensive class (index 1)
                if len(values.shape) > 1 and values.shape[1] >= 2:
                    offensive_values = values[:, 1]
                else:
                    offensive_values = values
                
                # Ensure we don't exceed array bounds
                num_features = min(len(words), len(offensive_values))
                
                # Create word-value pairs
                word_values = []
                for i in range(num_features):
                    word_values.append((words[i], offensive_values[i]))
                
                # Sort by absolute value
                word_values.sort(key=lambda x: abs(x[1]), reverse=True)
                
                print(f"   Top {min(6, len(word_values))} influential tokens:")
                for j, (word, value) in enumerate(word_values[:6], 1):
                    if value > 0:
                        direction = f"→ Offensive (+{value:.3f})"
                    else:
                        direction = f"→ Neutral ({value:.3f})"
                    print(f"   {j:2d}. '{word}': {direction}")
                
                # Extract offensive words
                shap_offensive_words = [word for word, value in word_values if value > 0]
                print(f"SHAP analysis completed successfully")
                
            else:
                print("Could not extract SHAP values from explanation")
                shap_offensive_words = []
        else:
            print("SHAP returned empty explanation")
            shap_offensive_words = []
            
    except Exception as e:
        print(f"SHAP analysis failed: {str(e)[:100]}...")
        shap_offensive_words = []
        print("Continuing with LIME analysis only...")
    
    print()

    
    # ============ COMPARISON ANALYSIS ============
    print(f"LIME vs SHAP vs HUMAN COMPARISON:")
    print("-" * 45)
    
    # Get human rationale words
    rationale_words = set()
    if rationale_1 != 'N/A':
        rationale_words.update(rationale_1.lower().split())
    if rationale_2 != 'N/A':
        rationale_words.update(rationale_2.lower().split())
    
    if true_label == 1:  # Only analyze for offensive examples
        print(f"   LIME top offensive words: {lime_offensive_words[:5]}")
        print(f"   SHAP top offensive words: {shap_offensive_words[:5]}")
        print(f"   Human rationale words: {list(rationale_words)}")
        
        # Check overlaps
        lime_words_lower = [word.lower() for word in lime_offensive_words[:5]]
        shap_words_lower = [word.lower() for word in shap_offensive_words[:5]]
        
        lime_human_overlap = [word for word in lime_words_lower if word in rationale_words]
        shap_human_overlap = [word for word in shap_words_lower if word in rationale_words]
        lime_shap_overlap = [word for word in lime_words_lower if word in shap_words_lower]
        
        print(f"   LIME-Human overlap: {lime_human_overlap}")
        print(f"   SHAP-Human overlap: {shap_human_overlap}")
        print(f"   LIME-SHAP overlap: {lime_shap_overlap}")
        
        # Analysis
        if lime_human_overlap and shap_human_overlap:
            print(f"   Both methods agree with human reasoning")
        elif lime_human_overlap:
            print(f"   LIME better matches human reasoning")
        elif shap_human_overlap:
            print(f"   SHAP better matches human reasoning")
        else:
            print(f"   Neither method matches human reasoning closely")
            
    else:
        print(f"   This is a neutral example - checking if methods agree it's non-offensive")
        if len(lime_offensive_words) == 0 and len(shap_offensive_words) == 0:
            print(f"   Both LIME and SHAP agree: no strong offensive words")
        else:
            print(f"   Methods disagree on offensive content")
            print(f"   LIME found: {lime_offensive_words[:3]}")
            print(f"   SHAP found: {shap_offensive_words[:3]}")
    
    print()
    
    # ============ SAVE PLOTS ============
    # Create LIME plot
    fig = lime_explanation.as_pyplot_figure()
    plt.title(f'LIME - Sample {i+1}: True={true_label}, Pred={pred_label}')
    lime_plot_file = f"{TRAINING_OUTPUT_DIR}/lime_plot_sample_{i+1}.png"
    plt.savefig(lime_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"LIME plot saved: lime_plot_sample_{i+1}.png")
    print()



# ---- 3.2 Save comprehensive analysis to file ----

results_file = f"{TRAINING_OUTPUT_DIR}/lime_analysis_results.txt"
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("LIME ANALYSIS RESULTS\n")
    f.write("=" * 50 + "\n\n")
    
    # Re-analyze each sample for the file (or store results during the loop)
    for i, idx in enumerate(sample_indices):
        text = test_dataset.texts[idx]
        true_label = test_dataset.labels[idx]
        
        # Get rationales and prediction
        original_item = data[test_indices[idx]]
        rationale_1 = original_item.get('rationales annotator 1', 'N/A')
        rationale_2 = original_item.get('rationales annotator 2', 'N/A')
        
        pred_probs = predict_fn([text])[0]
        pred_label = np.argmax(pred_probs)
        
        # LIME analysis
        lime_explanation = explain_instance(text, model, num_features=8)
        lime_features = lime_explanation.as_list()
        
        f.write(f"SAMPLE {i+1}/3 (Test Index: {idx})\n")
        f.write("-" * 40 + "\n")
        f.write(f"Text: {text}\n")
        f.write(f"True label: {true_label}\n")
        f.write(f"Predicted: {pred_label} (confidence: {pred_probs[pred_label]:.3f})\n")
        f.write(f"Human rationale 1: {rationale_1}\n")
        f.write(f"Human rationale 2: {rationale_2}\n\n")
        
        f.write("LIME Features:\n")
        for j, (feature, weight) in enumerate(lime_features[:5], 1):
            f.write(f"  {j}. '{feature}': {weight:+.3f}\n")
        f.write("\n")
        
        # Calculate overlap analysis
        f.write("OVERLAP ANALYSIS:\n")
        f.write("-" * 20 + "\n")
        
        # Get human rationale words
        rationale_words = set()
        if rationale_1 != 'N/A':
            rationale_words.update(rationale_1.lower().split())
        if rationale_2 != 'N/A':
            rationale_words.update(rationale_2.lower().split())
        
        # Get LIME important words (positive weights = offensive)
        lime_offensive_words = [word.lower() for word, weight in lime_features if weight > 0]
        lime_top_words = [word.lower() for word, weight in lime_features[:5]]
        
        # Calculate overlaps
        lime_human_overlap = [word for word in lime_top_words if word in rationale_words]
        overlap_percentage = len(lime_human_overlap) / max(len(lime_top_words), 1) * 100
        
        f.write(f"Human rationale words: {list(rationale_words)}\n")
        f.write(f"LIME top 5 words: {lime_top_words}\n")
        f.write(f"LIME offensive words: {lime_offensive_words}\n")
        f.write(f"Overlap words: {lime_human_overlap}\n")
        f.write(f"Overlap percentage: {overlap_percentage:.1f}% ({len(lime_human_overlap)}/{len(lime_top_words)})\n")
        f.write("\n" + "="*50 + "\n\n")

print(f"Detailed analysis saved to: {results_file}")




# ---- 3.3 Overall summary ----

print(f"\n{'='*40}")
print("OVERALL ANALYSIS SUMMARY")
print(f"{'='*40}")
print("This analysis compared three explanation approaches:")
print("1. LIME: Local perturbation-based explanations")
print("2. SHAP: Game theory-based feature attributions") 
print("3. Human rationales: Ground truth annotations")
print()
print("Key insights to look for:")
print("- Do LIME and SHAP agree on important words?")
print("- Which method better matches human reasoning?")
print("- Are there systematic differences between methods?")
print("- Do explanations make sense for the predictions?")
print()

print("Combined LIME and SHAP analysis complete!")
print(f"LIME plots: lime_plot_sample_1.png, lime_plot_sample_2.png, lime_plot_sample_3.png")
print(f"SHAP plots: shap_plot_sample_1.png, shap_plot_sample_2.png, shap_plot_sample_3.png")
