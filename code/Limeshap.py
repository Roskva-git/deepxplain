## Chapter 3: Analysis with LIME and SHAP

# Written by: Røskva
# Created: 09. July 2025
# Updated: 15. July 2025


# 3. ANALYSIS WITH LIME & SHAP
### 3.0 Setting up explainability functions
### 3.1 Calculating faithfulness metrics
### 3.2 Calculating plausibility metrics
### 3.3 Combined LIME and SHAP Analysis
### 3.4 Population-level statistics
### 3.5 Save comprehensive analysis to file
### 3.6 Overall summary


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
TRAINING_OUTPUT_DIR = "/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/training_results/training_20250708_1500"
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
    """Batched prediction function to avoid memory overflow"""
    batch_size = 8  # Reduce this if still getting OOM
    all_results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Process batch
        encodings = tokenizer(batch_texts, truncation=True, padding=True, return_tensors='pt')
        
        with torch.no_grad():
            # Move encodings to device
            encodings = {k: v.to(device) for k, v in encodings.items()}
            outputs = model(**encodings)
            batch_probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_results.extend(batch_probs)
            
        # Clear cache after each batch
        torch.cuda.empty_cache()
    
    return np.array(all_results)


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


## ---- 3.1 Calculating faithfulness metrics ----

# Sufficiency and comprehensiveness

def calculate_faithfulness_metrics(text, original_probs, lime_features, num_top_features=3):
    """Calculate comprehensiveness and sufficiency metrics"""
    
    # Get top important words (by absolute weight)
    top_features = sorted(lime_features, key=lambda x: abs(x[1]), reverse=True)[:num_top_features]
    top_words = [word for word, weight in top_features]
    
    # Split text into words
    words = text.split()
    
    # COMPREHENSIVENESS: Remove top features
    words_without_top = [word for word in words if word not in top_words]
    text_without_top = ' '.join(words_without_top) if words_without_top else '[EMPTY]'
    
    # SUFFICIENCY: Keep only top features  
    words_only_top = [word for word in words if word in top_words]
    text_only_top = ' '.join(words_only_top) if words_only_top else '[EMPTY]'
    
    try:
        # Get predictions for modified texts
        if text_without_top != '[EMPTY]':
            probs_without_top = predict_fn([text_without_top])[0]
        else:
            probs_without_top = [0.5, 0.5]  # Neutral when empty
            
        if text_only_top != '[EMPTY]':
            probs_only_top = predict_fn([text_only_top])[0]
        else:
            probs_only_top = [0.5, 0.5]  # Neutral when empty
        
        # Calculate metrics
        original_confidence = max(original_probs)
        
        # Comprehensiveness: How much confidence drops when removing top features
        confidence_without_top = max(probs_without_top)
        comprehensiveness = original_confidence - confidence_without_top
        
        # Sufficiency: How much confidence is retained with only top features  
        confidence_only_top = max(probs_only_top)
        sufficiency = confidence_only_top / original_confidence if original_confidence > 0 else 0
        
        return {
            'comprehensiveness': comprehensiveness,
            'sufficiency': sufficiency,
            'text_without_top': text_without_top,
            'text_only_top': text_only_top,
            'top_words': top_words,
            'confidence_original': original_confidence,
            'confidence_without_top': confidence_without_top,
            'confidence_only_top': confidence_only_top
        }
        
    except Exception as e:
        return {
            'comprehensiveness': 0.0,
            'sufficiency': 0.0,
            'error': str(e)
        }



## ---- 3.2 Calculating plausibility metrics ----

def calculate_plausibility_metrics(lime_features, rationale_1, rationale_2, num_top_features=5):
    """Calculate plausibility metrics: how well LIME matches human rationales"""
    
    # Get top LIME words
    lime_top_words = [word.lower() for word, weight in lime_features[:num_top_features]]
    
    # Get human rationale words
    human_words = set()
    if rationale_1 != 'N/A':
        human_words.update(rationale_1.lower().split())
    if rationale_2 != 'N/A':
        human_words.update(rationale_2.lower().split())
    
    # Convert to sets for easier calculation
    lime_set = set(lime_top_words)
    human_set = human_words
    
    # Calculate metrics
    intersection = lime_set.intersection(human_set)
    union = lime_set.union(human_set)
    
    # Token-level Precision: Of LIME's words, how many match humans?
    precision = len(intersection) / len(lime_set) if len(lime_set) > 0 else 0
    
    # Token-level Recall: Of human words, how many does LIME find?
    recall = len(intersection) / len(human_set) if len(human_set) > 0 else 0
    
    # F1 Score: Harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # IOU F1: Intersection over Union
    iou_f1 = len(intersection) / len(union) if len(union) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'iou_f1': iou_f1,
        'intersection': list(intersection),
        'lime_words': lime_top_words,
        'human_words': list(human_set),
        'intersection_count': len(intersection),
        'lime_count': len(lime_set),
        'human_count': len(human_set)
    }


## ---- 3.3 Combined LIME and SHAP Analysis ----

print("\n3.3 COMBINED LIME AND SHAP ANALYSIS WITH RATIONALES")
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


# ---- 3.4 Population-Level Statistics ----

print("\n3.4 CALCULATING POPULATION-LEVEL STATISTICS")
print("=" * 60)
print("Running LIME analysis on all offensive examples for robust statistics...")
print()

# Storage for population metrics
all_faithfulness_metrics = []
all_plausibility_metrics = []
all_overlap_percentages = []

# Run LIME on ALL offensive examples (silently)
for i, idx in enumerate(offensive_indices):
    # Progress tracking
    if i % 100 == 0:
        print(f"Processing sample {i+1}/{len(offensive_indices)}...")
        # Clear memory every 100 samples
        torch.cuda.empty_cache()
    
    # Get data for this sample
    text = test_dataset.texts[idx]
    true_label = test_dataset.labels[idx]
    original_item = data[test_indices[idx]]
    rationale_1 = original_item.get('rationales annotator 1', 'N/A')
    rationale_2 = original_item.get('rationales annotator 2', 'N/A')
    
    # Get prediction and LIME analysis
    pred_probs = predict_fn([text])[0]
    pred_label = np.argmax(pred_probs)
    lime_explanation = explain_instance(text, model, num_features=8)
    lime_features = lime_explanation.as_list()
    
    # Calculate metrics
    faithfulness = calculate_faithfulness_metrics(text, pred_probs, lime_features)
    plausibility = calculate_plausibility_metrics(lime_features, rationale_1, rationale_2)
    
    # Calculate overlap percentage
    rationale_words = set()
    if rationale_1 != 'N/A':
        rationale_words.update(rationale_1.lower().split())
    if rationale_2 != 'N/A':
        rationale_words.update(rationale_2.lower().split())
    lime_top_words = [word.lower() for word, weight in lime_features[:5]]
    lime_human_overlap = [word for word in lime_top_words if word in rationale_words]
    overlap_percentage = len(lime_human_overlap) / max(len(lime_top_words), 1) * 100
    
    # Store all metrics
    all_faithfulness_metrics.append(faithfulness)
    all_plausibility_metrics.append(plausibility)
    all_overlap_percentages.append(overlap_percentage)

    # Clear memory periodically
    if i % 10 == 0:  # Every 10 samples
        torch.cuda.empty_cache()

print(f"Completed analysis on {len(offensive_indices)} samples!")
print()

# Calculate aggregate statistics
comprehensiveness_scores = [m['comprehensiveness'] for m in all_faithfulness_metrics]
sufficiency_scores = [m['sufficiency'] for m in all_faithfulness_metrics]
precision_scores = [m['precision'] for m in all_plausibility_metrics]
recall_scores = [m['recall'] for m in all_plausibility_metrics]
f1_scores = [m['f1_score'] for m in all_plausibility_metrics]
iou_scores = [m['iou_f1'] for m in all_plausibility_metrics]


# Coverage metrics

# Check if LIME found any offensive words (positive weights)
lime_offensive_counts = []
for i, idx in enumerate(offensive_indices):
    lime_explanation = explain_instance(test_dataset.texts[idx], model, num_features=8) 
    lime_features = lime_explanation.as_list()
    offensive_words = [word for word, weight in lime_features if weight > 0]
    lime_offensive_counts.append(len(offensive_words))
    
samples_with_lime_offensive = sum(1 for count in lime_offensive_counts if count > 0)
samples_with_overlap = sum(1 for p in all_overlap_percentages if p > 0)
samples_with_good_agreement = sum(1 for f1 in f1_scores if f1 > 0.5)

# Display results
print("POPULATION-LEVEL STATISTICS")
print("=" * 40)
print(f"Total samples analyzed: {len(offensive_indices)}")
print()
print("FAITHFULNESS METRICS:")
print(f"  Comprehensiveness: {np.mean(comprehensiveness_scores):.3f} ± {np.std(comprehensiveness_scores):.3f}")
print(f"  Sufficiency: {np.mean(sufficiency_scores):.3f} ± {np.std(sufficiency_scores):.3f}")
print()
print("PLAUSIBILITY METRICS:")
print(f"  Token-level Precision: {np.mean(precision_scores):.3f} ± {np.std(precision_scores):.3f}")
print(f"  Token-level Recall: {np.mean(recall_scores):.3f} ± {np.std(recall_scores):.3f}")
print(f"  Token-level F1: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
print(f"  IOU F1: {np.mean(iou_scores):.3f} ± {np.std(iou_scores):.3f}")
print(f"  Overall agreement rate (F1>0.5): {samples_with_good_agreement}/{len(f1_scores)} ({samples_with_good_agreement/len(f1_scores)*100:.1f}%)")
print()
print("COVERAGE METRICS:")
print(f"  Samples with any human-LIME overlap: {samples_with_overlap}/{len(all_overlap_percentages)} ({samples_with_overlap/len(all_overlap_percentages)*100:.1f}%)")
print(f"  Average overlap percentage: {np.mean(all_overlap_percentages):.1f}% ± {np.std(all_overlap_percentages):.1f}%")
print()



# ---- 3.5 Save comprehensive analysis to file ----

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

        # Get faithfulness metrics
        faithfulness = calculate_faithfulness_metrics(text, pred_probs, lime_features)

        f.write("FAITHFULNESS METRICS:\n")
        f.write(f"Comprehensiveness: {faithfulness['comprehensiveness']:.3f}\n")
        f.write(f"Sufficiency: {faithfulness['sufficiency']:.3f}\n")
        f.write(f"Top words used: {faithfulness['top_words']}\n")
        f.write(f"Original confidence: {faithfulness['confidence_original']:.3f}\n")
        f.write(f"Without top words: {faithfulness['confidence_without_top']:.3f}\n")
        f.write(f"Only top words: {faithfulness['confidence_only_top']:.3f}\n")
        f.write("\n" + "="*50 + "\n\n")
        
        # Get plausibility metrics
        plausibility = calculate_plausibility_metrics(lime_features, rationale_1, rationale_2)
        
        f.write("PLAUSIBILITY METRICS:\n")
        f.write(f"Token-level Precision: {plausibility['precision']:.3f}\n")
        f.write(f"Token-level Recall: {plausibility['recall']:.3f}\n")
        f.write(f"Token-level F1 Score: {plausibility['f1_score']:.3f}\n")
        f.write(f"IOU F1 Score: {plausibility['iou_f1']:.3f}\n")
        f.write(f"Intersection words: {plausibility['intersection']}\n")
        f.write(f"LIME words ({plausibility['lime_count']}): {plausibility['lime_words']}\n")
        f.write(f"Human words ({plausibility['human_count']}): {plausibility['human_words']}\n")
        f.write("\n" + "="*50 + "\n\n")

    # Save population statistics to file
    f.write("\n" + "="*60 + "\n")
    f.write("POPULATION-LEVEL STATISTICS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total offensive samples analyzed: {len(offensive_indices)}\n\n")
    
    f.write("FAITHFULNESS METRICS (Population):\n")
    f.write(f"Comprehensiveness: {np.mean(comprehensiveness_scores):.3f} ± {np.std(comprehensiveness_scores):.3f}\n")
    f.write(f"Sufficiency: {np.mean(sufficiency_scores):.3f} ± {np.std(sufficiency_scores):.3f}\n\n")
            
    f.write("PLAUSIBILITY METRICS (Population):\n")
    f.write(f"Token-level Precision: {np.mean(precision_scores):.3f} ± {np.std(precision_scores):.3f}\n")
    f.write(f"Token-level Recall: {np.mean(recall_scores):.3f} ± {np.std(recall_scores):.3f}\n")
    f.write(f"Token-level F1: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}\n")
    f.write(f"IOU F1: {np.mean(iou_scores):.3f} ± {np.std(iou_scores):.3f}\n")
    f.write(f"Overall agreement rate (F1>0.5): {samples_with_good_agreement/len(f1_scores)*100:.1f}%\n\n")
            
    f.write("COVERAGE ANALYSIS:\n")
    f.write(f"Samples with any human-LIME overlap: {samples_with_overlap/len(all_overlap_percentages)*100:.1f}%\n")
    f.write(f"Average overlap percentage: {np.mean(all_overlap_percentages):.1f}% ± {np.std(all_overlap_percentages):.1f}%\n")

print(f"Detailed analysis saved to: {results_file}")


# ---- 3.6 Overall summary ----

print(f"\n{'='*60}")
print("COMPREHENSIVE LIME EXPLAINABILITY ANALYSIS - COMPLETE")
print(f"{'='*60}")
print("This analysis evaluated LIME explanations using both qualitative and quantitative approaches:")
print()
print("ANALYSIS COMPONENTS:")
print("1. Detailed Case Studies: 3 randomly selected offensive examples with full analysis")
print("2. Population Statistics: Comprehensive metrics across all 525 offensive examples")
print("3. Visual Explanations: LIME plots showing feature importance")
print("4. Comprehensive Results File: All findings saved for further analysis")
print()
print("METRICS EVALUATED:")
print("• Faithfulness Metrics:")
print("  - Comprehensiveness: Performance drop when removing LIME's important words")
print("  - Sufficiency: Model performance using only LIME's important words")
print("• Plausibility Metrics:")
print("  - Token-level Precision/Recall/F1: Agreement with human annotations")
print("  - IOU F1: Intersection over Union with human rationales")
print("• Coverage Analysis:")
print("  - Human-LIME overlap rates and agreement patterns")
print()
print("RESEARCH CONTRIBUTIONS:")
print("• Quantified how well LIME explanations match expert human reasoning")
print("• Measured explanation faithfulness to actual model decision processes")
print("• Provided both statistical rigor and interpretable case studies")
print("• Established baseline metrics for explainable hate speech detection")
print()

print("LIME EXPLAINABILITY ANALYSIS COMPLETE!")
print("=" * 40)
print("OUTPUT FILES:")
print(f"✓ Visual explanations: lime_plot_sample_1.png, lime_plot_sample_2.png, lime_plot_sample_3.png")
print(f"✓ Comprehensive analysis: lime_analysis_results.txt")
print()
print("METRICS INCLUDED:")
print(f"✓ Population statistics from {len(offensive_indices)} samples")
print("✓ Faithfulness: comprehensiveness, sufficiency") 
print("✓ Plausibility: precision, recall, F1, IOU F1")
print("✓ Coverage: overlap rates, agreement patterns")
print()
print("Next steps: Review results file for detailed findings and population-level insights")
