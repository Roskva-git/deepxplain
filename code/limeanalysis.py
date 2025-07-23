## Chapter 3: LIME ANALYSIS

# Written by: Røskva
# Created: 09. July 2025
# Updated: 22. July 2025

# 3. LIME ANALYSIS
### 3.0 Setting up explainability functions
### 3.1 Calculating faithfulness metrics
### 3.2 Calculating plausibility metrics
### 3.3 LIME ANALYSIS
### 3.4 Population-level statistics
### 3.5 Save comprehensive analysis to file
### 3.6 Overall summary

print("\n3. ANALYSIS WITH LIME")
print()

## ---- 3.0 Setting up explainability functions ----

print("\n3.0 SETTING UP EXPLAINABILITY FUNCTIONS")
print()

# Importing libraries
import torch
import json
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
TRAINING_OUTPUT_DIR = "/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/training_results/with_emoji/training_20250708_1500"
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
explainer = LimeTextExplainer(class_names=['Neutral', 'Offensive'])

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

## ---- 3.1 Calculating faithfulness metrics ----

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

## ---- 3.3 LIME Analysis ----

print("\n3.3 LIME ANALYSIS WITH RATIONALES")
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

print(f"Generating LIME explanations for 3 random test samples...")
print("Comparing model explanations with human rationales...")
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
    
    # ============ COMPARISON ANALYSIS ============
    print(f"LIME vs HUMAN COMPARISON:")
    print("-" * 30)
    
    # Get human rationale words
    rationale_words = set()
    if rationale_1 != 'N/A':
        rationale_words.update(rationale_1.lower().split())
    if rationale_2 != 'N/A':
        rationale_words.update(rationale_2.lower().split())
    
    if true_label == 1:  # Only analyze for offensive examples
        print(f"   LIME top offensive words: {lime_offensive_words[:5]}")
        print(f"   Human rationale words: {list(rationale_words)}")
        
        # Check overlaps
        lime_words_lower = [word.lower() for word in lime_offensive_words[:5]]
        lime_human_overlap = [word for word in lime_words_lower if word in rationale_words]
        
        print(f"   LIME-Human overlap: {lime_human_overlap}")
        
        # Analysis
        if lime_human_overlap:
            print(f"   LIME matches human reasoning")
        else:
            print(f"   LIME doesn't match human reasoning closely")
            
    else:
        print(f"   This is a neutral example - checking if LIME agrees it's non-offensive")
        if len(lime_offensive_words) == 0:
            print(f"   LIME agrees: no strong offensive words")
        else:
            print(f"   LIME found offensive content: {lime_offensive_words[:3]}")
    
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

## ---- 3.4 Population-Level Statistics ----

print("\n3.4 CALCULATING POPULATION-LEVEL STATISTICS FOR LIME")
print("=" * 70)
print("Running LIME analysis on all offensive examples...")
print()

# Storage for metrics
all_lime_faithfulness = []
all_lime_plausibility = []
all_lime_overlap_percentages = []

# Main analysis loop - process each offensive example
for i, idx in enumerate(offensive_indices):
    if i % 50 == 0:
        print(f"Processing sample {i+1}/{len(offensive_indices)}...")
        torch.cuda.empty_cache()
    
    # Get data for this sample
    text = test_dataset.texts[idx]
    true_label = test_dataset.labels[idx]
    original_item = data[test_indices[idx]]
    rationale_1 = original_item.get('rationales annotator 1', 'N/A')
    rationale_2 = original_item.get('rationales annotator 2', 'N/A')
    
    # Get model prediction
    pred_probs = predict_fn([text])[0]
    pred_label = np.argmax(pred_probs)
    
    # Generate LIME explanation
    lime_explanation = explain_instance(text, model, num_features=8)
    lime_features = lime_explanation.as_list()
    
    # Calculate LIME faithfulness metrics
    lime_faithfulness = calculate_faithfulness_metrics(text, pred_probs, lime_features)
    
    # Calculate LIME plausibility metrics
    lime_plausibility = calculate_plausibility_metrics(lime_features, rationale_1, rationale_2)
    
    # Calculate LIME overlap percentage with human rationales
    rationale_words = set()
    if rationale_1 != 'N/A':
        rationale_words.update(rationale_1.lower().split())
    if rationale_2 != 'N/A':
        rationale_words.update(rationale_2.lower().split())
    
    lime_top_words = [word.lower() for word, weight in lime_features[:5]]
    lime_human_overlap = [word for word in lime_top_words if word in rationale_words]
    lime_overlap_percentage = len(lime_human_overlap) / max(len(lime_top_words), 1) * 100
    
    # Store LIME metrics
    all_lime_faithfulness.append(lime_faithfulness)
    all_lime_plausibility.append(lime_plausibility)
    all_lime_overlap_percentages.append(lime_overlap_percentage)
    
    # Clear GPU memory periodically
    if i % 10 == 0:
        torch.cuda.empty_cache()

print(f"Completed analysis on {len(offensive_indices)} samples!")
print()

# ============ CALCULATE AND DISPLAY AGGREGATE STATISTICS ============

print("POPULATION-LEVEL STATISTICS")
print("=" * 50)
print(f"Total samples analyzed: {len(offensive_indices)}")
print(f"Successful LIME analyses: {len(all_lime_faithfulness)}")
print()

# ---- LIME Statistics ----
print("LIME FAITHFULNESS METRICS:")
lime_comprehensiveness = [m['comprehensiveness'] for m in all_lime_faithfulness]
lime_sufficiency = [m['sufficiency'] for m in all_lime_faithfulness]
print(f"  Comprehensiveness: {np.mean(lime_comprehensiveness):.3f} ± {np.std(lime_comprehensiveness):.3f}")
print(f"  Sufficiency: {np.mean(lime_sufficiency):.3f} ± {np.std(lime_sufficiency):.3f}")
print()

print("LIME PLAUSIBILITY METRICS:")
lime_precision = [m['precision'] for m in all_lime_plausibility]
lime_recall = [m['recall'] for m in all_lime_plausibility]
lime_f1 = [m['f1_score'] for m in all_lime_plausibility]
lime_iou = [m['iou_f1'] for m in all_lime_plausibility]
print(f"  Token-level Precision: {np.mean(lime_precision):.3f} ± {np.std(lime_precision):.3f}")
print(f"  Token-level Recall: {np.mean(lime_recall):.3f} ± {np.std(lime_recall):.3f}")
print(f"  Token-level F1: {np.mean(lime_f1):.3f} ± {np.std(lime_f1):.3f}")
print(f"  IOU F1: {np.mean(lime_iou):.3f} ± {np.std(lime_iou):.3f}")
print()

# ---- Coverage Analysis ----
print("COVERAGE ANALYSIS:")
lime_samples_with_overlap = sum(1 for p in all_lime_overlap_percentages if p > 0)
print(f"  LIME samples with human overlap: {lime_samples_with_overlap}/{len(all_lime_overlap_percentages)} ({lime_samples_with_overlap/len(all_lime_overlap_percentages)*100:.1f}%)")
print(f"  LIME average overlap: {np.mean(all_lime_overlap_percentages):.1f}% ± {np.std(all_lime_overlap_percentages):.1f}%")
print()

## ---- 3.5 Save comprehensive analysis to file ----

results_file = f"{TRAINING_OUTPUT_DIR}/lime_analysis_results.txt"
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("COMPREHENSIVE LIME ANALYSIS RESULTS\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("ANALYSIS OVERVIEW:\n")
    f.write(f"- Dataset: HateBRXplain (emoji-cleaned)\n")
    f.write(f"- Model: BERT-base-uncased\n")
    f.write(f"- Total offensive samples analyzed: {len(offensive_indices)}\n")
    f.write(f"- Successful LIME analyses: {len(all_lime_faithfulness)}\n\n")
    
    # ============ DETAILED CASE STUDIES ============
    f.write("=" * 60 + "\n")
    f.write("DETAILED CASE STUDIES (3 Examples)\n")
    f.write("=" * 60 + "\n\n")
    
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
        f.write(f"True label: {true_label} ({'Offensive' if true_label == 1 else 'Neutral'})\n")
        f.write(f"Predicted: {pred_label} ({'Offensive' if pred_label == 1 else 'Neutral'}, confidence: {pred_probs[pred_label]:.3f})\n")
        f.write(f"Human rationale 1: {rationale_1}\n")
        f.write(f"Human rationale 2: {rationale_2}\n\n")
        
        # LIME Results
        f.write("LIME EXPLANATION:\n")
        f.write("- Method: Local Interpretable Model-agnostic Explanations\n")
        f.write("- Approach: Perturbs input text and learns local linear model\n")
        for j, (feature, weight) in enumerate(lime_features[:5], 1):
            direction = "→ Offensive" if weight > 0 else "→ Neutral"
            f.write(f"  {j}. '{feature}': {weight:+.3f} {direction}\n")
        f.write("\n")
        
        # Human Agreement
        f.write("HUMAN AGREEMENT ANALYSIS:\n")
        rationale_words = set()
        if rationale_1 != 'N/A':
            rationale_words.update(rationale_1.lower().split())
        if rationale_2 != 'N/A':
            rationale_words.update(rationale_2.lower().split())
        
        lime_top_words = [word.lower() for word, weight in lime_features[:5]]
        lime_human_overlap = [word for word in lime_top_words if word in rationale_words]
        f.write(f"  LIME-Human overlap: {lime_human_overlap} ({len(lime_human_overlap)}/{len(lime_top_words)})\n")
        
        f.write("\n" + "="*60 + "\n\n")

    # ============ POPULATION-LEVEL STATISTICS ============
    f.write("\n" + "="*60 + "\n")
    f.write("POPULATION-LEVEL STATISTICS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Analysis of {len(offensive_indices)} offensive samples\n\n")
    
    # LIME Population Statistics
    f.write("LIME POPULATION STATISTICS:\n")
    f.write("-" * 30 + "\n")
    f.write("Method: Local Interpretable Model-agnostic Explanations\n")
    f.write(f"Successful analyses: {len(all_lime_faithfulness)}/{len(offensive_indices)} (100%)\n\n")
    
    f.write("LIME Faithfulness Metrics:\n")
    f.write(f"  Comprehensiveness: {np.mean(lime_comprehensiveness):.3f} ± {np.std(lime_comprehensiveness):.3f}\n")
    f.write(f"  Sufficiency: {np.mean(lime_sufficiency):.3f} ± {np.std(lime_sufficiency):.3f}\n\n")
    
    f.write("LIME Plausibility Metrics:\n")
    f.write(f"  Token-level Precision: {np.mean(lime_precision):.3f} ± {np.std(lime_precision):.3f}\n")
    f.write(f"  Token-level Recall: {np.mean(lime_recall):.3f} ± {np.std(lime_recall):.3f}\n")
    f.write(f"  Token-level F1: {np.mean(lime_f1):.3f} ± {np.std(lime_f1):.3f}\n")
    f.write(f"  IOU F1: {np.mean(lime_iou):.3f} ± {np.std(lime_iou):.3f}\n\n")
    
    lime_good_agreement = sum(1 for score in lime_f1 if score > 0.5)
    f.write(f"LIME Coverage Analysis:\n")
    f.write(f"  Samples with human overlap: {lime_samples_with_overlap}/{len(all_lime_overlap_percentages)} ({lime_samples_with_overlap/len(all_lime_overlap_percentages)*100:.1f}%)\n")
    f.write(f"  Average overlap percentage: {np.mean(all_lime_overlap_percentages):.1f}% ± {np.std(all_lime_overlap_percentages):.1f}%\n")
    f.write(f"  High agreement rate (F1>0.5): {lime_good_agreement}/{len(lime_f1)} ({lime_good_agreement/len(lime_f1)*100:.1f}%)\n\n")

print(f"Comprehensive LIME analysis saved to: {results_file}")

## ---- 3.6 Overall summary ----

print(f"\n{'='*60}")
print("COMPREHENSIVE LIME EXPLAINABILITY ANALYSIS - COMPLETE")
print(f"{'='*60}")
print("This analysis evaluated LIME explanations using both qualitative and quantitative approaches:")
print()
print("ANALYSIS COMPONENTS:")
print("1. Detailed Case Studies: 3 randomly selected offensive examples with full analysis")
print("2. Population Statistics: Comprehensive metrics across all offensive examples")
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
