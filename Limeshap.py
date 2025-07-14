## ---- 3.2 LIME Analysis Focused on Offensive Examples ----

## Chapter 3: LIME and SHAP

# Written by: Røskva
# Created: 10. July 2025
# Updated: 14. July 2025


print("\n3.2 LIME ANALYSIS - FOCUSING ON OFFENSIVE EXAMPLES WITH RATIONALES")
print()

import matplotlib.pyplot as plt

# Filter for offensive examples in the test set
print("Filtering test set for offensive examples...")
offensive_indices = []
for i, label in enumerate(test_dataset.labels):
    if label == 1:  # Offensive
        offensive_indices.append(i)

print(f"Found {len(offensive_indices)} offensive examples in test set of {len(test_labels)} total")
print("Selecting examples for detailed analysis...")
print()

# Select examples: prioritize offensive ones, include some neutral for comparison
np.random.seed(42)

# Get 2-3 offensive examples for detailed rationale comparison
if len(offensive_indices) >= 2:
    selected_offensive = np.random.choice(offensive_indices, min(3, len(offensive_indices)), replace=False)
else:
    selected_offensive = offensive_indices

# Get 1 neutral example for comparison analysis
neutral_indices = [i for i, label in enumerate(test_labels) if label == 0]
if len(neutral_indices) > 0:
    selected_neutral = [np.random.choice(neutral_indices, 1)[0]]
else:
    selected_neutral = []

# Combine for analysis
sample_indices = list(selected_offensive) + selected_neutral

print(f"Analysis plan:")
print(f"  - Offensive examples: {len(selected_offensive)} (with human rationales)")
print(f"  - Neutral examples: {len(selected_neutral)} (for comparison)")
print(f"  - Total examples: {len(sample_indices)}")
print()
print("Focus: Comparing LIME explanations with human rationales for offensive content")
print("Rationale: Only offensive examples have human annotations for comparison")
print()

for i, idx in enumerate(sample_indices):
    is_offensive_example = test_dataset.labels[idx] == 1
    
    print(f"{'='*70}")
    print(f"SAMPLE {i+1}/{len(sample_indices)} (Test Index: {idx})")
    print(f"Example Type: {'OFFENSIVE' if is_offensive_example else 'NEUTRAL'}")
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
    
    if is_offensive_example:
        print(f"Human Rationales (what humans identified as offensive):")
        print(f"   Annotator 1: \"{rationale_1}\"")
        print(f"   Annotator 2: \"{rationale_2}\"")
        print("   Note: These are the ground truth annotations for comparison")
    else:
        print(f"Human Rationales: Not available for neutral examples")
        print("   Note: Only offensive examples have rationale annotations")
    print()
    
    print(f"Ground Truth Label: {true_label} ({'Offensive' if true_label == 1 else 'Neutral'})")
    print()
    
    # Get model prediction using our prediction function
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
    print("Generating LIME explanation to understand model's reasoning...")
    
    # Generate LIME explanation using our explain_instance function
    lime_explanation = explain_instance(text, model, num_features=8)
    lime_features = lime_explanation.as_list()
    
    print(f"Top {min(8, len(lime_features))} influential words (sorted by absolute importance):")
    for j, (feature, weight) in enumerate(lime_features[:8], 1):
        if weight > 0:
            direction = f"-> Offensive (+{weight:.3f})"
        else:
            direction = f"-> Neutral ({weight:.3f})"
        print(f"   {j:2d}. '{feature}': {direction}")
    
    # Extract words that pushed toward offensive classification
    lime_offensive_words = [word for word, weight in lime_features if weight > 0]
    lime_neutral_words = [word for word, weight in lime_features if weight < 0]
    
    print(f"Summary: {len(lime_offensive_words)} words pushed toward Offensive, {len(lime_neutral_words)} toward Neutral")
    print()
    
    # ============ ANALYSIS BASED ON EXAMPLE TYPE ============
    if is_offensive_example:
        print(f"OFFENSIVE EXAMPLE ANALYSIS:")
        print("-" * 40)
        print("Comparing LIME explanations with human rationales...")
        
        # Get human rationale words for comparison
        rationale_words = set()
        if rationale_1 != 'N/A':
            rationale_words.update(rationale_1.lower().split())
        if rationale_2 != 'N/A':
            rationale_words.update(rationale_2.lower().split())
        
        print(f"LIME identified as offensive: {lime_offensive_words[:5]}")
        print(f"Human rationale words: {list(rationale_words)}")
        
        # Check overlap between LIME and human rationales
        lime_words_lower = [word.lower() for word in lime_offensive_words[:5]]
        lime_human_overlap = [word for word in lime_words_lower if word in rationale_words]
        
        print(f"LIME-Human overlap: {lime_human_overlap}")
        
        if lime_human_overlap:
            overlap_count = len(lime_human_overlap)
            lime_count = len(lime_words_lower)
            print(f"Agreement: LIME matches human reasoning on {overlap_count} word(s)")
            print(f"Overlap rate: {overlap_count}/{lime_count} = {overlap_count/max(lime_count,1)*100:.1f}% of LIME's top words")
        else:
            print(f"Disagreement: LIME focused on different words than human annotators")
            print(f"This could indicate: model bias, different reasoning strategies, or annotation gaps")
        
        # Assess model prediction quality
        if pred_label == 1 and lime_offensive_words:
            print(f"Model assessment: Correctly identified offensive content with clear reasoning")
        elif pred_label == 1 and not lime_offensive_words:
            print(f"Model assessment: Predicted offensive but LIME found no clear offensive words")
        elif pred_label == 0:
            print(f"Model assessment: Model missed offensive content that humans identified")
            
    else:
        print(f"NEUTRAL EXAMPLE ANALYSIS:")
        print("-" * 35)
        print("Checking if LIME correctly identifies non-offensive content...")
        
        print(f"LIME identified as offensive: {lime_offensive_words[:3]}")
        print(f"LIME identified as neutral: {lime_neutral_words[:3]}")
        
        # Assess LIME's behavior on neutral text
        strong_offensive = [w for w, weight in lime_features if weight > 0.3]
        
        if len(lime_offensive_words) == 0:
            print(f"LIME assessment: Correctly found no offensive words")
        elif len(strong_offensive) == 0:
            print(f"LIME assessment: Found only weak offensive signals - reasonable for neutral text")
        else:
            print(f"LIME assessment: Found strong offensive words in neutral text - potential false positive")
        
        # Assess model prediction
        if pred_label == 0:
            print(f"Model assessment: Correctly classified as neutral")
        else:
            print(f"Model assessment: Incorrectly classified neutral text as offensive")
    
    print()
    
    # ============ SAVE LIME PLOT ============
    print("Saving LIME visualization...")
    
    # Create LIME plot using built-in visualization
    fig = lime_explanation.as_pyplot_figure()
    plt.title(f'LIME Analysis - Sample {i+1}: {"Offensive" if is_offensive_example else "Neutral"} Example')
    lime_plot_file = f"{TRAINING_OUTPUT_DIR}/lime_plot_sample_{i+1}.png"
    plt.savefig(lime_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"LIME plot saved: lime_plot_sample_{i+1}.png")
    print()

# ============ SAVE TEXT SUMMARY ============
print("\nSaving analysis summary to text file...")

# Collect all analysis results for saving
analysis_results = []

# Note: We need to recollect the analysis data since it was processed in the loop above
# For now, we'll create a summary structure
summary_data = {
    'offensive_count': len(selected_offensive),
    'neutral_count': len(selected_neutral),
    'total_examples': len(sample_indices),
    'sample_indices': sample_indices,
    'analysis_timestamp': str(np.datetime64('now'))
}

# Save comprehensive text summary
summary_file = f"{TRAINING_OUTPUT_DIR}/lime_analysis_summary.txt"
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("LIME ANALYSIS WITH HUMAN RATIONALE COMPARISON\n")
    f.write("="*60 + "\n\n")
    
    f.write("ANALYSIS OVERVIEW\n")
    f.write("-"*20 + "\n")
    f.write(f"Date: {summary_data['analysis_timestamp']}\n")
    f.write(f"Offensive examples analyzed: {summary_data['offensive_count']}\n")
    f.write(f"Neutral examples analyzed: {summary_data['neutral_count']}\n")
    f.write(f"Total examples: {summary_data['total_examples']}\n\n")
    
    f.write("METHODOLOGY\n")
    f.write("-"*15 + "\n")
    f.write("- Primary focus on offensive examples due to human rationale availability\n")
    f.write("- Neutral examples included to verify model doesn't over-detect offensive content\n")
    f.write("- Human rationales provide ground truth for what should be considered offensive\n")
    f.write("- LIME explanations reveal which features the model actually learned to focus on\n")
    f.write("- Comparison shows whether model reasoning aligns with human judgment\n\n")
    
    f.write("RESEARCH QUESTIONS ADDRESSED\n")
    f.write("-"*30 + "\n")
    f.write("1. Does LIME identify the same offensive words as human annotators?\n")
    f.write("2. Can LIME explanations help us validate model predictions?\n")
    f.write("3. Are there systematic differences in how humans vs. models identify hate speech?\n")
    f.write("4. Does the model correctly avoid false positives on neutral content?\n\n")
    
    f.write("DETAILED RESULTS\n")
    f.write("-"*20 + "\n")
    f.write("Note: Detailed analysis for each example was printed to terminal during execution.\n")
    f.write("Key findings include feature importance rankings, rationale comparisons,\n")
    f.write("and assessment of model reasoning quality for each analyzed example.\n\n")
    
    f.write("OUTPUT FILES GENERATED\n")
    f.write("-"*25 + "\n")
    for i in range(len(sample_indices)):
        f.write(f"- lime_plot_sample_{i+1}.png: Visual LIME explanation for sample {i+1}\n")
    f.write(f"- lime_analysis_summary.txt: This summary file\n\n")
    
    f.write("NEXT STEPS\n")
    f.write("-"*15 + "\n")
    f.write("1. Review individual LIME plots to understand feature importance patterns\n")
    f.write("2. Analyze agreement/disagreement patterns between LIME and human rationales\n")
    f.write("3. Look for systematic biases or unexpected feature focus in model reasoning\n")
    f.write("4. Consider implications for model trustworthiness and deployment\n")

print(f"Analysis summary saved to: {summary_file}")
print()

# ============ SAVE DETAILED RESULTS (if we want to re-run analysis for saving) ============
print("Saving detailed results...")

# Create a more detailed results file
detailed_file = f"{TRAINING_OUTPUT_DIR}/lime_detailed_results.txt"
with open(detailed_file, 'w', encoding='utf-8') as f:
    f.write("DETAILED LIME ANALYSIS RESULTS\n")
    f.write("="*40 + "\n\n")
    
    f.write("This file contains detailed results that were displayed during the analysis.\n")
    f.write("For complete analysis including rationale comparisons, refer to terminal output.\n\n")
    
    f.write("SAMPLE INDICES ANALYZED\n")
    f.write("-"*25 + "\n")
    for i, idx in enumerate(sample_indices):
        is_offensive = test_labels[idx] == 1
        f.write(f"Sample {i+1}: Test index {idx} ({'Offensive' if is_offensive else 'Neutral'})\n")
    
    f.write(f"\nSamples were selected to prioritize offensive examples (which have human rationales)\n")
    f.write(f"while including neutral examples for comparison analysis.\n\n")
    
    f.write("ANALYSIS APPROACH\n")
    f.write("-"*20 + "\n")
    f.write("For each sample:\n")
    f.write("1. Generated LIME explanation showing feature importance\n")
    f.write("2. Compared LIME features with human rationales (for offensive examples)\n")
    f.write("3. Assessed model prediction quality and reasoning\n")
    f.write("4. Saved visual explanation as PNG file\n\n")
    
    f.write("KEY INSIGHTS\n")
    f.write("-"*15 + "\n")
    f.write("- LIME provides interpretable explanations for model decisions\n")
    f.write("- Human rationales serve as ground truth for feature importance\n")
    f.write("- Comparison reveals whether model learned appropriate features\n")
    f.write("- Analysis helps identify potential biases or reasoning errors\n")

print(f"Detailed results saved to: {detailed_file}")
print()

# ============ SUMMARY OF FINDINGS ============
print(f"\n{'='*60}")
print("SUMMARY: LIME ANALYSIS WITH HUMAN RATIONALE COMPARISON")
print(f"{'='*60}")

offensive_count = len(selected_offensive)
neutral_count = len(selected_neutral)

print(f"Analysis Overview:")
print(f"  - Offensive examples analyzed: {offensive_count}")
print(f"  - Neutral examples analyzed: {neutral_count}")
print(f"  - Total examples: {len(sample_indices)}")
print()

print(f"Research Questions Addressed:")
print(f"  1. Does LIME identify the same offensive words as human annotators?")
print(f"  2. Can LIME explanations help us validate model predictions?")
print(f"  3. Are there systematic differences in how humans vs. models identify hate speech?")
print(f"  4. Does the model correctly avoid false positives on neutral content?")
print()

print(f"Methodology Notes:")
print(f"  - Primary focus on offensive examples due to rationale availability")
print(f"  - Neutral examples included to verify model doesn't over-detect")
print(f"  - Human rationales provide ground truth for feature importance")
print(f"  - LIME explanations reveal what the model actually learned")
print()

print(f"Output Files:")
print(f"  - Visual explanations: lime_plot_sample_1.png through lime_plot_sample_{len(sample_indices)}.png")
print(f"  - Text summary: lime_analysis_summary.txt")
print(f"  - Detailed results: lime_detailed_results.txt")
print(f"  - Each plot shows feature importance with LIME's built-in visualization")
print()

print("LIME analysis complete!")
print("Next steps: Review plots and text files to analyze patterns in model reasoning")