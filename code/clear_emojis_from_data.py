# Simple Emoji Removal Script for SHAP Analysis

"""
Minimal Emoji Removal Script
Only removes emojis, keeps everything else intact (accents, punctuation, etc.)

Author: Røskva  
Created: 16. July 2025
Purpose: Minimal cleaning for SHAP compatibility while preserving text integrity
"""

import json
import re
import os

def remove_emojis_only(text):
    """
    Remove ONLY emojis, keep everything else exactly as is
    Preserves: accents, punctuation, special characters, spacing
    """
    if not text or text == 'N/A':
        return text
    
    # Comprehensive emoji pattern - covers all major emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "\U0001F200-\U0001F2FF"  # enclosed CJK letters and months
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # geometric shapes extended
        "\U0001F800-\U0001F8FF"  # supplemental arrows-C
        "\U00002B50-\U00002B55"  # stars
        "]+", 
        flags=re.UNICODE
    )
    
    # Replace emojis with single space and clean up multiple spaces
    cleaned = emoji_pattern.sub(' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces -> single space
    cleaned = cleaned.strip()
    
    return cleaned

def clean_dataset_minimal():
    """Clean dataset by removing only emojis"""
    
    # File paths
    input_path = '/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/dataset/HateBRXplain.json'
    output_path = '/fp/projects01/ec35/homes/ec-roskvatb/deepxplain/data/dataset/HateBRXplain_no_emojis.json'
    
    # Safety checks
    if not os.path.exists(input_path):
        print(f"Error: Original file not found: {input_path}")
        return
    
    if os.path.exists(output_path):
        response = input(f"File already exists: {output_path}\nOverwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    print(f"🔄 Loading dataset from: {input_path}")
    
    # Load data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items")
    except Exception as e:
        print(f"Error loading: {e}")
        return
    
    # Process data
    print("Removing emojis only...")
    
    emoji_found_count = 0
    changed_count = 0
    
    for i, item in enumerate(data):
        if i % 1000 == 0:
            print(f"   Processing {i}/{len(data)}...")
        
        # Clean main comment
        original = item.get('comment', '')
        cleaned = remove_emojis_only(original)
        
        if cleaned != original:
            changed_count += 1
            # Count items that actually had emojis
            if re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', original):
                emoji_found_count += 1
        
        item['comment'] = cleaned
        
        # Clean rationales
        for key in ['rationales annotator 1', 'rationales annotator 2']:
            if key in item and item[key] != 'N/A':
                original_rationale = item[key]
                cleaned_rationale = remove_emojis_only(original_rationale)
                item[key] = cleaned_rationale
    
    print(f"Processing complete!")
    print(f"Items with emojis: {emoji_found_count}")
    print(f"Items changed: {changed_count}")
    
    # Save
    print(f"Saving to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved successfully!")
        
        # File size comparison
        orig_size = os.path.getsize(input_path) / (1024*1024)
        new_size = os.path.getsize(output_path) / (1024*1024)
        print(f"Sizes: {orig_size:.1f}MB → {new_size:.1f}MB")
        
    except Exception as e:
        print(f"Save error: {e}")
        return
    
    print(f"\nDone! Minimal cleaning complete.")
    print(f"Dataset ready for both LIME and SHAP analysis")

def test_cleaning():
    """Test the cleaning function with examples"""
    test_cases = [
        "Grande Merda 💩💩💩💩💩💩",
        "Que vontade de dar um abraço nos dois! ❤️",
        "A máscara dessa traidora Joice Falsemam durou menos",
        "Não é possível! Está tão difícil... 😭",
        "Normal text without any emojis",
        "Acentos são preservados: ção, não, coração"
    ]
    
    print("TESTING EMOJI REMOVAL:")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        cleaned = remove_emojis_only(text)
        changed = " ✓" if cleaned != text else ""
        print(f"{i}. Original: {text}")
        print(f"   Cleaned:  {cleaned}{changed}")
        print()

if __name__ == "__main__":
    print("🧹 MINIMAL EMOJI REMOVAL FOR SHAP")
    print("=" * 40)
    print("This script ONLY removes emojis, preserving:")
    print("Portuguese accents (ção, não, etc.)")
    print("All punctuation")
    print("Special characters")
    print("Original spacing and formatting")
    print()
    
    # Show test examples
    test_cleaning()
    
    # Run cleaning
    response = input("Proceed with minimal cleaning? (y/n): ")
    if response.lower() == 'y':
        clean_dataset_minimal()
    else:
        print("Cancelled.")
