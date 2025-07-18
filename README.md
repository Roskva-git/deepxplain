# deepxplain
Summer project 2025

## Explainable AI for Hate Speech Detection
This repository contains my work on explainable AI techniques for hate speech detection in Portuguese text.

## Overview
We are exploring how LIME and SHAP can help explain machine learning models that detect hate speech, with a focus on Brazilian Portuguese social media data.

## Files
- `deepxplain-notes.md` - Research notes, definitions, and reflections
  
- `sentiment_analysis_with_emoji.csv` - Analysis results and model outputs for original data
- `sentiment_analysis_no_emoji.csv` - Analysis results and model outputs for cleaned data
- `LIME_comparison table.csv` - Results from LIME analyses
  
- `sentiment_analysis_preproc.py` - Code for preprocessing the dataset before sentiment analysis
- `foxready_sentiment_analysis_training.py` - Code for sentiment analysis
- `Limeshap.py` - Code for implementing LIME
- `clear_emojis_from_data.py` - Cleans the orginial dataset for emojis, necessary for SHAP
- `shapanalysis.py` - Code for implementing SHAP


## Current Status
Successfully implemented LIME for model explanation. SHAP implementation currently in progress. 
