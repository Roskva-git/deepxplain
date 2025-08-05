# Aligning Attention with Human Rationales for Self-Explaining Hate Speech Detection

Notes for the powerpoint presentation

- [Background](#background)
- [Explainability](#explainability)
- [Related work](#related-work)
- [Supervised rational attention](#supervised-rational-attention)
- [SRA framework](#sra-framework)
- [Workflow](#workflow)
- [Results 1](#results-1)
- [Results 2](#results-2)
- [Results 3](#results-3)
- [Conclusion](#conclusion)

## Background


## Explainability

## Related work

## Supervised rational attention

## SRA framework

## Workflow

## Results 1

## Results 2

## Results 3

## Conclusion


**SHAP** = Shapley Additive exPlanations

- Game theoretic approach to explain the output of machine learning models
- Shapley values always sum up to the difference between the game outcome when all players are present and the outcome when no players are present. Baseline vs current model (equivalent to explained variance in frequentist statistics?)
- I attempted to implement both LIME and SHAP for explainability analysis. While LIME worked successfully and provided meaningful insights that aligned with human rationales, SHAP encountered technical difficulties with the Portuguese BERT model tokenization. The LIME analysis alone provided comprehensive explanations showing how the model identifies offensive language patterns

**LIME** = Local Interpretable Model-agnostic Explanation

- Post-hoc method used to interpret predictions

**Post-Hoc methods**, why it could be helpful and why it's problematic

- Disadvantages of a post-hoc explanation is that we could simply be looking at correlations. Does this tell us how the model works or is it an explanation of the result without necessarily telling us anything about the process.

**BERT** = Bidirectional Encoder Representations from Transformers developed by (Devlin et al., 2019)
