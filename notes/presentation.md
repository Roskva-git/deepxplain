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
Meta -> seven mill hate speech appeals monthly, YouTube -> 98% of violent extremist content is flagged by machine learning algorithms.

OL is confrontational, rude or aggressive, HS -> directed at target. Can be explicit or implicit.

Bias in all steps. Reporting, datasets, underfitting, 
Automated classifiers could reinforce societal biases and marginalize vulnerable groups (ethnic stereotypes, negative sentiments about disabled etc)

BB: A bit like behaviorism in psychology. You see the input and the output, but not the reasoning. Problems with debugging and trust.

Post-hoc: Shows what happens when you influence the output, but not what the model used to arrive at a conclusion

Attention: BERT has 144 heads focusing on different things, often unclear what mattered for the final decision
May focus heavily on identity terms (mention vs target), whereas we would focus more on context and intent

## Explainability
GDPR article 15.1h:
The data subject shall have the right to obtain from the controller confirmation as to whether or not personal data concerning him or her are being processed, 
and, where that is the case, access to the personal data and the following information:
[a-g] h: the existence of automated decision-making, including profiling, referred to in Article 22(1) and (4) and, at least in those cases, 
**meaningful information about the logic involved, as well as the significance and the envisaged consequences of such processing for the data subject**

and article 22

Accountability: Can verify decisions through explanations. 
Trace decision back to causes and tell who/what is reponsible.
Enables corrections.

Fairness: can address hidden biases in classification (good classification score, but always flagging minority mentions).
Disparate impact analysis, does it require more evidence to flag hate speech against one group than another?
Builds trust because we know why the model acts the way it does.



## Related work

## Supervised rational attention

## SRA framework

## Workflow

## Results 1
Performance comparison of hate speech detection models on the HateXplain test set. 
Best results are in bold, second-best are underlined.

Classification performance (Accuracy, Macro F1, AUROC)
Explainability metrics (IoU F1, Token F1, AUPRC), 
Fairness metrics (GMB-Subgroup, GMB-BPSN, GMB-BNSP AUCs), and 
faithfulness metrics (Comprehensiveness, Sufficiency). 

Higher values are better for all metrics except Sufficiency (lower is better). 
Models with [LIME] use post-hoc explanations, while [Attn] indicates attention-based explanations. 

SRA achieves the best AUROC, IoU F1, and Token F1 scores among all methods and
the best GMB-BNSP fairness metric, also competitive classification and faithfulness performance. 

SRA results are averaged across 5 random seeds with standard deviations shown in
parentheses.

## Results 2
Table 2: Comparison of explainability methods (Portuguese)
Evaluate post-hoc explanation methods (LIME, SHAP) against intrinsic SRA

Explainability metrics (IoU F1, Token Precision/Recall/F1) measure alignment with human rationales.

Faithfulness metrics (comp, suff) assess if explanations reflect actual model reasoning. 

Higher values indicate better performance, except Sufficiency (lower is better). 

SRA provides explanations intrinsically during prediction, while LIME andSHAP require additional post-hoc computation. 

SRA has better token precision (0.935) and good overall performance, and provides real-time explanations. 
SRA results are averaged across 5 random seeds with standard deviations.

## Results 3

## Conclusion
