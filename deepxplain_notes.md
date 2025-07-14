# Contents

- [Explanations](#explanations)
- [Hate speech](#hate-speech)
- [Understanding attention](#understanding-attention)
- [Reflections](#reflections)
- [Bibliography](#bibliography)

## Explanations

**SHAP** = Shapley Additive exPlanations

- Game theoretic approach to explain the output of machine learning models
- Shapley values always sum up to the difference between the game outcome when all players are present and the outcome when no players are present. Baseline vs current model (equivalent to explained variance in frequentist statistics?)
- I attempted to implement both LIME and SHAP for explainability analysis. While LIME worked successfully and provided meaningful insights that aligned with human rationales, SHAP encountered technical difficulties with the Portuguese BERT model tokenization. The LIME analysis alone provided comprehensive explanations showing how the model identifies offensive language patterns

**LIME** = Local Interpretable Model-agnostic Explanation

- Post-hoc method used to interpret predictions

**Post-Hoc methods**, why it could be helpful and why it's problematic

- Disadvantages of a post-hoc explanation is that we could simply be looking at correlations. Does this tell us how the model works or is it an explanation of the result without necessarily telling us anything about the process.

**BERT** = Bidirectional Encoder Representations from Transformers developed by (Devlin et al., 2019)

- Encoder only
- BERT is a pre-trained bidirectional language representation model which uses a fine-tuning approach to apply pre-trained language representations to downstream tasks. Prior to BERT, the standard approach for pre-training language models was using a left-to-right architecture, where every token can only attend to previous tokens in the self-attention layers. This directional restriction is sub-optimal for sentence-level tasks, like question answering and other heavily context-dependent tasks such as hate speech detection. BERT circumvents unidirectionality constraints by using a "masked language model" (MLM) pre-training objective, inspired by the Cloze task, where randomly masked tokens are predicted based on bidirectional context. While the model is pre-trained on unlabelled text, the fine-tuning is done with labelled datasets specific to the target task, such as hate speech detection in our case.
- Language specific BERT models have been created, specifically the Brazilian-Portuguese ones we have used in this study (INSERT CITATION)

**Faithfulness** (DeYoung et al., 2020; Jacovi & Goldberg, 2020)

- Comprehensiveness
- Sufficiency

**Plausibility** (DeYoung et al., 2020)

- IOU F1 score
- Token-level precision
- F1 score

**Sentiment analysis**

## Hate speech

Introducing the HateXplain dataset, with a mission to reduce unintended bias toward communities who are targets of hate speech, by incorporating human rationales behind hate speech classification (Mathew et al., 2021)

- Problems with the dataset: Annotated by Mturk workers. Inter-rater agreement of 0.46, and we found many instances that were wrong.

Uncover biased text features by applying explanation techniques (Gongane et al., 2024)

Current context is hate speech in Brazil (in Portuguese, a low resource language) (Salles et al., 2025)

**HateBRXplain corpus** (Vargas et al., 2025; Vargas et al., 2022)

- The HateBRXplain corpus comprises a collection of 7000 Instagram comments collected from the posts of Brazilian politicians. The posts are manually annotated as either "offensive" or "non-offensive", with offensiveness labelled as highly, moderately or slightly offensive. Nine forms of hate speech were identified, namely xenophobia, racism, homophobia, sexism, religious intolerance, partyism, apology to dictatorship, antisemitism and fat-phobia.

## Understanding attention

What is attended in BERT? Which tokens are attended, 61,7% attention on CLS. BERT naturally performs very well even without further fine-tuning. (INSERT BRAGES EXPLANATION). Summary vector CLS?

## Reflections

1. **What are the advantages of incorporating human-based rationales into the models?**
   - Value-based choices (normative answers)

2. **Why should we provide explainable models for sensitive tasks such as hate speech detection?**
   - Avoid bias (ethnic stereotypes, over-sensitivity to group identifiers)

3. **There are well-known metrics to evaluate improvements in model performance, but how can we measure gains in interpretability?** How do we evaluate the quality of the explanations? (e.g., *Faithfulness*: comprehensiveness, sufficiency; *Plausibility*: IOU F1-score, token-level precision, recall, and F1-score).

4. **We need to discuss the assumptions we are making about our research. Why are we doing this at all?**

Imposing human based rationales on statistical learning models can arguably be necessary with the types of classifications we are making in this project. Hate speech is defined by what sort of values we have in society, it is a fluent property and not an objective truth. By having human annotators single out the words that make a text offensive or inoffensive, we have a greater chance of making our models pick up on the right cues for what makes a specific utterance hateful. We can avoid more random rationales that often come with machine learning because they are good at picking up any pattern, even ones that we don't see as meaningful. This is an advantage in some settings, but when it comes to value-based choices, we need to be in control of the rationales.

Explainability in the process of detecting hate speech is important because democracy vs freedom of speech.

It is also important to note that hate speech is heavily dependent on the context of the utterance, and of the sender. Hate speech expresses hate or encourages discrimination against a person or a group based on characteristics such as gender, political affiliation, race, sexuality and so on. However, there are certain important nuances and distinctions to make. There are many examples of minority groups reclaiming words or phrases that have been used in a negative fashion by mainstream society, such as black communities with the word "nigger" and LGBT communities with words such as "queer" or "gay". We need to be sensitive to this, as both human annotators and a machine learning model run the risk of defining words such as group identifiers as offensive, when it is in fact context dependent. Implementing hate speech detection to censor public communication could in the worst-case lead to discrimination of minorities, which is exactly what we want to avoid.

## Bibliography

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (Version 2). arXiv. https://doi.org/10.48550/ARXIV.1810.04805

DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., & Wallace, B. C. (2020). *ERASER: A Benchmark to Evaluate Rationalized NLP Models* (Version 2). arXiv. https://doi.org/10.48550/ARXIV.1911.03429

Gongane, V. U., Munot, M. V., & Anuse, A. D. (2024). A survey of explainable AI techniques for detection of fake news and hate speech on social media platforms. *Journal of Computational Social Science*, *7*(1), 587--623. https://doi.org/10.1007/s42001-024-00248-9

Jacovi, A., & Goldberg, Y. (2020). *Towards Faithfully Interpretable NLP Systems: How should we define and evaluate faithfulness?* (Version 3). arXiv. https://doi.org/10.48550/ARXIV.2004.03685

Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P., & Muhherjee, A. (2021). *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*. 14867--14875. https://github.com/punyajoy/HateXplain

Salles, I., Vargas, F., & Benevenuto, F. (2025). HateBRXplain: A Benchmark Dataset with Human-Annotated Rationales for Explainable Hate Speech Detection in Brazilian Portuguese. *Proceedings of the 31st International Conference on Computational Linguistics*, 6659--6669.

Vargas, F. A., Carvalho, I., de Góes, F. R., Benevenuto, F., & Pardo, T. A. S. (2022). *HateBR: A Large Expert Annotated Corpus of Brazilian Instagram Comments for Offensive Language and Hate Speech Detection*. https://doi.org/10.48550/ARXIV.2103.14972

Vargas, F., Carvalho, I., Pardo, T. A. S., & Benevenuto, F. (2025). Context-aware and expert data resources for Brazilian Portuguese hate speech detection. *Natural Language Processing*, *31*(2), 435--456. https://doi.org/10.1017/nlp.2024.18