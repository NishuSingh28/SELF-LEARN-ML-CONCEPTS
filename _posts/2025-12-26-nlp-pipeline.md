---
layout: default
title: "FUNDAMENTALS OF NLP"
date: 2025-12-26
categories: [natural language processing]
---

# NLP Pipeline — Concepts, Design & Practical Guide

This is my consolidated, first-person guide to building production-ready Natural Language Processing systems. It presents a complete pipeline — from raw data to deployed product — and explains the rationale behind choices, practical techniques I use, and small reproducible examples.


## 1. Why a pipeline matters

Building an NLP product is rarely just "train a model." It is an engineering process that transforms noisy, real-world language into signals that models and users can rely on. My pipeline is a sequence of intentional stages that ensure reliability, interpretability, and maintainability:

1. **Data acquisition** — gather the right data.
2. **Text preparation** — clean and normalize text.
3. **Feature engineering** — convert text into features the model can use.
4. **Modeling** — choose and train the algorithm.
5. **Evaluation & monitoring** — verify technical quality and product impact.
6. **Deployment & updates** — ship, watch, and iterate.

These stages are not strictly linear. In production I iterate: monitoring uncovers distribution drift which sends the work back to data collection or feature engineering.


## 2. Data acquisition — start with the right sources

I treat data as the most important asset. I first decide whether internal data exists; if not, I plan how to create or augment it.

Typical sources and treatments:
- **Internal CSV / database exports** — preferred for labeled projects; extract snapshots and schema.
- **Public datasets and academic corpora** — useful for bootstrapping representations.
- **APIs & web scraping** — when product data lives on external sites; I use robust HTML parsing (handle different DOMs) and respect robots.txt and legal constraints.
- **OCR and PDFs** — extract using tuned OCR when reviews or forms are scanned.
- **Speech → text** — use a reliable STT pipeline when reviews come as audio.
- **When data is scarce**:
  - Bootstrapping heuristics to label an initial seed set.
  - Data augmentation: synonym replacement, back-translation, paraphrasing.
  - Active collection: lightweight forms, targeted user prompts, or crowdsourcing.

Key practical checks:
- Confirm the **distribution** of examples covers intended users.
- Inspect **format quality** (HTML, encoding, markup noise).
- Record provenance and sampling strategy so later bias can be diagnosed.


## 3. Text preparation — robust cleaning and normalization

I split preprocessing into three tiers: **basic**, **optional**, and **advanced**. Always prefer the minimal effective preprocessing: too much normalization can erase signal.

### Basic (always)
- Remove HTML tags, control characters and long whitespace.
- Normalize Unicode (NFC/NFKC) and convert emojis to textual tokens where helpful.
- Lowercase text unless case is informative.
- Simple tokenization (sentence and word levels).
- Handle encoding errors and strip invisible characters.

### Optional (use as needed)
- Remove punctuation or digits when they are irrelevant.
- Stopword removal when bag-of-words noise is dominating.
- Stemming or lemmatization when morphological normalization improves downstream features.
- Language detection for multilingual inputs.

### Advanced (for complex tasks)
- Part-of-speech tagging, dependency parsing, and constituency parsing for syntactic features.
- Coreference resolution to map pronouns and entities to their referents.
- Named-entity recognition for entity-aware features.
- Transliteration / translational normalizers for social-media transliterations and mixed scripts.

### Practical transformations I use
- **Emoji normalization**: convert emojis to short text codes so the model sees consistent tokens.
- **Fast-typing & spelling normalization**: map frequent misspellings and abbreviations (e.g., "u" → "you") when dealing with social text.
- **Regex-based cleaning**: remove noisy HTML remnants, repeated punctuation, or boilerplate.


## 4. Feature engineering — from words to numbers

Feature engineering choices follow the modeling approach and data regime.

### When using classical ML (small–medium data)
- Hand-crafted features provide interpretability and a performance boost:
  - Word counts (positive/negative word counts, exclamation count).
  - TF-IDF vectors (word / n-gram based).
  - Jaccard / overlap features between pairs (for semantic similarity tasks).
  - Metadata indicators (same author, same topic tag, source channel).
- Domain-aware engineered features: for product reviews, include star rating; for emails, include sender reputation indicators.

### When using deep learning (large data)
- Minimal preprocessing and let the model learn representations.
- Use transformer embeddings (BERT, RoBERTa) or sentence embeddings as features.
- Fine-tune large pretrained models when labeled data is adequate.

### Hybrid approach
- Combine hand-crafted features with dense embeddings: often gives robust results even with moderate data.


## 5. Modeling — practical decision criteria

I select modeling approaches based on data size, latency needs, interpretability requirements, and budget.

### Common options
- **Rule-based heuristics** — excellent baseline when data is tiny or deterministic rules work.
- **Classical ML** — logistic regression, gradient-boosted trees, SVM, when interpretability or structured features matter.
- **Deep learning** — transformers, siamese networks for semantic similarity, when there is enough labeled data.
- **Cloud NLP APIs** — pragmatic if time-to-market and budget allow; useful for prototyping.

### Transfer learning
I often start with pretrained transformers and fine-tune them for domain tasks. Transfer learning can drastically reduce labeled-data needs while improving quality.

### Training & validation practices
- Maintain a held-out test set and perform cross-validation if dataset is small.
- Use stratified splits for imbalanced tasks.
- Monitor both training curves and validation metrics; guard against overfitting.


## 6. Evaluation — technical and product metrics

I distinguish **intrinsic** (technical) and **extrinsic** (product) evaluation.

### Intrinsic metrics
- Classification: accuracy, precision, recall, F1, confusion matrix.
- Similarity/ranking: ROC-AUC, MAP, nDCG.
- Generation: perplexity, BLEU/ROUGE where applicable.

### Extrinsic / product metrics
- User adoption (e.g., suggestion acceptance rate), reduction in duplicated content, moderation workload reduction.
- Business impact (engagement, retention, conversion).

**I always measure both.** A model that looks great technically but fails in the UI is not a success.


## 7. Deployment, monitoring, and updates

### Deployment patterns
- **Microservice model**: expose model through REST/gRPC endpoints for real-time scoring.
- **Batch jobs**: for offline scoring or nightly reprocessing.
- **Edge / mobile**: when latency and offline availability require bundled models.

### Monitoring
- Track model metrics (e.g., precision/recall) over time and compare them to historical baselines.
- Instrument production logs to compute **product-level KPIs** like suggestion clicks or moderation actions accepted.
- Visualize trends and alert on drift (sudden metric shifts).

### Updating strategy
- Retrain when performance degrades or dataset distribution changes.
- Use a canary / A/B test rollout before full replacement.
- Maintain versioning and rollback capability.

## 8. A working example: duplicate question detector (design + examples)

I design this pipeline with the objective: given two short texts (questions), decide whether they are semantically the same.

### Pipeline (concise)
1. Data: collect labeled pairs (public datasets, internal logs, or construct via heuristics). Augment with paraphrases.
2. Preprocess: basic cleaning, tokenization, emoji demojize, lowercasing.
3. Features:
   - TF-IDF vectors and cosine/Jaccard overlap.
   - Sentence embeddings (BERT) + cosine similarity.
   - Hand-crafted features: shared tokens count, common named entities.
4. Model: baseline logistic regression on feature vector + BERT siamese model for production quality.
5. Evaluate: F1 for balanced performance; monitor acceptance rate in product UI.
6. Deploy: real-time microservice, precompute embeddings for high-traffic questions to speed similarity checks.

### Minimal, reproducible code snippets

I show a compact TF-IDF baseline and a small example of normalizing HTML and emojis. Code blocks below are Python examples and are enclosed in triple single quotes as requested.

'''python
# html_and_emoji_cleaner.py
import re
import emoji

def remove_html_tags(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def demojize_text(text: str) -> str:
    # convert emojis to textual tokens, e.g., "I ❤️ NLP" -> "I :red_heart: NLP"
    return emoji.demojize(text)

def normalize_text(text: str) -> str:
    text = remove_html_tags(text)
    text = demojize_text(text)
    text = text.lower()
    return text

# Example
raw = "<p>I love NLP! 😄 Visit <a href='x'>here</a></p>"
print(normalize_text(raw))
# -> "i love nlp ! :smile: visit here"
'''

'''python
# tfidf_similarity_baseline.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    "How can I learn natural language processing?",
    "What are good ways to start learning NLP?"
]

vect = TfidfVectorizer(ngram_range=(1,2)).fit_transform(corpus)
sim = cosine_similarity(vect[0:1], vect[1:2])[0][0]
print(f"TF-IDF cosine similarity: {sim:.4f}")
'''

## 9. Decision heuristics & rules of thumb

- If labeled data is **< a few thousand**: start with rules and classical ML; add transfer learning only after a validation plan exists.
- If labeled data is **large** (> 50k good-quality examples): fine-tune a transformer model.
- If product needs **real-time low-latency**: precompute embeddings, use approximate nearest neighbor indices (e.g., FAISS).
- For **explainability**, prefer classical features or add explainability layers (LIME/SHAP) to dense models.



