---
layout: default
title: "NLP TEXT PROCESSING"
date: 2025-12-26
categories: [natural-language-processing]
---

## 1. Introduction

Text preprocessing is the process of transforming raw textual data into a clean, consistent, and structured form that machine learning and deep learning models can reliably consume.

Real-world text data is noisy. It often contains HTML artifacts, URLs, emojis, spelling inconsistencies, informal abbreviations, and formatting irregularities. Without preprocessing, these issues degrade model performance and introduce instability in production systems.

A well-designed text preprocessing pipeline:
- Reduces noise
- Improves generalization
- Ensures consistency between training and inference
- Prevents subtle production bugs

---

## 2. Core Design Principles

Text preprocessing must follow these principles:

- **Minimal transformation**  
  Only transformations that provide measurable benefit should be applied.

- **Task dependency**  
  No preprocessing step is universally correct. Choices depend on the NLP task.

- **Consistency**  
  Training and inference pipelines must apply identical preprocessing logic.

- **Measurability**  
  Each preprocessing step should be validated using downstream metrics.

Preprocessing is therefore **context-driven**, not rule-based.

---

## 3. Levels of Preprocessing

### 3.1 Basic Preprocessing (Almost Always Applied)

- Unicode normalization
- Whitespace normalization
- HTML / markup removal
- URL removal
- Lowercasing (unless case is semantically important)

### 3.2 Task-Dependent Preprocessing

- Punctuation removal
- Stopword removal
- Chat shorthand expansion
- Emoji handling
- Spelling correction

### 3.3 Advanced Preprocessing

- Tokenization strategies
- Stemming
- Lemmatization
- POS-aware or limited lemmatization

---

## 4. Typical End-to-End Preprocessing Pipeline

1. Normalize whitespace and Unicode artifacts  
2. Remove HTML tags  
3. Remove URLs  
4. Handle emojis (remove or convert to text)  
5. Convert text to lowercase  
6. Expand chat abbreviations  
7. Remove punctuation (if appropriate)  
8. Optional spelling correction  
9. Optional stopword removal  
10. Tokenization  
11. Optional stemming or lemmatization  

---

## 5. Reusable Python Utility Functions

```python
# text_cleaners.py
import re
import string
from typing import Dict
import emoji

_HTML_RE = re.compile(r'<[^>]+>')
_URL_RE = re.compile(r'(https?:\/\/\S+|www\.\S+|\bhttps?:\S+\b)')
_PUNCT_TRANSLATION = str.maketrans('', '', string.punctuation)

def remove_html_tags(text: str) -> str:
    if not text:
        return ''
    text = _HTML_RE.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def remove_urls(text: str) -> str:
    if not text:
        return ''
    return _URL_RE.sub(' ', text)

def normalize_basic(text: str, lower: bool = True) -> str:
    if not text:
        return ''
    text = text.replace('\u00A0', ' ').strip()
    return text.lower() if lower else text

def remove_punctuation(text: str) -> str:
    if not text:
        return ''
    return text.translate(_PUNCT_TRANSLATION)

def expand_shorthand(text: str, lexicon: Dict[str, str]) -> str:
    if not text:
        return ''
    tokens = text.split()
    return ' '.join(lexicon.get(tok, tok) for tok in tokens)

def demojize_text(text: str) -> str:
    if not text:
        return ''
    return emoji.demojize(text)
````

---

## 6. A Practical Preprocessing Pipeline

```python
# simple_pipeline.py
import re
import string
import emoji

SHORTHAND = {
    "u": "you",
    "ur": "your",
    "gn": "good night",
    "imo": "in my opinion",
    "idk": "i do not know",
}

_HTML_RE = re.compile(r'<[^>]+>')
_URL_RE = re.compile(r'(https?:\/\/\S+|www\.\S+|\bhttps?:\S+\b)')
_PUNCT_TRANSLATION = str.maketrans('', '', string.punctuation)

def preprocess_text(text: str) -> str:
    if not text:
        return ''

    text = text.replace('\u00A0', ' ').strip()
    text = _HTML_RE.sub(' ', text)
    text = _URL_RE.sub(' ', text)
    text = emoji.demojize(text)
    text = text.lower()

    tokens = text.split()
    tokens = [SHORTHAND.get(tok, tok) for tok in tokens]

    text = ' '.join(tokens).translate(_PUNCT_TRANSLATION)
    return re.sub(r'\s+', ' ', text).strip()
```

---

## 7. Explanation of Key Techniques

### Lowercasing

Lowercasing reduces vocabulary size and avoids redundant token representations.
It should be avoided when case conveys meaning, such as in Named Entity Recognition.

### HTML and URL Removal

HTML tags and URLs are presentation artifacts.
They are removed unless URL structure provides useful metadata.

### Punctuation Removal

Punctuation often increases vocabulary size without improving meaning.
However, it may encode sentiment or intent and should be retained when useful.

### Chat Shorthand Expansion

Chat and social-media text frequently contains abbreviations.
Expanding them improves semantic consistency.

### Emoji Handling

Emojis encode emotion and sentiment.
Converting emojis into textual tokens preserves meaning in a model-friendly way.

### Spelling Correction

Spelling correction reduces noise but can alter meaning.
It should be applied conservatively and evaluated carefully.

### Stopword Removal

Stopwords reduce noise in bag-of-words models.
They are often unnecessary for transformer-based models.

### Tokenization

* Classical ML models use word or n-gram tokenization
* Transformer models use model-specific subword tokenizers
* Noisy text benefits from robust NLP libraries such as spaCy

### Stemming vs. Lemmatization

* **Stemming**: fast but aggressive; may produce non-words
* **Lemmatization**: linguistically correct but slower
* **Limited lemmatization**: balances readability and normalization

---

## 8. Performance and Production Considerations

* Compile regular expressions once
* Prefer vectorized operations over Python loops
* Precompute cleaned text or embeddings for low-latency inference
* Version preprocessing code and reuse it across training and deployment

---

## 9. Testing Checklist

* [ ] Training and inference use identical preprocessing
* [ ] Manual inspection of before/after samples completed
* [ ] Validation metrics confirmed or improved
* [ ] Edge cases covered (emails, numbers, currencies, mixed language)

---

## 10. Example Notebook Snippet

```python
import pandas as pd
from simple_pipeline import preprocess_text

df = pd.read_csv("movies_raw_sample.csv")

df["clean_description"] = df["description"].fillna("").apply(preprocess_text)
df["token_count"] = df["clean_description"].str.split().apply(len)

print("Average tokens:", df["token_count"].mean())
print("Vocabulary size:", len(set(" ".join(df["clean_description"]).split())))

df.to_csv("processed_movies.csv", index=False)
```

---

## 11. Final Notes

Text preprocessing is an iterative engineering process.
The optimal pipeline depends on data quality, task requirements, and system constraints.
When designed carefully, preprocessing forms a stable foundation for reliable NLP systems.

