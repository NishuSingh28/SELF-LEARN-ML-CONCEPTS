---
layout: default
title: "NLP TEXT PRESENTATION"
date: 2025-12-26
categories: [natural-language-processing]
---

## 1. Introduction to Text Representation

Text representation (also called feature extraction from text or text vectorization) is the process of converting text into numerical form so that it can be used by machine learning algorithms.

Machine learning models do not understand:
- words
- sentences
- grammar
- meaning

They only operate on **numbers**.

Therefore, any NLP system must convert text into numbers **before** applying a machine learning model.

This conversion step is one of the **most important stages** in an NLP pipeline, especially in classical machine learning–based NLP.

---

## 2. Why Feature Extraction from Text Is Required

Every machine learning problem follows the same principle:
- Inputs must be numerical features
- Outputs are predicted based on those features

In NLP:
- Raw input → text
- Required input → numbers

The quality of features determines the quality of the model.

This is why the well-known principle applies strongly here:

> **Garbage In, Garbage Out**

A simple algorithm with strong features often outperforms a complex algorithm with weak features.

---

## 3. Why Text Representation Is Difficult

Text representation is significantly harder than representing other data types.

### 3.1 Comparison with Other Modalities

- **Images**  
  Images are grids of pixel values. Each pixel is already a number.

- **Audio**  
  Audio is a waveform represented as amplitude values over time.

- **Text**  
  Text has:
  - symbols instead of numbers
  - meaning instead of magnitude
  - structure instead of fixed dimensions

There is no natural numeric mapping for words.

---

## 4. The Core Idea Behind Text Representation

The fundamental goal of text representation is:

> Convert text into numbers **such that semantic meaning is preserved as much as possible**

A good representation should ensure:
- Similar texts → similar vectors
- Different texts → distant vectors
- Important words → higher influence
- Unimportant words → lower influence

Every technique discussed attempts to satisfy this goal with different trade-offs.

---

## 5. Important Terminology

### Corpus
The complete collection of all documents in the dataset.

### Vocabulary
The set of all **unique words** appearing in the corpus.

### Document
A single unit of text (e.g., one review, sentence, or article).

### Token / Term
An individual word or unit inside a document.

---

## 6. One-Hot Encoding

### 6.1 Concept

One-hot encoding assigns a unique binary vector to each word.

- Vector length = vocabulary size
- Exactly one element is `1`
- All other elements are `0`

### 6.2 Example

Vocabulary:
```

["people", "watch", "campus", "write", "comment"]

```

Vectors:
```

people  → [1, 0, 0, 0, 0]
watch   → [0, 1, 0, 0, 0]
campus  → [0, 0, 1, 0, 0]

```

A document is formed by stacking or combining word vectors.

---

### 6.3 Advantages

- Extremely simple and intuitive
- Easy to implement
- Useful for understanding basic ideas

---

### 6.4 Major Problems with One-Hot Encoding

#### 1. Sparsity
Most values are zero.  
With large vocabularies, vectors become extremely sparse.

#### 2. High Dimensionality
Real datasets can have tens of thousands of unique words.

Example:
- Vocabulary size = 50,000
- Each word → 50,000-dimensional vector

#### 3. No Fixed Input Size
Documents with different numbers of words produce vectors of different shapes, which machine learning models cannot handle.

#### 4. Out-of-Vocabulary (OOV) Problem
If a new word appears during prediction that was not in the training vocabulary, it cannot be represented.

#### 5. No Semantic Meaning
All words are equally distant.

Example:
```

distance(good, great) = distance(good, terrible)

```

This is fundamentally incorrect semantically.

Because of these issues, one-hot encoding is rarely used in production NLP systems.

---

## 7. Bag of Words (BoW)

### 7.1 Core Idea

Instead of representing individual words, Bag of Words represents an **entire document** by counting how many times each vocabulary word appears.

Word order is ignored.

---

### 7.2 Construction Process

1. Build the vocabulary from the corpus
2. For each document, count occurrences of each vocabulary word
3. Represent the document as a fixed-length vector

---

### 7.3 Example

Vocabulary:
```

["people", "watch", "campus", "write", "comment"]

```

Document:
```

"people watch campus watch"

```

Vector:
```

[1, 2, 1, 0, 0]

```

---

### 7.4 Why BoW Is Better Than One-Hot Encoding

- Fixed-size vectors for all documents
- Handles variable document length
- Out-of-vocabulary words are ignored safely
- Works well for text classification

---

### 7.5 Geometric Interpretation

Each document becomes a point in a high-dimensional space.

Similarity between documents is computed using distance or cosine similarity.

Documents with similar word distributions are closer in this space.

---

### 7.6 Limitations of BoW

#### 1. Sparsity
Still produces sparse vectors.

#### 2. No Word Order
```

"This movie is good"
"This movie is not good"

```
BoW treats them as nearly identical.

#### 3. Weak Semantic Understanding
Synonyms are treated as unrelated words.

---

## 8. N-Grams (Bi-grams, Tri-grams)

### 8.1 Motivation

BoW ignores word order, which is often crucial for meaning.

Example:
```

not good ≠ good

```

---

### 8.2 What Are N-Grams?

An n-gram is a sequence of `n` consecutive words.

Examples:
- Unigram (n=1): `movie`
- Bigram (n=2): `not good`
- Trigram (n=3): `movie is good`

---

### 8.3 Bag of N-Grams

Instead of single words, the vocabulary consists of n-grams.

This partially captures word order and local context.

---

### 8.4 Example

Sentence:
```

"this movie is very good"

```

Bigrams:
```

this movie
movie is
is very
very good

```

---

### 8.5 Benefits of N-Grams

- Captures local context
- Improves sentiment detection
- Distinguishes negation
- Better semantic separation than BoW

---

### 8.6 Disadvantages of N-Grams

#### 1. Vocabulary Explosion
Vocabulary size increases rapidly as `n` increases.

#### 2. Increased Computation
More features → slower training and inference.

#### 3. Sparsity Remains
Vectors are still mostly zeros.

#### 4. OOV Problem Still Exists
Unseen n-grams are ignored.

---

## 9. TF-IDF (Term Frequency – Inverse Document Frequency)

### 9.1 Motivation

BoW and N-Grams treat all words equally.

TF-IDF assigns **importance weights**.

---

### 9.2 Term Frequency (TF)

Measures how often a term appears in a document.

```

TF(term, document) =
(number of occurrences of term in document)/(total number of terms in document)

```

---

### 9.3 Inverse Document Frequency (IDF)

Measures how rare a term is across the corpus.

```

IDF(term) = log(
total number of documents/number of documents containing the term)

```

Rare terms → high IDF  
Common terms → low IDF

---

### 9.4 Why Log Is Used

- Prevents extreme values
- Smooths weight differences
- Avoids domination by rare words
- Stabilizes numerical scale

---

### 9.5 TF-IDF Intuition

A word is important if:
- It appears frequently in one document
- But rarely in other documents

This matches human intuition about relevance.

---

### 9.6 Advantages of TF-IDF

- Reduces impact of common words
- Highlights discriminative terms
- Widely used in search engines
- Strong baseline for many NLP tasks

---

### 9.7 Disadvantages of TF-IDF

- Sparse vectors
- High dimensionality
- No synonym understanding
- No deep semantics
- Vocabulary-dependent

---

## 10. sklearn Parameters and Practical Notes

Common parameters in vectorizers:

- `binary=True`  
  Converts counts to presence/absence.  
  Often useful in sentiment analysis.

- `max_features`  
  Limits vocabulary size by keeping most frequent terms.

- `ngram_range=(1,2)`  
  Enables unigrams and bigrams together.

- `stop_words`  
  Removes common stopwords automatically.

---

## 11. Custom (Handcrafted) Features

Standard representations are often combined with domain-specific features.

Examples:
- Count of positive words
- Count of negative words
- Ratio of positive to negative words
- Total word count
- Average word length
- Punctuation count

This hybrid approach often improves performance in classical ML pipelines.

---

## 12. Assignment

### Objective

Implement and compare text representation techniques on a movie review dataset.

### Tasks

1. Apply text preprocessing
2. Compute:
   - Corpus size
   - Vocabulary size
3. Implement:
   - One-hot encoding
   - Bag of Words
   - Bag of N-Grams
   - TF-IDF
4. Compare:
   - Vector dimensions
   - Sparsity
   - Model performance
5. Add custom handcrafted features
6. Analyze advantages and limitations

---

## 13. Key Takeaways

- Text representation is essential for NLP
- Meaning preservation is the core challenge
- One-hot encoding is intuitive but impractical
- Bag of Words improves structure but ignores order
- N-Grams add local context
- TF-IDF adds importance weighting
- All classical methods struggle with semantics
- These methods build the foundation for embeddings



