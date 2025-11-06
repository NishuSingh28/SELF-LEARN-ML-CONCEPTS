---
layout: default
title: "TRANSFORMERS"
date: 2025-10-17
categories: [deep-learning]
---

# Understanding the Transformer Architecture

## 1. Overview

A **Transformer** is a Neural Network architecture designed for **Sequence-to-Sequence (Seq2Seq)** tasks — where both input and output are sequential data.

**Examples:** Machine Translation, Question Answering, Text Summarization.

| Task Type | Architecture |
|------------|---------------|
| Tabular Data | ANN |
| Image Data | CNN |
| Sequential Data | RNN/LSTM |
| Seq2Seq (Modern) | Transformer |

**Core Structure:** Encoder–Decoder  
**Key Innovation:** Replaces RNN/LSTM with **Self-Attention**, enabling **parallel processing** of entire sequences — faster and more scalable.

## 2. Historical Evolution

### (a) Seq2Seq with LSTM — 2014

**Paper:** “Sequence to Sequence Learning with Neural Networks”  
- Introduced the Encoder–Decoder model using LSTMs.  
- **Bottleneck:** Entire input compressed into one fixed-size *context vector*.  
- For long inputs, this caused **information loss**.

Mathematically:

$$
c = h_T
$$

where \( c \) is the final encoder hidden state used as context.

### (b) Attention Mechanism — 2014

**Paper:** “Neural Machine Translation by Jointly Learning to Align and Translate”  
- Added **Attention**: each decoding step gets a **unique context vector**.  
- This context is a weighted sum of all encoder hidden states:

$$
c_t = \sum_i \alpha_{t,i} h_i
$$

where the attention weights are computed as:

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}
$$

This allowed the model to “focus” on different words dynamically.  
However, it still processed data sequentially, making training slow.

### (c) Transformers — 2017

**Paper:** “Attention Is All You Need”  
- Removed RNNs entirely.  
- Built on **Self-Attention**, **Feed-Forward Networks**, **Residual Connections**, and **Layer Normalization**.  
- Enables **parallelization** and **massive scalability**.  

**Result:** Foundation for **Transfer Learning** in NLP — leading to models like **BERT** and **GPT**.

## 3. Why Transformers Changed Everything

### (a) Revolution in NLP
- Solved long-standing NLP problems rapidly.  
- Enabled chatbots, search engines, summarizers, and assistants.

### (b) Democratization of AI
- Pre-trained models (BERT, GPT) available via **Hugging Face**.  
- Fine-tuning yields state-of-the-art results with little data.

### (c) Multimodal Capability
Transformers can handle multiple data types:
- Text: GPT, BERT  
- Images: Vision Transformers  
- Audio: Whisper  
- Cross-modal: DALL-E, CLIP, Runway ML  

### (d) Generative AI Era
- Powers text, code, image, and video generation.  
- Core of today’s **Generative AI** systems.

### (e) Unified Deep Learning
Now used across NLP, Computer Vision, Reinforcement Learning, and even scientific discovery (e.g., **AlphaFold**).

## 4. Advantages

- **Scalable:** Parallel training on large datasets.  
- **Transferable:** Pre-train once, fine-tune many tasks.  
- **Flexible:** Works with text, images, audio, and more.  
- **Adaptable:** Encoder-only (BERT), Decoder-only (GPT), or Encoder–Decoder (T5).  
- **Ecosystem:** Hugging Face, PyTorch, TensorFlow support.

## 5. Key Applications

| Domain | Example |
|--------|----------|
| Conversational AI | ChatGPT (GPT-based) |
| Text-to-Image | DALL-E, Midjourney |
| Protein Folding | AlphaFold |
| Code Generation | OpenAI Codex / GitHub Copilot |

## 6. Challenges & Limitations

- **Compute-Intensive:** Requires large GPU clusters.  
- **Data-Hungry:** Needs billions of tokens.  
- **Energy Cost:** High carbon footprint.  
- **Black-Box Behavior:** Hard to interpret internal reasoning.  
- **Bias & Ethics:** Models reflect biases in web data.

## 7. The Future of Transformers

**Efficiency Improvements:**  
Techniques like pruning, quantization, and knowledge distillation reduce model size.

**Enhanced Multimodality:**  
Combining text, images, audio, and time-series data more effectively.

**Responsible AI:**  
Improving fairness, transparency, and data ethics.

**Domain-Specific Models:**  
Specialized models such as LegalGPT, DoctorGPT, EduGPT.

**Multilingual Expansion:**  
Better support for regional and low-resource languages.

**Interpretability Research:**  
From black boxes to transparent “white-box” models through attention visualization.

## 8. Summary

| Aspect | RNN | Transformer |
|--------|------|--------------|
| Sequential Processing | Yes | No |
| Parallelization | No | Yes |
| Attention | Partial | Full (Self-Attention) |
| Scalability | Limited | Massive |
| Training Speed | Slow | Fast |
| Transfer Learning | Rare | Common |

**In Essence:**  
Transformers replaced recurrence with attention — unlocking parallelization, scalability, and transfer learning, the cornerstones of modern AI.

---

# Self-Attention — Part 1: What and Why

## 1. Introduction and Goal of the Series
This is the first video in a three-part series on Self-Attention:

- **Part 1:** What is Self-Attention?  
- **Part 2:** How does Self-Attention work?  
- **Part 3:** A deeper, geometric look behind the scenes  

The goal is to build a strong understanding of Self-Attention as the foundation for learning **Transformers** and modern **Generative AI** models.

## 2. The Core Problem in NLP — Representing Words as Numbers
Any NLP task (sentiment analysis, translation, etc.) requires converting text into numerical form because computers process numbers, not words.  
This conversion process is called **Vectorization**.

Let the vocabulary be  

$$
V = \{\text{Mat}, \text{Cat}, \text{Rat}\}
$$

Each word must be represented as a vector  

$$
\vec{w}_i \in \mathbb{R}^n
$$

## 3. Evolution of Word Vectorization Techniques

### (a) One-Hot Encoding
Each word is represented as a binary vector:

$$
\text{Cat} = [0, 1, 0], \quad \text{Mat} = [1, 0, 0]
$$

- Unique but inefficient and does not capture relationships between words.

### (b) Bag of Words (BoW)
Represents a sentence by counting the frequency of each word from the vocabulary.  
However, it ignores word order and context.

### (c) TF–IDF
Improves BoW by weighting rare but informative words higher.

### (d) Word Embeddings
A major leap forward — dense, real-valued vectors learned using neural networks trained on large text corpora.

Each word \( w_i \) is mapped to  

$$
\vec{w}_i \in \mathbb{R}^d
$$

where \( d \) can be 256, 512, etc.

- **Semantic Property:** Similar words have similar vectors  

$$
\text{sim}(\vec{w}_{\text{king}}, \vec{w}_{\text{queen}}) > \text{sim}(\vec{w}_{\text{king}}, \vec{w}_{\text{cricketer}})
$$

- Each dimension roughly encodes some hidden semantic feature (e.g., royalty, gender, profession).

## 4. The Limitation — “Average Meaning”

Word embeddings are **static** — a single fixed vector represents each word regardless of context.

### Example: The word “Apple”
If during training:
- 90% of uses mean “fruit”
- 10% mean “company”

Then  

$$
\vec{w}_{\text{apple}} = 0.9 \cdot \vec{v}_{\text{fruit}} + 0.1 \cdot \vec{v}_{\text{company}}
$$

This results in an **average meaning** vector.  
The same vector will be used for both:
- “I ate an apple.”  
- “Apple launched a new phone.”

## 5. The Context Problem

Consider the sentence:  

**“The apple launched a new phone while I was eating an orange.”**

The correct interpretation of “apple” here is the **company**, not the fruit.  
However, a static embedding cannot adapt to the context.  
We need embeddings that can **change dynamically** based on surrounding words.

The ideal embedding function should be:

$$
\vec{w}_i' = f(\vec{w}_1, \vec{w}_2, \dots, \vec{w}_n)
$$

so that each word’s new embedding \(\vec{w}_i'\) depends on **all** words in the sentence.

## 6. The Solution — Self-Attention

Self-Attention is the mechanism that produces **contextual embeddings**.

It takes the static embeddings as input:

$$
[\vec{w}_1, \vec{w}_2, \dots, \vec{w}_n]
$$

and outputs:

$$
[\vec{w}_1', \vec{w}_2', \dots, \vec{w}_n']
$$

where each \(\vec{w}_i'\) captures the meaning of the word **in context**.

Formally:

$$
\vec{w}_i' = \text{SelfAttention}(\vec{w}_1, \vec{w}_2, \dots, \vec{w}_n)
$$

These contextual embeddings are then used by Transformer models and modern NLP architectures.

## 7. Conclusion and Preview

**Summary:**  
Self-Attention transforms static word embeddings into contextualized embeddings that understand the relationships between words within a sentence.

**Next Part:**  
We will explore *how* Self-Attention works internally using **Query (Q)**, **Key (K)**, and **Value (V)** vectors.

---

# The Complete Guide to Self-Attention

This guide synthesizes the entire lecture, from the initial problem to the final, refined mechanism of Self-Attention.

## Part 1: The "Why" — The Problem with Static Embeddings

### 1. The Fundamental NLP Task
Represent words as numbers so computers can understand them. For a vocabulary 

$$
V = \{ w_1, w_2, \dots, w_n \},
$$ 

each word is mapped to a vector

$$
\vec{w}_i \in \mathbb{R}^d.
$$

### 2. The Evolution of Word Representations
- **One-Hot Encoding:** Inefficient, no semantic meaning.  
- **Bag-of-Words (BoW):** Slightly better, captures frequency but ignores order.  
- **Word Embeddings (Word2Vec, GloVe):** Dense vectors that capture semantic meaning. Similar words have similar vectors:

$$
\text{sim}(\vec{w}_{\text{king}}, \vec{w}_{\text{queen}}) > \text{sim}(\vec{w}_{\text{king}}, \vec{w}_{\text{cricketer}}).
$$

### 3. The Limitation — Static Embeddings
The same vector is used for a word in every context:
- “I deposited money in the **bank**.”  
- “I sat by the river **bank**.”  

The embedding represents an **average meaning** across all usages. For example, for the word *apple*:

$$
\vec{w}_{\text{apple}} \approx 0.9\,\vec{v}_{\text{fruit}} + 0.1\,\vec{v}_{\text{company}}
$$

### 4. The Goal — Contextualized Embeddings
We want embeddings that **dynamically change** depending on the sentence:

$$
\vec{w}_i' = f(\vec{w}_1, \vec{w}_2, \dots, \vec{w}_n)
$$

For example, "bank" in a financial context should differ from "bank" in a geographical context.

## Part 2: The "What" — First Principles of Self-Attention

### Core Idea
Each word is represented as a mixture of all words in the sentence. For the sentence "Money bank grows":

$$
\begin{aligned}
\text{New("Money")} &= f_1(\text{money}) + f_2(\text{bank}) + f_3(\text{grows}) \\
\text{New("Bank")} &= f_4(\text{money}) + f_5(\text{bank}) + f_6(\text{grows}) \\
\text{New("Grows")} &= f_7(\text{money}) + f_8(\text{bank}) + f_9(\text{grows})
\end{aligned}
$$

### Step 1: Calculate Similarity
Compute dot product between word embeddings:

$$
s_{ij} = e_i \cdot e_j
$$

Higher values indicate more similarity in context.

### Step 2: Normalize Scores with Softmax
Convert similarity scores into probabilities:

$$
a_{ij} = \frac{e^{s_{ij}}}{\sum_{k=1}^n e^{s_{ik}}}, \quad \sum_j a_{ij} = 1
$$

### Step 3: Compute Weighted Sum
The contextual embedding for each word is:

$$
h_i = \sum_{j=1}^n a_{ij}\, e_j
$$

### Pros & Cons
- **Pros:** Works; highly parallelizable.  
- **Cons:** No learnable parameters; embeddings are generic, not task-specific.

## Part 3: The Refinement — Query, Key, and Value

### Motivation
Each word embedding should have three roles:
1. **Query (Q):** "How relevant are other words to me?"  
2. **Key (K):** "How relevant am I to other words?"  
3. **Value (V):** The actual content to contribute to output.

### Linear Transformations
Project embeddings into Q, K, V vectors using learnable matrices:

$$
\begin{aligned}
Q &= X W_Q \\
K &= X W_K \\
V &= X W_V
\end{aligned}
$$

where \( X \in \mathbb{R}^{n \times d} \) is the input embedding matrix, and \( W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k} \) are learnable parameters.

## Part 4: The Complete Mechanism — Scaled Dot-Product Attention

### Self-Attention Formula
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

### Step-by-Step Walkthrough
1. **Compute attention scores:**

$$
S = Q K^\top, \quad S_{ij} = q_i \cdot k_j
$$

2. **Scale the scores:**

$$
\hat{S} = \frac{S}{\sqrt{d_k}}
$$

3. **Apply softmax row-wise:**

$$
A = \text{softmax}(\hat{S}), \quad \sum_j A_{ij} = 1
$$

4. **Weighted sum of values:**

$$
\text{Output} = A V, \quad h_i = \sum_j A_{ij} v_j
$$

### Multi-Head Attention (Optional)
Split projections into \(H\) heads:

$$
\text{head}_h = \text{Attention}(Q^{(h)}, K^{(h)}, V^{(h)}), \quad \text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
$$

## Final Summary

| Aspect | Simple Model | Transformer Model |
|:---|:---|:---|
| Vectors Used | Original embeddings only | Q, K, V projections |
| Learnable Parameters | None | \(W_Q, W_K, W_V\) |
| Output | General contextual embeddings | Task-specific embeddings |
| Core Formula | \(h_i = \sum_j a_{ij} e_j\) | \(\text{softmax}\big(\frac{QK^\top}{\sqrt{d_k}}\big) V\) |
| Parallelization | Yes | Highly parallelizable |

Self-Attention converts **static embeddings** into **dynamic, context-aware embeddings**, forming the foundation of Transformers, LLMs, and modern Generative AI.
