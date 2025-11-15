---
layout: default
title: "ENCODER AND DECODER"
date: 2025-10-17
categories: [deep-learning]
---

# Encoder-Decoder Architecture: Complete Mathematical Summary

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Architecture Overview](#architecture-overview)
3. [Mathematical Components](#mathematical-components)
4. [Training Process](#training-process)
5. [Improvements](#improvements)
6. [Limitations & Future](#limitations)

---

## Problem Statement

### Sequence-to-Sequence (Seq2Seq) Challenge

**Input**: Variable-length sequence (e.g., English sentence)
**Output**: Variable-length sequence (e.g., Hindi translation)

**Example**: 
- Input: "Nice to meet you" (4 words)
- Output: "आपसे मिलकर अच्छा लगा" (6 words)

### Why Challenging?

1. **Variable Input Length**: Cannot use fixed-size weight matrices
2. **Variable Output Length**: Model must decide when to stop
3. **Length Mismatch**: Input length ≠ Output length
4. **Temporal Dependencies**: Order matters crucially

**Mathematical Problem**: 
```
Traditional NN: y = f(x), x ∈ ℝⁿ (fixed n)
Seq2Seq: Y = f(X), X ∈ ℝ^(T_x × d), Y ∈ ℝ^(T_y × d'), T_x ≠ T_y
```

---

## Architecture Overview

### High-Level Structure

```
[Input Sequence] → [ENCODER] → [Context Vector] → [DECODER] → [Output Sequence]
```

### Components

1. **Encoder**: LSTM that processes input word-by-word
   - Compresses entire input into fixed-size context vector
   - Final state: (h_T, c_T) = context vector

2. **Context Vector**: Information bottleneck
   - h_T: Filtered working memory (tanh-squashed)
   - c_T: Complete long-term memory (unbounded)
   - Dimension: typically 512-1024

3. **Decoder**: LSTM that generates output word-by-word
   - Initialized with encoder's final state
   - Generates until <END> token
   - Uses softmax over target vocabulary

---

## Mathematical Components

### 1. LSTM Equations (Both Encoder & Decoder)

**Forget Gate** (what to remove from memory):
```
f_t = σ(W_f·x_t + U_f·h_{t-1} + b_f)
```

**Input Gate** (what to add to memory):
```
i_t = σ(W_i·x_t + U_i·h_{t-1} + b_i)
c̃_t = tanh(W_c·x_t + U_c·h_{t-1} + b_c)
```

**Cell State Update** (memory update):
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
```

**Output Gate** (what to reveal):
```
o_t = σ(W_o·x_t + U_o·h_{t-1} + b_o)
h_t = o_t ⊙ tanh(c_t)
```

**Key**: All gates use **same weights** across all time steps (weight sharing)

### 2. Why Three Separate Gates?

| Gate | Activation | Range | Purpose |
|------|------------|-------|---------|
| Forget | sigmoid | (0,1) | Scale down old memory |
| Input | sigmoid | (0,1) | Scale new candidates |
| Candidate | tanh | (-1,1) | Both strengthen & weaken |
| Output | sigmoid | (0,1) | Filter cell state |

**Mathematical Elegance**: 
- f_t ≈ 1 → gradient flows unimpeded (solves vanishing gradients!)
- Three degrees of freedom → more expressive

### 3. Context Vector Deep Dive

**What it stores**:
```
Context = [h_T; c_T] ∈ ℝ^(2d)

h_T: Recent information, filtered by output gate
c_T: Complete historical information, unfiltered
```

**Information Capacity**:
- If d=512, using float32: 512 × 32 × 2 = 32,768 bits ≈ 4KB
- Sufficient for typical sentences due to language redundancy
- **Bottleneck** for very long sequences (→ Attention mechanism)

**Distributed Representation**:
```
c ≈ α₁v_subject + α₂v_action + α₃v_tense + α₄v_sentiment + ...
```
Each dimension doesn't represent one concept; patterns across ALL dimensions encode meaning

---

## Training Process

### 1. Forward Pass

**Encoder** (t = 1 to T_x):
```
For each word x_t:
    h_t^enc, c_t^enc = LSTM_enc(e(x_t), h_{t-1}^enc, c_{t-1}^enc)

Context = (h_{T_x}^enc, c_{T_x}^enc)
```

**Decoder** (t = 1 to T_y):
```
Initialize: h_0^dec = h_{T_x}^enc, c_0^dec = c_{T_x}^enc

For each time step t:
    h_t^dec, c_t^dec = LSTM_dec(e(y_{t-1}), h_{t-1}^dec, c_{t-1}^dec)
    z_t = W_out · h_t^dec + b_out
    p_t = softmax(z_t)  ∈ ℝ^|V_target|
```

### 2. Loss Calculation

**Cross-Entropy Loss**:
```
L_t = -log p_t^(y_t*)

Total Loss: L = Σ_{t=1}^{T_y} L_t = -Σ_{t=1}^{T_y} log p_t^(y_t*)
```

**Why negative log?**
- p=1 (perfect) → L=0
- p=0.5 → L=0.69
- p=0.1 → L=2.30
- p→0 → L→∞ (heavy penalty)

### 3. Backward Pass (BPTT)

**Gradient Flow**:
```
Loss → Output Layer → h_T^dec → ... → h_1^dec → Context → h_T^enc → ... → h_1^enc
```

**Key Gradients**:
```
∂L/∂z_t = p_t - y_t*  (softmax gradient)

∂L/∂W_out = Σ_t (∂L/∂z_t)(h_t^dec)^T

∂L/∂c_t = ∂L/∂h_t ⊙ o_t ⊙ tanh'(c_t) + ∂L/∂c_{t+1} ⊙ f_{t+1}
```

**Why LSTM avoids vanishing gradients**:
```
∂c_t/∂c_{t-1} = f_t  (element-wise!)

If f_t ≈ 1, gradient flows perfectly through time
No repeated matrix multiplication like RNN
```

### 4. Parameter Update

**Adam Optimizer** (preferred over SGD):
```
m_t = β₁m_{t-1} + (1-β₁)∇L
v_t = β₂v_{t-1} + (1-β₂)(∇L)²
θ_new = θ_old - η · m̂_t / (√v̂_t + ε)
```

**Typical hyperparameters**:
- Learning rate η: 0.001
- β₁: 0.9, β₂: 0.999
- Batch size: 32-128

### 5. Teacher Forcing

**Training** (with teacher forcing):
```
y_{t-1}^input = y_{t-1}*  (TRUE word from data)
```

**Testing** (free running):
```
y_{t-1}^input = argmax(p_{t-1})  (model's prediction)
```

**Why Teacher Forcing?**
- ✅ 3-10× faster training
- ✅ Stable gradients
- ✅ Guaranteed convergence
- ❌ Exposure bias (train-test mismatch)

**Solution: Scheduled Sampling**
```
p(epoch) = max(0, 1 - k·epoch)

y_{t-1}^input = {
    y_{t-1}*     with probability p (teacher)
    argmax(p_{t-1})  with probability 1-p (free)
}
```

Start with p=1 (pure teacher forcing), gradually decrease to p=0

---

## Improvements

### 1. Embeddings Instead of One-Hot

**Problem**: One-hot encoding for 100K vocab → 100K dimensions

**Solution**: Word embeddings
```
e: V → ℝ^d  (e.g., d=300)

x_t → e(x_t) → LSTM
```

**Benefits**:
- Dense representation (no zeros)
- Captures semantic similarity
- Dimensionality reduction (100K → 300)

**Options**:
- Pre-trained: Word2Vec, GloVe
- Learned: Train with model

### 2. Deep (Stacked) LSTMs

**Architecture**:
```
Layer 3: LSTM₃ → LSTM₃ → LSTM₃ → ...
           ↑        ↑        ↑
Layer 2: LSTM₂ → LSTM₂ → LSTM₂ → ...
           ↑        ↑        ↑
Layer 1: LSTM₁ → LSTM₁ → LSTM₁ → ...
           ↑        ↑        ↑
Input:    x₁       x₂       x₃
```

**Why 3-4 layers?**
- **Better long-term dependencies**: More capacity to store context
- **Hierarchical features**: Lower layers = words, middle = phrases, upper = sentences
- **Better performance**: Empirically 2-4 BLEU points improvement

**Tradeoff**: More parameters (slower training, more GPU memory)

### 3. Bidirectional Encoder

**Standard LSTM**: Only sees past context
```
h_t = LSTM(x_t, h_{t-1})  ← only looks left
```

**Bidirectional**:
```
h_t^→ = LSTM^→(x_t, h_{t-1}^→)  ← forward
h_t^← = LSTM^←(x_t, h_{t+1}^←)  ← backward
h_t = [h_t^→; h_t^←]  ← concatenate
```

**Why?** Encoder can look at ENTIRE sentence (both directions)

### 4. Input Sequence Reversal

**Trick**: Reverse the source sequence

Normal: "think about it" → encoder
Reversed: "it about think" → encoder

**Why it helps** (for some language pairs):
- First source word closer to first target word
- Better gradient flow for early words
- Helped 2-3 BLEU on English→French

**Caveat**: Doesn't always help (language-dependent)

---

## Limitations & Future

### Fundamental Limitations

1. **Information Bottleneck**
   ```
   lim_{T→∞} I(Input_T; Context)/T → 0
   ```
   As sentences get longer, context vector retains smaller percentage

2. **Fixed Context Vector**
   - All input information compressed to fixed size
   - Long documents: information loss

3. **Sequential Processing**
   - Cannot parallelize across time steps
   - Slow for very long sequences

### Solutions (Next Evolution)

1. **Attention Mechanism** (Bahdanau et al. 2015)
   - Decoder can "look back" at all encoder states
   - Different context vector at each decoder step
   - Solves information bottleneck

2. **Transformer Architecture** (Vaswani et al. 2017)
   - Removes recurrence entirely
   - Self-attention + multi-head attention
   - Fully parallelizable
   - State-of-the-art for all seq2seq tasks

### Original Paper Results (Sutskever et al. 2014)

**Dataset**: WMT English-French (12M sentence pairs)
**Architecture**:
- 4-layer LSTMs (both encoder & decoder)
- 1000 hidden units per layer
- 160K source vocab, 80K target vocab
- Input reversal used

**Results**:
- BLEU: 34.8 (beat phrase-based baseline)
- Training: 7 days on 8 GPUs

---

## Quick Reference Formulas

### Complete LSTM Cell
```
f_t = σ(W_f·x_t + U_f·h_{t-1} + b_f)
i_t = σ(W_i·x_t + U_i·h_{t-1} + b_i)
c̃_t = tanh(W_c·x_t + U_c·h_{t-1} + b_c)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
o_t = σ(W_o·x_t + U_o·h_{t-1} + b_o)
h_t = o_t ⊙ tanh(c_t)
```

### Encoder-Decoder Connection
```
h_0^dec = h_{T_x}^enc
c_0^dec = c_{T_x}^enc
```

### Output & Loss
```
p_t = softmax(W_out·h_t^dec + b_out)
L = -Σ_t log p_t^(y_t*)
```

### Teacher Forcing
```
Training: h_t^dec = LSTM(e(y_{t-1}*), h_{t-1}^dec)
Testing:  h_t^dec = LSTM(e(argmax p_{t-1}), h_{t-1}^dec)
```

---

## Summary

**Encoder-Decoder solves seq2seq problems by**:
1. **Encoding** variable-length input into fixed context
2. **Decoding** from context to variable-length output
3. **Using LSTMs** to handle long-term dependencies
4. **Training end-to-end** with backpropagation through time
5. **Teacher forcing** for fast, stable training

**Key Innovation**: Separates input/output processing, connected only through learned context vector

**Legacy**: Foundation for modern NLP (BERT, GPT, etc. all build on these ideas) ----
