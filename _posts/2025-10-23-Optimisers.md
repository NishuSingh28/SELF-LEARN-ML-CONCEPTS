---
layout: default
title: "OPTIMIZERS"
date: 2025-10-17
categories: [deep-learning]
---

# Optimizers in Deep Learning: A Comprehensive Introduction

## Overview

This guide explores **optimizers in deep learning**, focusing on why they are essential for speeding up and improving neural network training performance.

---

## Key Concepts

### 1. Introduction & Goal

The primary objective is to **enhance neural network performance**, with specific emphasis on accelerating the training process. Several techniques have been developed to achieve this:

- Weight Initialization  
- Batch Normalization  
- Choice of Activation Function
- Optimisers 

**Current Focus:** Lets discuss the most crucial technique: **Optimizers**

---

### 2. Role of an Optimizer

**Fundamental Training Goal:**

Find optimal values for weights and biases that minimize the **loss function** (the difference between predicted and actual values).  

This process resembles navigating a complex, multi-dimensional landscape (the loss function graph) to locate the **global minimum** (the point of lowest loss).

**Optimizer Function:**

- Guides the search for optimal weights and biases  
- Determines how to update weights efficiently to reach the minimum loss

---

### 3. The Baseline: Gradient Descent

**Core Algorithm:**  

The most fundamental optimizer is **Gradient Descent**.  

**Basic update rule:**  
New Weight = Old Weight - Learning Rate * Gradient

markdown
Copy code

**Learning Rate Criticality:**

- Too Small → Training becomes excessively slow  
- Too Large → Training instability and overshooting minima

**Gradient Descent Variants:**

| Type                     | Description                            | Characteristics             |
|---------------------------|----------------------------------------|----------------------------|
| Batch Gradient Descent     | Uses entire dataset for one update      | Slow but stable            |
| Stochastic Gradient Descent (SGD) | Updates weights after every data point | Many updates, potentially noisy |
| Mini-Batch Gradient Descent | Updates weights after small batches   | Most common, balanced approach |

---

### 4. Challenges with Basic Gradient Descent

Four primary problems necessitate advanced optimizers:

**Challenge 1: Choosing the Right Learning Rate**  
- Difficulty in finding an optimal, fixed learning rate  
- **Solution:** Learning Rate Scheduling (pre-defined reduction over time)  
- **Limitation:** Must be pre-set, may not adapt to all datasets  

**Challenge 2: Uniform Learning Rate for All Parameters**  
- Basic Gradient Descent applies the same learning rate to every weight  
- Reality: Different weights may require different update aggressiveness  

**Challenge 3: Local Minima Entrapment**  
- Loss landscapes contain multiple pits (local minima) and one deepest pit (global minimum)  
- Basic optimizers frequently get stuck in local minima, missing superior global solutions  

**Challenge 4: Saddle Point Problem**  
- Saddle Points: Flat regions where gradient is zero in some directions but not others  
- Basic Gradient Descent becomes "stuck" despite nearby lower points  

---

### 5. Advanced Optimizers

To address these challenges, more sophisticated optimizers have been developed:

**Upcoming Optimizers to Explore:**

- **Momentum** – Utilizes moving average of gradients  
- **Adagrad** – Adapts learning rates per parameter  
- **RMSprop** – Improves upon Adagrad with moving averages  
- **Adam** – Most popular and widely used optimizer  
- **Nadam** – Combines Adam with Nesterov acceleration

---

# Exponentially Weighted Moving Average (EWMA)

## Overview

Exponentially Weighted Moving Average (EWMA) is a statistical technique used to analyze time-series data by assigning different weights to observations, with more recent data points receiving higher weights than older ones.

## Key Concepts

### What is EWMA?

EWMA is a technique to identify trends in time-series data where:

- Data points are ordered chronologically (e.g., daily stock prices, temperature recordings)
- More recent observations receive higher weights compared to older ones
- The weight of any data point decreases exponentially over time

### Mathematical Formulation

The EWMA at time $t$ is calculated as:

$$
V_t = \beta \cdot V_{t-1} + (1 - \beta) \cdot \theta_t
$$

Where:

- $V_t$ = EWMA at current time $t$  
- $V_{t-1}$ = EWMA at previous time $t-1$  
- $\theta_t$ = Actual data value at time $t$  
- $\beta$ = Decay parameter $(0 \leq \beta \leq 1)$  

### Parameter $\beta$ (Beta)

The $\beta$ parameter controls how much weight is given to historical data:

- **High $\beta$ (e.g., 0.9)**: More emphasis on past observations, smoother trend line  
- **Low $\beta$ (e.g., 0.1)**: More responsive to recent changes, follows current data closely  
- **Common Practice**: $\beta = 0.9$ is frequently used in optimization algorithms  

### Mathematical Intuition

Expanding the EWMA formula recursively:

$$
\begin{aligned}
V_t &= \beta \cdot V_{t-1} + (1 - \beta) \cdot \theta_t \\
V_t &= \beta \cdot \left[ \beta \cdot V_{t-2} + (1 - \beta) \cdot \theta_{t-1} \right] + (1 - \beta) \cdot \theta_t \\
V_t &= \beta^2 \cdot V_{t-2} + \beta(1 - \beta) \cdot \theta_{t-1} + (1 - \beta) \cdot \theta_t
\end{aligned}
$$

This shows that:

- Current data point $\theta_t$ has weight $(1 - \beta)$  
- Previous data point $\theta_{t-1}$ has weight $\beta(1 - \beta)$  
- Data point from two steps back $\theta_{t-2}$ has weight $\beta^2(1 - \beta)$  
- The weights decrease exponentially as we go further back in time  

### Applications

EWMA is widely used in:

1. **Time Series Forecasting**  
2. **Financial Analysis** (e.g., stock price trends)  
3. **Signal Processing**  
4. **Deep Learning** (optimization algorithms like Momentum)  

### Key Properties

1. **Recency Effect**: Newer data points have higher influence  
2. **Exponential Decay**: Weight decreases exponentially for older observations  
3. **Memory**: The algorithm "remembers" past observations but with diminishing importance  
4. **Smoothness**: Higher $\beta$ values result in smoother trend lines  

## Importance in Deep Learning

EWMA forms the foundation for several optimization algorithms:

- **Momentum Optimization**  
- **RMSprop**  
- **Adam Optimizer**  

Understanding EWMA enables optimizers to:

- Maintain a "velocity" of parameter updates  
- Reduce oscillations in gradient descent  
- Accelerate convergence toward minima

---

## Momentum Optimization: Accelerating Neural Network Training

### 1. Overview and Core Concept

**Momentum Optimization** is a powerful technique designed to accelerate the training of neural networks and improve convergence efficiency, especially in complex, non-convex optimization landscapes.

**Core Idea:** Momentum mimics the physical property of inertia—like a ball rolling downhill. It uses the history of past gradient directions to build "velocity," allowing the optimization process to move faster in consistent directions and overcome small obstacles.  

**Analogy:** A ball with accumulated momentum can easily roll through small bumps (local minima) that would otherwise trap a slower object (Standard Gradient Descent).

### 2. Mathematical Formulation

Momentum introduces a **velocity vector ($v_t$)** that tracks the exponentially weighted moving average of past gradients. The parameter update is applied in the direction of this velocity, not the instantaneous gradient.

#### Standard Gradient Descent (for comparison)

$$
w_{t+1} = w_t - \eta \cdot \nabla J(w_t)
$$

#### Momentum Optimization

Let’s rename:

$$
v_t = \tilde{g_t}
$$

Now our recursion looks like:

$$
v_t = \beta v_{t-1} + (1 - \beta) g_t
$$

and our update rule is:

$$
w_{t+1} = w_t - \eta v_t
$$

Now expand the update over two steps:

$$
w_{t+1} = w_t - \eta (\beta v_{t-1} + (1 - \beta) g_t)
$$

but we know that

$$
w_t = w_{t-1} - \eta v_{t-1}
$$

Substitute this:

$$
w_{t+1} = w_{t-1} - \eta v_{t-1} - \eta (\beta v_{t-1} + (1 - \beta) g_t)
$$

Simplify:

$$
w_{t+1} = w_{t-1} - \eta (1 + \beta) v_{t-1} - \eta (1 - \beta) g_t
$$

Now see what appears: \(v_{t-1}\), which is proportional to \(w_t - w_{t-1}\).

This means:

$$
w_{t+1} - w_t = -\eta (1 - \beta) g_t - \eta \beta (w_t - w_{t-1})
$$

That second term, $\beta (w_t - w_{t-1})$, is literally **velocity**.  
It says: “Keep moving in the previous direction with damping factor $\beta$.”

### 3. The Role of Beta ($\beta$) and Mathematical Intuition

The $\beta$ parameter is crucial as it dictates the algorithm’s **memory**.

- **Low-Pass Filter:** The velocity term $v_t$ acts as a low-pass filter on the gradients, effectively smoothing out high-frequency noise and amplifying consistent directional signals.  
- **Averaging Window:** $\beta$ determines the effective averaging window, approximately $\frac{1}{1-\beta}$.  
- **Typical Values:**  
  - $\beta = 0.9$ (which means $\approx 10$ past gradients contribute)  
  - $\beta = 0.99$ (for greater historical influence)  
- $\beta = 0$: The formula reverts to standard Gradient Descent $v_t = \nabla J(w_t)$.  
- $\beta \approx 1$: The velocity term has infinite memory, leading to excessive oscillation.

### 4. Benefits: Why Momentum Works

Momentum addresses the difficulties of **non-convex optimization** (local minima, saddle points, high curvature) by providing three key benefits:

1. **Faster Convergence:** By accumulating velocity in consistent directions, momentum speeds up movement along shallow dimensions and reduces oscillations in steep, narrow valleys (high-curvature areas), leading to quicker training times.  
2. **Escape from Local Minima:** The accumulated velocity carries the optimizer through small loss valleys (“bumps”), enabling it to escape suboptimal local minima that would otherwise trap standard Gradient Descent.  
3. **Smoother Navigation:** Momentum averages out noisy gradients, providing a more stable update direction and preventing the optimizer from being jerked around by noisy data, especially when using mini-batches.

### 5. Limitations and Best Practices

While highly effective, Momentum introduces a primary challenge related to overshooting the optimum.

#### The Oscillation Problem

* **Issue:** If the accumulated velocity ($v_t$) is too high (due to high $\beta$), the algorithm can overshoot the minimum. This causes the path to spiral or oscillate around the optimum before finally settling down as the velocity naturally dampens.  
* **Analogy:** The ball rolling with too much speed swings past the lowest point of the valley.

#### Best Practices for Tuning

Momentum requires careful tuning to balance speed and stability:

| Parameter | Too Low (e.g., $\beta < 0.9$) | Too High (e.g., $\beta \approx 1$) | Default/Recommendation |
| :--- | :--- | :--- | :--- |
| **Momentum ($\beta$)** | Behaves like slow Standard GD | Excessive oscillation, slow settling | **0.9** (Good default) or 0.99 |
| **Learning Rate ($\eta$)** | Slow convergence | Divergence (moves away from minimum) | Requires adjustment from GD values |

Momentum is highly recommended for deep networks with complex loss landscapes where training speed and robustness against local minima are critical.

---

# Nesterov Accelerated Gradient (NAG): An Intelligent Optimization Technique

## Overview

Nesterov Accelerated Gradient (NAG) improves upon standard Momentum by incorporating a **"look-ahead" mechanism**, reducing oscillations and accelerating convergence in deep learning models.

## Momentum vs NAG

### Standard Momentum
Oscillations often occur due to velocity buildup, delayed response, and overshooting:

$$
v_t = \beta v_{t-1} + \eta \nabla J(w_t), \quad w_{t+1} = w_t - v_t
$$

### NAG Formulation
NAG anticipates the next position before calculating the gradient:

$$
\text{look\_ahead} = w_t - \beta v_{t-1}, \quad 
v_t = \beta v_{t-1} + \eta \nabla J(\text{look\_ahead}), \quad 
w_{t+1} = w_t - v_t
$$

**Key Difference:**  
- Momentum: Gradient at current position \(w_t\)  
- NAG: Gradient at anticipated position \(w_t - \beta v_{t-1}\)

### Geometric Interpretation
- **Momentum:** Moves blindly using past velocity, often overshooting minima.  
- **NAG:** Looks ahead, detects curvature changes, and adjusts trajectory proactively.  
- **Effect:** Smoother convergence with reduced oscillations.

## Parameter Dynamics

- **Momentum coefficient** $\beta$: High $\beta$ (e.g., $\beta = 0.9$) can cause oscillations in standard momentum, but NAG maintains stability.  
- **Learning rate** $\eta$: Typically $\eta = 0.01$, works well for both methods, but NAG offers better control.

## Performance Characteristics

**Advantages:**
1. Faster convergence with minimal oscillations  
2. Improved stability in high-curvature regions  
3. Intelligent adaptation to changing loss landscape  

**Potential Limitations:**
- Reduced ability to escape shallow minima in highly non-convex landscapes  
- Slightly higher computational cost per iteration

## Implementation (Keras)

```python
from tensorflow.keras.optimizers import SGD

# Standard Momentum
momentum_optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=False)

# Nesterov Accelerated Gradient
nag_optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

```

## Parameter Dynamics

### Beta ($\beta$) Parameter Role
- **High β (0.9)**: Strong momentum memory, potential oscillations
- **Medium β (0.5-0.8)**: Balanced approach
- **NAG Advantage**: Can utilize higher β values while maintaining stability

## Performance

**Advantages:**
1. Faster convergence with minimal oscillations  
2. Stable in high-curvature regions  
3. Anticipates landscape changes to minimize overshooting  

**Limitations:**
- May struggle with shallow local minima  
- Slight computational overhead per iteration

## Application Scenarios

- **Deep Neural Networks:** Efficiently handles complex loss landscapes  
- **High-Curvature Problems:** Excels in ravines and curved paths  
- **Production Systems:** Stable training with less hyperparameter tuning  

**Alternatives:** Basic SGD for simple convex problems or minimal compute scenarios.

## Comparative Summary

| Aspect | Standard Momentum | NAG |
|--------|-----------------|-----|
| Convergence Speed | Fast | Faster |
| Oscillation Control | Poor | Excellent |
| Parameter Sensitivity | Medium | Low |
| Implementation Complexity | Simple | Simple |
| Training Stability | Moderate | High |

## Strategic Insights

- **Predictive Intelligence:** Anticipates gradient changes  
- **Momentum Preservation:** Maintains acceleration benefits with control  
- **Efficient Navigation:** Reduces computational waste  

**Recommendation:** Start with β=0.9 and η=0.01; NAG is often a superior default choice.

## Conclusion

NAG transforms momentum from a blind force into an **intelligent guidance system**. Its "look-ahead" mechanism improves optimization behavior with minimal overhead, making it a cornerstone technique in modern deep learning.

---

# Adaptive Gradient (AdaGrad) Optimizer: An In-Depth Look

## 1. Introduction and Core Mechanism

The **Adaptive Gradient ($\text{AdaGrad}$)** optimizer — short for *Adaptive Gradient* — is an influential optimization algorithm that improves upon fixed learning rate methods like standard Gradient Descent and Momentum.

- **Core Idea:** $\text{AdaGrad}$ dynamically adjusts the learning rate during training. Instead of a fixed learning rate, it adapts **per parameter** based on historical gradient information.
- **Per-Parameter Learning Rate:** Each weight ($W$) and bias ($B$) in the model receives its own adaptive learning rate.

## 2. Advantages — When $\text{AdaGrad}$ Performs Well

$\text{AdaGrad}$ particularly excels in two key situations where fixed-rate optimizers struggle:

1. **Features with Different Scales** — Works well when input features vary widely in magnitude  
   (e.g., CGPA on a 0–10 scale vs. Salary on a 0–200,000 scale).

2. **Sparse Features** — Effective for sparse datasets (e.g., "Is IIT graduate?" where most values are 0).  
   Parameters corresponding to rare features receive larger updates.

## 3. The Problem — The Elongated Bowl & Sparse Data

Sparse data distorts the loss landscape, creating the **Elongated Bowl Problem**:

- **Loss Surface:** Becomes stretched and narrow (like a ravine) instead of circular.
- **Behavior of Standard Optimizers:** They move quickly along steep directions (non-sparse features) but slowly along flat regions (sparse features).
- **Mathematical Reason:** Small gradients on sparse features lead to negligible parameter updates, slowing learning in those directions.

## 4. $\text{AdaGrad}$’s Solution — Gradient Scaling

$\text{AdaGrad}$ resolves this inefficiency through **adaptive gradient scaling**, balancing updates for both sparse and dense features.

### Intuition

- Large gradients → smaller updates (learning rate decreases)
- Small gradients → larger updates (learning rate increases)

### Mathematical Formulation

The weight update rule for parameter $W$ at time step $t+1$ is:

$$
W_{t+1} = W_t - \frac{\eta}{\sqrt{V_t + \epsilon}} \cdot \nabla J(W_t)
$$

where $V_t$ is the **accumulated sum of squared gradients**:

$$
V_t = V_{t-1} + (\nabla J(W_t))^2
$$

### Term Breakdown

- **$V_t$ (Accumulator):** Non-decreasing sum of squared gradients for $W$.  
- **Adaptive Learning Rate:**  
  $$\frac{\eta}{\sqrt{V_t + \epsilon}}$$  
  acts as the **per-parameter adaptive step size**.
  - Large gradients $\Rightarrow$ $V_t$ increases $\Rightarrow$ smaller step.
  - Small gradients $\Rightarrow$ $V_t$ small $\Rightarrow$ larger step.
- **$\eta$:** Global base learning rate.
- **$\epsilon$:** Small constant (e.g., $10^{-8}$) to prevent division by zero.

## 5. Disadvantage — The Vanishing Learning Rate

While powerful, $\text{AdaGrad}$ has a major limitation:

- **Problem:** Learning rate decays too aggressively over time.  
- **Cause:** $V_t$ accumulates squared gradients indefinitely (non-decreasing).  
- **Effect:**  
As $V_t \to \infty$,

$$
\frac{\eta}{\sqrt{V_t + \epsilon}} \to 0
$$

causing the effective learning rate to vanish — training slows and may stop before convergence.



causing the effective learning rate to vanish — training slows and may stop before convergence.

### Successor Optimizers

To overcome this, newer optimizers modify $\text{AdaGrad}$:

- **RMSProp:** Uses an *exponentially decaying average* of squared gradients.  
- **Adam:** Combines RMSProp with Momentum for adaptive, stable updates.

## Summary Table

| Property | Description |
|-----------|-------------|
| **Core Idea** | Per-parameter adaptive learning rate |
| **Strengths** | Great for sparse and varied-scale features |
| **Weakness** | Learning rate vanishes over time |
| **Key Equation** | $W_{t+1} = W_t - \dfrac{\eta}{\sqrt{V_t + \epsilon}} \nabla J(W_t)$ |
| **Successors** | RMSProp, Adam |

### ✨ Key Takeaway

$\text{AdaGrad}$ introduced the concept of *adaptive learning rates*, revolutionizing optimization by letting each parameter learn at its own pace. Although its vanishing-rate issue limits use in deep networks, it remains foundational for understanding modern optimizers.

---

# RMSProp Optimizer: Solving the AdaGrad Dilemma

## 1. Introduction and Core Concept

The $\text{RMSProp}$ (Root Mean Square Propagation) optimization technique is a major **improvement over $\text{AdaGrad}$**.

- **Full Form:** Root Mean Square Propagation  
- **Context:** To understand $\text{RMSProp}$, recall that $\text{AdaGrad}$ optimizes models trained on **sparse data** (where many features are mostly zeros).

## 2. Recap of $\text{AdaGrad}$’s Problem

$\text{AdaGrad}$ addressed the **Elongated Bowl Problem**, improving optimization for sparse vs. non-sparse features.  
However, it suffers from a **Vanishing Learning Rate** issue.

- **$\text{AdaGrad}$’s Mechanism:**  
  Learning rate is scaled as $\frac{\eta}{\sqrt{V_t + \epsilon}}$,  
  where $V_t$ accumulates all past squared gradients.  

- **The Flaw:**  
  Since $V_t$ keeps increasing, the effective learning rate becomes extremely small.

- **Result:**  
  Updates nearly stop, preventing convergence to the global minimum — especially in complex, non-convex neural networks.

## 3. $\text{RMSProp}$ Solution: Exponentially Decaying Average

$\text{RMSProp}$ modifies $\text{AdaGrad}$ by replacing the cumulative sum with an **Exponentially Weighted Moving Average (EWMA)**.  
This ensures $V_t$ doesn’t grow indefinitely, avoiding vanishing learning rates.

### Mathematical Formulation

Parameter update rule (same as $\text{AdaGrad}$):

$$
W_{t+1} = W_t - \frac{\eta}{\sqrt{V_t + \epsilon}} \cdot \nabla J(W_t)
$$

Accumulator update (redefined):

$$
V_t = \beta \cdot V_{t-1} + (1 - \beta) \cdot (\nabla J(W_t))^2
$$

Where:

- $\beta$ : Decay rate hyperparameter (commonly $0.9$–$0.99$; e.g., $0.95$)
- $\eta$ : Global learning rate  
- $\epsilon$ : Small constant to prevent division by zero

**Mechanism:**
1. $\beta V_{t-1}$ keeps the previous average but decays it over time.
2. $(1 - \beta)(\nabla J(W_t))^2$ adds the current gradient’s contribution.
3. Thus, **recent gradients get more weight**, older gradients gradually fade.

**Benefit:**  
$V_t$ no longer explodes — keeping the effective learning rate stable and enabling smooth convergence.

## 4. Performance and Insights

- **Convex Problems:**  
  Performs similarly to $\text{AdaGrad}$ (since learning rate decay is mild).  

- **Non-Convex Problems (Deep NNs):**  
  Performs far better — continues updating parameters effectively without vanishing learning rate.

- **Empirical Strength:**  
  Before $\text{Adam}$, $\text{RMSProp}$ was the most widely used optimizer for neural networks.

- **Modern Usage:**  
  Still competitive and effective; some architectures perform better with $\text{RMSProp}$ than with $\text{Adam}$.

 **Summary Table**

| Concept | AdaGrad | RMSProp |
|----------|----------|----------|
| Accumulation Type | Cumulative sum of squared gradients | Exponentially decaying average |
| Learning Rate Behavior | Shrinks continuously (vanishing) | Stabilized (no vanishing) |
| Best For | Sparse, convex problems | Non-convex, deep neural networks |
| Key Parameter | $\eta$ | $\eta$, $\beta$ |

---

# ADAM (Adaptive Moment Estimation) Optimizer: The Deep Learning Workhorse

## 1. Overview

The **ADAM optimizer** is one of the most widely used algorithms in deep learning today.  
It combines the **momentum** concept from the *Momentum Optimizer* and the **adaptive learning rate scaling** from *RMSProp* — effectively merging the best of both worlds.

> ADAM = Momentum + RMSProp

This fusion gives ADAM both **stability** (from momentum) and **adaptivity** (from RMSProp).

## 2. Core Concepts

### Momentum (First Moment)

- Maintains an **exponentially weighted average of past gradients**.
- Acts like velocity — helps move consistently in the right direction and reduces oscillations.

### Adaptive Learning Rate (Second Moment)

- Maintains an **exponentially weighted average of squared gradients**.
- Adapts learning rate for each parameter based on how large or small its gradients are.
- Handles **sparse data** and features with varying scales efficiently.

## 3. Mathematical Formulation

ADAM maintains **two moment estimates** for each parameter:

**First Moment (Momentum term):**

$$
M_t = \beta_1 \cdot M_{t-1} + (1 - \beta_1) \cdot \nabla J(W_t)
$$

**Default:** $\beta_1 = 0.9$

**Second Moment (RMS term):**

$$
V_t = \beta_2 \cdot V_{t-1} + (1 - \beta_2) \cdot (\nabla J(W_t))^2
$$

**Default:** $\beta_2 = 0.999$


### Bias Correction

Since both $M_t$ and $V_t$ start at zero, ADAM applies **bias correction**:

$$
\hat{M}_t = \frac{M_t}{1 - \beta_1^t}, \quad \hat{V}_t = \frac{V_t}{1 - \beta_2^t}
$$



### Parameter Update Rule

Finally, the parameter update combines both corrected moments:

$$
W_{t+1} = W_t - \frac{\eta}{\sqrt{\hat{V}_t} + \epsilon} \cdot \hat{M}_t
$$

Where:
- $\eta$ → Base learning rate (default $0.001$)
- $\epsilon$ → Small constant (e.g., $10^{-8}$) to prevent division by zero  
- $\hat{M}_t$ → Controls **direction** (momentum)
- $\sqrt{\hat{V}_t}$ → Controls **scaling** (adaptive rate)
  
## 4. Key Insights

| Concept | Description |
|----------|-------------|
| **Momentum** | Smooths updates and accelerates movement along consistent directions. |
| **Adaptive Rate** | Adjusts learning rate per parameter using historical gradient information. |
| **Bias Correction** | Eliminates initialization bias of early moments. |
| **Efficiency** | Requires little tuning and converges fast even on noisy or sparse gradients. |

## 5. Performance Summary

- **Speed:** Fast convergence due to combined advantages of momentum and adaptive scaling.  
- **Stability:** Handles noisy gradients and complex, non-convex loss surfaces.  
- **Versatility:** Performs well on CNNs, RNNs, Transformers, and most deep architectures.  
- **Default Choice:** ADAM is the **go-to optimizer** in most frameworks (TensorFlow, PyTorch, etc.).

**Final Takeaway**

> ADAM brings the best of both momentum and adaptivity.  
> It is the *default and most reliable optimizer* for training modern deep neural networks.

| Hyperparameter | Typical Value | Description |
|----------------|----------------|-------------|
| $\eta$ | 0.001 | Base learning rate |
| $\beta_1$ | 0.9 | Decay for first moment |
| $\beta_2$ | 0.999 | Decay for second moment |
| $\epsilon$ | $10^{-8}$ | Numerical stability constant |

---
