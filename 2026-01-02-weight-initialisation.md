---
layout: default
title: "WEIGHT INITIALISATION"
date: 2026-01-02
categories: [deep-learning]
---

# Deep Learning Weight Initialization – Complete Interview Q&A (End-to-End)

---

## Q1. Why is weight initialization important in deep neural networks?

### Answer:
Weight initialization is the process of setting initial weight values before training starts.  
It is crucial because it:
- ensures stable **gradient flow**
- prevents **vanishing and exploding gradients**
- **breaks symmetry** between neurons
- improves **training speed and convergence**

If weights are:
- **too small** → gradients shrink exponentially → vanishing gradients
- **too large** → gradients grow exponentially → exploding gradients
- **all same or zero** → neurons learn identical features → ineffective learning

Proper initialization preserves variance of activations and gradients across layers.

---

## Q2. What can go wrong if all weights are initialized to zero?

### Answer:
- All neurons in a layer produce identical outputs
- They receive identical gradients during backpropagation
- Symmetry is not broken
- Neurons learn the same features
- Network behaves like a single neuron

Therefore, zero initialization prevents meaningful learning.

---

## Q3. Why does initializing all weights to the same value fail?

### Answer:
- Causes **symmetry problem**
- Neurons become indistinguishable
- Gradients remain identical
- Feature diversity is lost

Random initialization is required to break symmetry.

---

## Q4. Why can biases be initialized to zero but not weights?

### Answer:
- Weights connect neurons and cause symmetry issues if identical
- Biases only shift the activation function
- Biases do not affect symmetry or gradient variance
- Zero bias does not block learning

Special case:
- ReLU networks sometimes use small positive bias to avoid dead ReLU neurons

---

## Q5. What is the vanishing gradient problem?

### Answer:
- Gradients shrink exponentially during backpropagation
- Occurs when weights are too small or activations saturate
- Common with sigmoid/tanh in deep networks
- Leads to very slow or stalled learning in early layers

---

## Q6. What is the exploding gradient problem?

### Answer:
- Gradients grow exponentially during backpropagation
- Occurs when weights are too large
- Leads to unstable updates, NaNs, or divergence
- Common in deep networks and RNNs

---

## Q7. How does weight initialization affect gradient flow?

### Answer:
- Gradients are multiplied by weights at each layer
- Poor scaling causes exponential decay or growth
- Proper initialization preserves gradient magnitude
- Enables stable and efficient backpropagation

---

## Q8. What is Xavier (Glorot) initialization?

### Answer:
Xavier initialization is designed for **tanh / sigmoid** activations.

Goal:
- Preserve variance of activations and gradients across layers

Formula:
- Uniform:
  Var(W) = 2 / (fan_in + fan_out)
- Normal:
  Var(W) = 1 / fan_in

Works well for zero-centered activations.

---

## Q9. Why does Xavier initialization fail for ReLU?

### Answer:
- ReLU zeroes out ~50% of activations
- Variance decreases after activation
- Leads to vanishing gradients
- Xavier does not compensate for inactive neurons

---

## Q10. What is He (Kaiming) initialization and why is it better for ReLU?

### Answer:
He initialization is designed for **ReLU and its variants**.

Problem:
- ReLU deactivates half the neurons

Solution:
- Increase weight variance

Formula:
Var(W) = 2 / fan_in

Preserves variance and enables deep ReLU networks to train effectively.

---

## Q11. What happens if we initialize weights using N(0,1) in a deep network?

### Answer:
Forward pass:
- Activations grow exponentially
- Saturation and numerical instability

Backward pass:
- Gradients explode
- Unstable updates and divergence

Reason:
- Fixed variance ignores fan-in and network depth
- Does not preserve variance across layers

---

## Q12. What is fan-in and fan-out?

### Answer:
- **Fan-in**: number of input connections to a neuron
- **Fan-out**: number of output connections from a neuron

Dense layer:
- fan_in = number of input features
- fan_out = number of output neurons

Convolution layer:
- fan_in = kernel_height × kernel_width × input_channels
- fan_out = kernel_height × kernel_width × output_channels

---

## Q13. Why do fan-in and fan-out matter in weight initialization?

### Answer:
- Fan-in affects activation variance during forward pass
- Fan-out affects gradient variance during backward pass
- Proper scaling prevents vanishing/exploding gradients
- Used directly in Xavier and He initialization formulas

---

## Q14. Does Batch Normalization remove the need for weight initialization?

### Answer:
No.

Batch Normalization:
- normalizes activations
- stabilizes training
- reduces sensitivity to initialization

But:
- does not break symmetry
- does not fix poor initial forward pass
- weights still affect gradients

Proper initialization is still required.

---

## Q15. What types of weight initialization exist?

### Answer:
- Random initialization
- Zero initialization (invalid for weights)
- Xavier / Glorot
- He / Kaiming
- LeCun initialization
- Orthogonal initialization
- Identity initialization
- Sparse initialization
- Pretrained initialization
- Uniform vs Normal distributions

---

## Q16. What is LeCun initialization?

### Answer:
Used with **SELU activation**.

Formula:
Var(W) = 1 / fan_in

Maintains self-normalizing behavior of SELU networks.

---

## Q17. What is Orthogonal initialization?

### Answer:
- Weight matrix initialized as orthogonal
- Columns are orthonormal
- Preserves vector norm
- Prevents vanishing and exploding gradients

Commonly used in:
- RNNs
- LSTMs
- Deep linear networks

---

## Q18. Why is Orthogonal initialization effective for RNNs and LSTMs?

### Answer:
- Same weight matrix is reused across time steps
- Repeated multiplication causes gradient instability
- Orthogonal matrices preserve magnitude
- Stabilizes gradients over long sequences

Does not replace gating mechanisms but improves stability.

---

## Q19. What is identity initialization?

### Answer:
- Weights initialized as identity matrix
- Special case of orthogonal initialization
- Used in IRNNs and deep linear networks
- Preserves signal perfectly at initialization

---

## Q20. What is pretrained initialization?

### Answer:
- Weights initialized from a trained model
- Used in transfer learning
- Faster convergence
- Better generalization
- Most effective practical initialization

---

## Final One-Line Interview Summary

> Weight initialization is essential to break symmetry and preserve activation and gradient variance across layers, ensuring stable, efficient, and scalable deep learning training.
