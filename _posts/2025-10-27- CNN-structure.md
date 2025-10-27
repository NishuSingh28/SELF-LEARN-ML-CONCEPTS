---
layout: default
title: "CNN AND OTHER MODELS"
date: 2025-10-27
categories: [deep-learning]
---

# Understanding CNN Architecture (Step-by-Step with MathJax)

A Convolutional Neural Network (CNN) is designed to automatically extract hierarchical features from images. Each layer has a clear purpose: detect simple features first, then combine them into complex representations, and finally classify the image.

## 1. Input Layer

Let the input image be:

$$
\mathbf{X} \in \mathbb{R}^{H \times W \times C}
$$

- $H$ = height, $W$ = width, $C$ = number of channels (3 for RGB).  
**Why:** The input layer preserves the spatial structure of the image for feature extraction.

## 2. Convolutional Layer

Convolution applies $K$ filters of size $F \times F$ across the image. The output feature map size is:

$$
O_H = \frac{H + 2P - F}{S} + 1, \quad
O_W = \frac{W + 2P - F}{S} + 1
$$

- $P$ = padding (to preserve borders)  
- $S$ = stride (step size of filter)  
- Number of parameters:  

$$
\text{Params} = (F \cdot F \cdot C + 1) \cdot K
$$

**Why:** Convolution detects local patterns such as edges, corners, and textures. Using multiple filters allows the network to capture different features simultaneously.

## 3. Activation Layer

After convolution, we apply a non-linear activation function. The most common is ReLU:

$$
f(x) = \max(0, x)
$$

**Why:** Without non-linearity, CNNs would be equivalent to a linear model, unable to learn complex patterns.

## 4. Pooling Layer

Pooling reduces spatial dimensions while keeping essential information. For a window size $F_p \times F_p$ and stride $S_p$:

$$
O_H^{\text{pool}} = \frac{O_H - F_p}{S_p} + 1, \quad
O_W^{\text{pool}} = \frac{O_W - F_p}{S_p} + 1
$$

- **Max pooling:** $y = \max(x_{i,j})$  
- **Average pooling:** $y = \frac{1}{F_p^2} \sum x_{i,j}$  

**Why:** Pooling reduces computation, introduces translation invariance, and prevents overfitting.

## 5. Repeating Conv → Activation → Pooling

We can repeat this block multiple times:

$$
[\text{Conv} \rightarrow \text{ReLU} \rightarrow \text{Pooling}]^N
$$

**Why:** Early layers capture low-level features (edges), deeper layers capture high-level features (object parts).

## 6. Flatten Layer

After the last pooling layer, the 3D feature map is flattened into a 1D vector:

$$
\mathbf{x}_{\text{flat}} \in \mathbb{R}^{O_H^{\text{pool}} \cdot O_W^{\text{pool}} \cdot K}
$$

**Why:** Fully connected layers expect 1D input to perform global reasoning.

## 7. Fully Connected (Dense) Layer

Each neuron connects to all inputs:

$$
\mathbf{y} = \mathbf{W}\mathbf{x}_{\text{flat}} + \mathbf{b}
$$

- Parameters:

$$
\text{Params} = N_{\text{in}} \cdot N_{\text{out}} + N_{\text{out}}
$$

**Why:** Fully connected layers combine features learned by convolution to make final decisions.

## 8. Softmax Output Layer

For classification into $C$ classes:

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}, \quad i = 1,2,...,C
$$

**Why:** Converts network outputs (logits) into probabilities.

## 9. End-to-End Workflow

1. **Input image**: $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$.  
2. **Convolution**: Extract local features.  
3. **Activation (ReLU)**: Introduce non-linearity.  
4. **Pooling**: Reduce dimensions and retain strong features.  
5. **Repeat** convolution → activation → pooling as needed.  
6. **Flatten**: Convert 3D feature maps to 1D vector.  
7. **Fully connected layers**: Combine features globally.  
8. **Softmax**: Predict class probabilities.

## 10. Key Design Principles

- Spatial hierarchy: small patterns first, then larger structures.  
- Number of filters increases in deeper layers for richer representations.  
- Pooling helps control overfitting and computation.  
- Activation functions (like ReLU) enable learning complex mappings.  
- CNNs are end-to-end trainable: from raw image to class prediction.

**Summary:** A CNN transforms raw pixel data into meaningful features, condenses them, and uses fully connected layers to perform classification. Each step has a clear mathematical foundation.

---

# CNN vs ANN

Understanding why Convolutional Neural Networks (CNNs) are essential for image processing requires comparing them with traditional Artificial Neural Networks (ANNs) and highlighting their architectural advantages.

## 1. The Motivation for CNNs

Using a standard ANN for image classification has three major limitations:

1. **High Computational Cost** – Flattening images results in a huge number of input neurons, leading to massive parameter counts.
2. **Overfitting** – With so many parameters, the network can memorize the training data instead of learning meaningful patterns.
3. **Loss of Spatial Arrangement** – Flattening images destroys the spatial structure of pixels and the local patterns that are crucial for recognition.

CNNs were designed to address all three of these issues efficiently.

## 2. How ANN and CNN Differ in Handling Images

Consider grayscale images of size $28 \times 28 \times 1$ (like the MNIST dataset):

### **ANN Approach**

- Flatten the image into a vector of $28 \times 28 = 784$ input neurons.
- Each neuron in the hidden layer connects to all 784 input neurons.
- **Mathematical operation:**  

$$
y = f(\mathbf{W} \cdot \mathbf{x} + b)
$$

- **Problem:** Flattening destroys spatial information, and the number of parameters grows rapidly.

### **CNN Approach**

- Keep the image in its 3D shape: $28 \times 28 \times 1$.
- Apply convolutional filters that slide across the image to detect local patterns.
- **Mathematical operation (convolution):**  

$$
y = f\left(\sum_{i,j,k} X_{i,j,k} \cdot W_{i,j,k} + b\right)
$$

- **Advantage:** Filters connect only to a small local region (receptive field), preserving spatial structure and reducing parameters.

## 3. Key Insight: Mathematical Similarity

Both ANNs and CNNs perform the **same fundamental operations**:

1. **Weighted sum:** Multiply inputs by weights and sum them.
2. **Add bias:** Include an offset term.
3. **Non-linear activation:** Pass the result through an activation function.

The difference lies in **connectivity**:

- ANN: Each neuron is fully connected to all inputs.
- CNN: Each filter is connected only to a local region, which slides across the input.

  
## 4. Local vs Global Connectivity

- **ANN:** Global connectivity – every neuron sees the entire input image.
- **CNN:** Local connectivity – each filter sees only a small patch of the image and gradually scans across the entire image.
- **Benefit:** CNNs can learn localized features first (edges, textures) and build up to global patterns efficiently.

## 5. Parameter Efficiency

Consider a color image of size $224 \times 224 \times 3$:

### **CNN Parameter Calculation**

- Filter size: $5 \times 5 \times 3 = 75$ weights per filter
- Add 1 bias term → 76 parameters per filter
- 32 filters → $32 \times 76 = 2,432$ total parameters  

**Observation:** The number of parameters is independent of the input image size.

### **ANN Parameter Calculation**

- Input neurons: $224 \times 224 \times 3 = 150,528$
- Hidden layer: 1,000 neurons
- Total parameters:  

$$
150,528 \times 1,000 + 1,000 = 150,529,000
$$

**Observation:** Increasing the image size dramatically increases the parameter count, making ANN impractical for large images.

## 6. Implications for Backpropagation

Understanding that CNNs perform the same basic operations as ANNs allows us to apply the **backpropagation algorithm** with modifications for convolutional layers:

- Compute gradients of the convolutional filters.
- Update weights and biases just like in ANN.
- The local connectivity and shared weights significantly reduce computation while still learning meaningful patterns.

## 7. Summary of Advantages

1. **Computational Efficiency** – CNNs drastically reduce the number of trainable parameters.  
2. **Reduced Overfitting** – Fewer parameters mean less chance of memorizing the training data.  
3. **Spatial Awareness** – CNNs naturally capture local and global patterns without flattening.  
4. **Scalability** – CNNs handle large images without exploding parameter counts.  
5. **Automatic Feature Learning** – Filters automatically learn to recognize edges, textures, and complex patterns.  


## 8. Conclusion

CNNs maintain the **mathematical power of neural networks** while being **optimized for spatial data like images**. Their architecture enables efficient, scalable learning of hierarchical features, making them the foundation of modern computer vision.

By understanding the differences between ANN and CNN, one can appreciate why CNNs revolutionized image processing tasks: they combine **mathematical rigor** with **practical efficiency**, automatically recognizing patterns regardless of their location in the image.

