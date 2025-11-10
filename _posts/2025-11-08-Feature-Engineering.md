---
layout: default
title: "FEATURE ENGINEERING"
date: 2025-09-25
categories: [machine-learning]
---
# **Feature Scaling: Normalization**

## ** Overview**
Normalization is a key **feature scaling** technique used in data preprocessing. It rescales numerical features to a **common scale** without distorting the relationships between values. This makes features comparable and improves algorithmic performance.


## **1️ What is Normalization?**
- **Definition:** Process of transforming numeric data to a standard scale, typically between 0 and 1.  
- **Goal:** Remove unit dependence (like meters, kilograms, etc.) and ensure all features contribute equally to the model.  
- **Use Case:** Especially useful for algorithms sensitive to magnitude differences (e.g., KNN, Gradient Descent).

## **2️ Types of Normalization**
1. **Min-Max Scaling** *(Most common)*
2. **Mean Normalization**
3. **Max-Abs Scaling**
4. **Robust Scaling**

## **3️ Min-Max Scaling**

### **Formula**
```python
X_new = (X_original - X_min) / (X_max - X_min)
```

### **Properties**
- Rescales data to range **[0, 1]**
- `X_min → 0` and `X_max → 1`
- Maintains shape of data distribution (only compresses scale)

### **Geometric Intuition**
Min-Max scaling **squeezes** the dataset into a **unit square (0 to 1)** on all axes — preserving relative positions but standardizing scale.

### **Code Example**
```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- Always `fit()` on training data and `transform()` both train and test.

## **4️ Other Normalization Techniques**

### **A. Mean Normalization**
```python
X_new = (X_original - Mean) / (X_max - X_min)
```
- Centers data around 0  
- Output typically lies in range **[-1, 1]**  
- Rarely used (often replaced by Standardization)

### **B. Max-Abs Scaling**
```python
X_new = X_original / |X_max|
```
- Scales features by their **maximum absolute value**
- Output in range **[-1, 1]**
- Works well with **sparse data** or data **centered around zero**
- Available via `MaxAbsScaler` in sklearn

### **C. Robust Scaling**
```python
X_new = (X_original - Median) / (Q3 - Q1)
```
- Uses **median** and **IQR (Interquartile Range)** for scaling  
- **Robust to outliers**
- Best for datasets with **extreme values**
- Available via `RobustScaler` in sklearn

## **5️ Normalization vs. Standardization**

| Criteria | **Normalization (Min-Max)** | **Standardization (Z-score)** |
|-----------|-----------------------------|--------------------------------|
| **Range** | [0, 1] | Mean = 0, Std = 1 |
| **Sensitive to outliers?** | Yes | Less sensitive |
| **Best for** | Bounded data (e.g., images) | Most ML algorithms |
| **Formula** | (X - X_min)/(X_max - X_min) | (X - mean)/std |

## **6️ When to Use What**

1. **Check if scaling is needed**  
   - Scaling required: KNN, K-Means, Gradient Descent  
   - Not required: Tree-based models (Random Forest, XGBoost)

2. **Prefer Standardization** for most ML tasks (default choice)

3. **Use Normalization** when:
   - Features have **known bounds** (e.g., pixel intensities 0–255)
   - Model requires **bounded input**

4. **Use Robust Scaling** if:
   - Data contains **outliers**

5. **Use Max-Abs Scaling** for:
   - **Sparse matrices** or data centered around 0

## **7️ Key Takeaways**
- Normalization ensures **uniform scale** across features.  
- Different techniques suit different data distributions.  
- **Min-Max Scaling** is simple and widely used.  
- **RobustScaler** is ideal for noisy datasets.  
- Always **fit scalers on training data** to prevent data leakage.

## ** Summary**
Normalization rescales data for consistency and comparability across features. It’s vital for models relying on distance or gradient-based computations. Choosing the right scaling method depends on data characteristics, algorithm type, and the presence of outliers.
