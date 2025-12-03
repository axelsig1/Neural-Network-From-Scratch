# Neural Network From Scratch

A from-scratch implementation of a neural network in Python, built to understand and learn the fundamentals of deep learning without relying on high-level frameworks.

## Overview

This project implements core neural network components including:

- **Dense Layers**: Fully connected neural network layers with forward and backward passes
- **Activation Functions**: 
  - ReLU (Rectified Linear Unit)
  - Softmax
- **Loss Functions**: Categorical Cross-Entropy loss
- **Optimizers**: Stochastic Gradient Descent (SGD)

## Project Structure

```
├── Main_code.py          # Complete neural network implementation with training loop
├── p1.py                 # Neural network variant with decision boundary visualization
├── p6-1.py               # Additional implementation/experiment
├── p7.py                 # Additional implementation/experiment
├── p9.py                 # Additional implementation/experiment
├── derivative.py         # Derivative calculations and utility functions
├── cd.py                 # Additional utility module
└── README.md             # This file
```

## Features

### Neural Network Architecture

- Multi-layer dense network with configurable layer sizes
- ReLU activation for hidden layers
- Softmax activation with categorical cross-entropy loss for classification
- Backpropagation with automatic gradient calculation

### Training

- Forward pass through all layers
- Loss calculation using categorical cross-entropy
- Backward pass with gradient computation
- Parameter updates using SGD optimizer
- Accuracy tracking during training

### Visualization

- Decision boundary visualization
- Training metrics display (loss, accuracy, weight shapes)

## Requirements

```
numpy
nnfs
matplotlib
```

## Installation

```bash
pip install numpy nnfs matplotlib
```

## Usage

### Training a Model

```python
# Create dataset (spiral data with 100 samples and 3 classes)
X, y = spiral_data(samples=100, classes=3)

# Build network layers
dense1 = Layer_Dense(2, 64)      # Input: 2 features -> 64 neurons
dense2 = Layer_Dense(64, 64)     # 64 -> 64 neurons
dense3 = Layer_Dense(64, 3)      # 64 -> 3 output classes

# Create optimizer
optimizer = Optimizer_SGD(learning_rate=0.85)

# Training loop
for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    
    # Calculate loss
    loss = loss_activation.forward(dense3.output, y)
    
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense3.backward(loss_activation.dinputs)
    # ... more backward passes
    
    # Update parameters
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
```

## Main Components

### Layer_Dense
Implements a fully connected layer with:
- Weight and bias initialization
- Forward pass: `output = input @ weights + biases`
- Backward pass: gradient computation for backpropagation

### Activation_ReLU
Rectified Linear Unit activation:
- Forward: `output = max(0, input)`
- Backward: passes gradients only for positive inputs

### Activation_Softmax
Softmax activation for multi-class classification:
- Converts logits to probability distribution
- Backward pass uses Jacobian matrix computation

### Loss_CategoricalCrossentropy
Cross-entropy loss for multi-class classification:
- Supports both sparse labels (single class index) and one-hot encoded labels
- Forward: computes cross-entropy loss
- Backward: computes gradient w.r.t. network output

### Optimizer_SGD
Stochastic Gradient Descent:
- Updates layer weights and biases
- Configurable learning rate

## Learning Outcomes

This implementation demonstrates:

- How neural networks perform forward and backward propagation
- Gradient computation and parameter updates
- Loss calculation for classification tasks
- How different activation functions affect learning
- The importance of proper data representation (sparse vs one-hot labels)

## Notes

- This is an educational implementation optimized for understanding, not performance
- For production use, consider frameworks like TensorFlow, PyTorch, or JAX
- The decision boundary visualization helps understand how the network separates classes

## Motivation

Created as a learning project for understanding neural network fundamentals from scratch.

---

**Language**: Python 3  
**Last Updated**: November 2025
