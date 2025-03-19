# Handwritten Digit Recognition Neural Network

This project is a simple implementation of a neural network for **handwritten digit recognition** using the **MNIST dataset**. The model is built from scratch to practice concepts learned in the Neural Network course at **EMU**.

## Overview

The goal of this project is to classify handwritten digits (0-9) from images. The neural network is implemented using **Python**, **NumPy**, and **Pandas** for data manipulation, and **Matplotlib** for visualization. The model is trained on the MNIST dataset and uses forward and backward propagation, ReLU activation, softmax output, and gradient descent for optimization.

### Key Features
- **Data Preprocessing**: The dataset is loaded, shuffled, normalized, and split into training and development sets.
- **Neural Network**: A simple feed-forward neural network with one hidden layer.
- **Activation Function**: ReLU is used for the hidden layer, and softmax is used for the output layer to handle multi-class classification.
- **Backpropagation**: Gradient descent is used to update the model weights.
- **Testing**: The model is tested on individual images, and predictions are displayed alongside the actual labels.

## Technologies Used
- **Python** (core programming language)
- **NumPy** (for array and matrix operations)
- **Pandas** (for data handling)
- **Matplotlib** (for data visualization)
- **MNIST Dataset** (for training and testing)

### Training the Model
The neural network is trained using the `gradient_descent` function with a learning rate of `0.50` for `500` iterations.

### Test Predictions
To test the model with any random image, call the `test_prediction` function by passing the index of the image you want to test:
```python
test_prediction(40000, W1, b1, W2, b2)
