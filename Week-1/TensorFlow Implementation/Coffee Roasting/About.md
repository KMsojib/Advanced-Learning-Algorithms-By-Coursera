# Coffee Roasting Neural Network Model

## Introduction
This project builds a simple neural network to determine whether a given coffee roasting temperature and duration will result in good coffee or not. The model takes in temperature and duration as input and predicts **YES** (Good Coffee) or **NO** (Bad Coffee) based on a binary classification using a sigmoid activation function.

## Problem Statement
Given an input feature vector:

**X = [temperature, duration]**

For example:
```
X = [200.0, 17.0]
```
The neural network will predict whether this setting will yield good coffee.

## Neural Network Architecture
The model consists of **two layers**:

### **Layer 1**
- Input: X (temperature, duration)
- Neurons: 3
- Activation Function: **Sigmoid**
- Output: a1

```python
X = np.array([[200.0, 17.0]])
Layer_1 = Dense(units=3, activation='sigmoid')
a1 = Layer_1(X)
```

### **Layer 2**
- Neurons: 1
- Activation Function: **Sigmoid**
- Output: a2 (Final Prediction)

```python
Layer_2 = Dense(units=1, activation='sigmoid')
a2 = Layer_2(a1)
```

## Prediction Rule
After processing through the network:

```
If a2 >= 0.5:
    Yhat = 1 (YES - Good Coffee)
else:
    Yhat = 0 (NO - Bad Coffee)
```

## Model Summary
- **Model Name**: Coffee Roasting Model
- **Input Features**: Temperature, Duration
- **Hidden Layer**: 3 neurons, Sigmoid Activation
- **Output Layer**: 1 neuron, Sigmoid Activation
- **Output Interpretation**:
  - **Yhat = 1** → "Yes" (Good Coffee)
  - **Yhat = 0** → "No" (Bad Coffee)

## Requirements
- Python 3.x
- TensorFlow
- NumPy

## Implementation Steps
1. Define the neural network layers.
2. Use sigmoid activation functions.
3. Input temperature and duration.
4. Compute the prediction using the model.

## Usage
You can modify the temperature and duration values in `X` to test different settings for coffee roasting.

## Conclusion
This simple neural network helps in predicting whether the given roasting settings result in a **good** or **bad** coffee. The sigmoid function ensures binary classification, making it suitable for this task.

