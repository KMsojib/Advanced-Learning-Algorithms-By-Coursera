import numpy as np

def dense(a_in, W, b, g=lambda x: x): 
    """
    Implements a dense (fully connected) layer.

    Args:
        a_in: Input to the layer (NumPy array).
        W: Weight matrix (NumPy array).
        b: Bias vector (NumPy array).
        g: Activation function (default is identity).

    Returns:
        Output activations of the layer (NumPy array).
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out

def sequential(x, W1, b1, W2, b2, W3, b3, W4, b4, g=lambda x: x):
    """
    Implements a sequential neural network.

    Args:
        x: Input to the network (NumPy array).
        W1, b1, W2, b2, W3, b3, W4, b4: Weights and biases for each layer.
        g: Activation function (default is identity).

    Returns:
        Final output of the network (NumPy array).
    """
    a1 = dense(x, W1, b1, g)
    a2 = dense(a1, W2, b2, g)
    a3 = dense(a2, W3, b3, g)
    a4 = dense(a3, W4, b4, g)
    return a4

# Example Usage of the functions
W = np.array([[1, -3, 5], [2, 4, -6]])
b = np.array([-1, 1, 2])
a_in = np.array([-2, 4])

# Simulate a network with 4 layers (using the same W and b for all layers for simplicity)
W1 = W
b1 = b
W2 = W
b2 = b
W3 = W
b3 = b
W4 = W
b4 = b

# Example with identity activation
output_identity = sequential(a_in, W1, b1, W2, b2, W3, b3, W4, b4)
print("Output with identity activation:", output_identity)

# Example with sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

output_sigmoid = sequential(a_in, W1, b1, W2, b2, W3, b3, W4, b4, g=sigmoid)
print("Output with sigmoid activation:", output_sigmoid)