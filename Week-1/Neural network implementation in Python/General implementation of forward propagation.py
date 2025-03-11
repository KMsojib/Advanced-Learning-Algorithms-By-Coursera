import numpy as np

def dense(a_in, W, b, g=lambda x: x):  # Identity activation by default
    
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out

def sequential(x, W1, b1, W2, b2, W3, b3, W4, b4, g=lambda x: x):
    a1 = dense(x, W1, b1, g)
    a2 = dense(a1, W2, b2, g)
    a3 = dense(a2, W3, b3, g)
    a4 = dense(a3, W4, b4, g)
    return a4


W = np.array([[1, -3, 5], [2, 4, -6]])
b = np.array([-1, 1, 2])
a_in = np.array([-2, 4])

W1 = W
b1 = b
W2 = W
b2 = b
W3 = W
b3 = b
W4 = W
b4 = b


output_identity = sequential(a_in, W1, b1, W2, b2, W3, b3, W4, b4)
print("Output with identity activation:", output_identity)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

output_sigmoid = sequential(a_in, W1, b1, W2, b2, W3, b3, W4, b4, g=sigmoid)
print("Output with sigmoid activation:", output_sigmoid)