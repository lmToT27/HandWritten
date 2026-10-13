import numpy as np

# TODO: Set the number of classes for EMNIST Balanced (47 classes)
NUM_CLASSES = 47

def init_params():
    # TODO: Initialize W1 (shape 256, 784) and b1 (shape 256, 1) using He Initialization
    # TODO: Initialize W2 (shape NUM_CLASSES, 256) and b2 (shape NUM_CLASSES, 1) using He Initialization
    # TODO: Return W1, b1, W2, b2

    # make weight for layer 1
    W1 = np.random.randn(784, 256)
    sigma1 = np.sqrt(2 / 784)
    W1 = W1 * sigma1
    # make bias for layer 1
    b1 = np.zeros((1, 256))

    # make weight for layer 2
    W2 = np.random.randn(256, NUM_CLASSES)
    sigma2 = np.sqrt(2 / 256)
    W2 = W2 * sigma2
    #make bias for layer 2
    b2 = np.zeros((1, NUM_CLASSES))

    return W1, b1, W2, b2

def relu(Z):
    # TODO: Implement ReLU activation function: max(0, Z)
    return np.maximum(0, Z)

def relu_deriv(Z):
    # TODO: Implement ReLU derivative: return 1 if Z > 0, else 0
    return np.where(Z > 0, 1, 0)

def softmax(Z):
    # TODO: Implement Softmax function. 
    # Note: subtract np.max(Z, axis=0) before np.exp to avoid numerical overflow (numerical stability)
    
    #subtract to max element
    Z = Z - np.max(Z, axis=1, keepdims=True)
    E = np.exp(Z)
    row_sum = np.sum(E, axis=1, keepdims=True)
    sm = E / row_sum
    return sm

def forward_prop(W1, b1, W2, b2, X):
    # TODO: Calculate Z1 = W1.dot(X) + b1
    # TODO: Calculate A1 = relu(Z1)
    # TODO: Calculate Z2 = W2.dot(A1) + b2
    # TODO: Calculate A2 = softmax(Z2)
    # TODO: Return Z1, A1, Z2, A2
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(labels):
    # TODO: Create a one-hot matrix with NUM_CLASSES columns.
    # TODO: Set value 1 at the correct index for each label. Return the transposed matrix.
    Y = np.zeros(len(labels), NUM_CLASSES)
    for i in range(len(labels)):
        Y[i, labels[i]] = 1
    return Y.T

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # TODO: Get number of samples m = Y.size
    # TODO: Create one_hot_Y using one_hot(Y)
    # TODO: Calculate dZ2 = A2 - one_hot_Y
    # TODO: Calculate dW2 = (1/m) * dZ2.dot(A1.T)
    # TODO: Calculate db2 = (1/m) * sum of columns in dZ2
    # TODO: Calculate dZ1 = W2.T.dot(dZ2) * relu_deriv(Z1)
    # TODO: Calculate dW1 = (1/m) * dZ1.dot(X.T)
    # TODO: Calculate db1 = (1/m) * sum of columns in dZ1
    # TODO: Return dW1, db1, dW2, db2
    m = len(Y)
    one_hot_Y = one_hot(Y)
    dZ2 = (1/m) * (A2 - one_hot_Y)
    dW2 = (A1.T) @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = (1/m) * dZ2 @ W2.T * relu_deriv(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2