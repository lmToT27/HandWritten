import numpy as np

# TODO: Set the number of classes for EMNIST Balanced (47 classes)
NUM_CLASSES = 47

def init_params():
    # TODO: Initialize W1 (shape 256, 784) and b1 (shape 256, 1) using He Initialization
    W1 = np.random.randn(256, 784) * np.sqrt(2. / 784)
    b1 = np.zeros((256, 1))
    # TODO: Initialize W2 (shape NUM_CLASSES, 256) and b2 (shape NUM_CLASSES, 1) using He Initialization
    W2 = np.random.radn(NUM_CLASSES, 256) * np.sqrt(2. / 256)
    b2 = np.zeros((NUM_CLASSES, 1))

    return W1, b1, W2, b2

def relu(Z):
    # TODO: Implement ReLU activation function: max(0, Z)
    return np.maximum(0, Z)

def relu_deriv(Z):  
    # TODO: Implement ReLU derivative: return 1 if Z > 0, else 0
    return Z > 0

def softmax(Z):
    # TODO: Implement Softmax function. 
    # Note: subtract np.max(Z, axis=0) before np.exp to avoid numerical overflow (numerical stability)
    shift_Z = Z - np.max(Z, axis = 0)
    exp_Z = np.exp(shift_Z)
    return exp_Z / np.sum(exp_Z, axis = 0)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    # TODO: Create a one-hot matrix with NUM_CLASSES columns.
    # TODO: Set value 1 at the correct index for each label. Return the transposed matrix.
    m = len(Y)
    res = np.zeros((m, NUM_CLASSES))
    res[np.arange(m), Y] = 1
    return res.T

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
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) *  dZ2 @ A1.T
    db2 = (1 / m) * np.sum(Z2, axis = 1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * relu_deriv(Z1)
    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * np.sum(Z1, axis = 1, keepdims=True)
    return dW1, db1, dW2, db2
