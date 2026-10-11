import numpy as np

# TODO: Set the number of classes for EMNIST Balanced (47 classes)
NUM_CLASSES = 47

def init_params():
    # TODO: Initialize W1 (shape 256, 784) and b1 (shape 256, 1) using He Initialization
    # TODO: Initialize W2 (shape NUM_CLASSES, 256) and b2 (shape NUM_CLASSES, 1) using He Initialization
    # TODO: Return W1, b1, W2, b2
    pass

def relu(Z):
    # TODO: Implement ReLU activation function: max(0, Z)
    pass

def relu_deriv(Z):
    # TODO: Implement ReLU derivative: return 1 if Z > 0, else 0
    pass

def softmax(Z):
    # TODO: Implement Softmax function. 
    # Note: subtract np.max(Z, axis=0) before np.exp to avoid numerical overflow (numerical stability)
    pass

def forward_prop(W1, b1, W2, b2, X):
    # TODO: Calculate Z1 = W1.dot(X) + b1
    # TODO: Calculate A1 = relu(Z1)
    # TODO: Calculate Z2 = W2.dot(A1) + b2
    # TODO: Calculate A2 = softmax(Z2)
    # TODO: Return Z1, A1, Z2, A2
    pass

def one_hot(Y):
    # TODO: Create a one-hot matrix with NUM_CLASSES columns.
    # TODO: Set value 1 at the correct index for each label. Return the transposed matrix.
    pass

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
    pass