import numpy as np
# TODO: Import functions from src.model (init_params, forward_prop, backward_prop)
from src.model import init_params, forward_prop, backward_prop
# TODO: Import data loading functions from src.data_loader
from src.data_loader import load_emnist_images, load_emnist_labels

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # TODO: Update weights using Gradient Descent formula: W = W - alpha * dW
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    # TODO: Return updated W1, b1, W2, b2
    return W1, b1, W2, b2

def get_predictions(A2):
    # TODO: Use np.argmax to get the class with the highest probability at the output layer (axis=0)
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    # TODO: Calculate the percentage of correct predictions compared to the actual Y
    return np.mean(predictions == Y)

def train_model():
    # TODO: Call function to load images and labels (X_train, Y_train)
    print("Loading dataset...")
    X_train = load_emnist_images("dataset/emnist-balanced-train-images-idx3-ubyte.gz")
    Y_train = load_emnist_labels("dataset/emnist-balanced-train-labels-idx1-ubyte.gz")
    print(f"  Dataset loaded: {X_train.shape[1]} samples, {X_train.shape[0]} features")

    # TODO: Define alpha (learning rate) and number of epochs (iterations)
    alpha = 0.1       # learning rate
    iterations = 500  # number of epochs
    batch_size = 1024 # mini-batch size
    m = X_train.shape[1]

    W1, b1, W2, b2 = init_params()
    print(f"\nStarting training for {iterations} epochs (batch_size={batch_size}, lr={alpha})...")
    print("-" * 55)

    # TODO: Write a for loop from 0 to iterations:
    for epoch in range(iterations):
        # Shuffle dataset at the start of each epoch
        perm = np.random.permutation(m)
        X_shuffled = X_train[:, perm]
        Y_shuffled = Y_train[perm]

        # Mini-batch training
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[start:end]

            #       - Call forward_prop
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_batch)
            #       - Call backward_prop
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X_batch, Y_batch)
            #       - Call update_params
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        #       - Every 50 epochs, print out the accuracy using get_accuracy()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            _, _, _, A2_full = forward_prop(W1, b1, W2, b2, X_train)
            predictions = get_predictions(A2_full)
            acc = get_accuracy(predictions, Y_train)
            print(f"  Epoch {epoch + 1:>4}/{iterations}  |  Accuracy: {acc * 100:.2f}%")

    # TODO: After the loop finishes, use np.savez("saved_models/mlp_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    import os
    os.makedirs("saved_models", exist_ok=True)
    np.savez("saved_models/mlp_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    # TODO: Print a success message confirming the model has been saved
    print("-" * 55)
    print("Model saved to: saved_models/mlp_weights.npz")