import numpy as np
# TODO: Import functions from src.model (init_params, forward_prop, backward_prop)
# TODO: Import data loading functions from src.data_loader

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # TODO: Update weights using Gradient Descent formula: W = W - alpha * dW
    # TODO: Return updated W1, b1, W2, b2
    pass

def get_predictions(A2):
    # TODO: Use np.argmax to get the class with the highest probability at the output layer (axis=0)
    pass

def get_accuracy(predictions, Y):
    # TODO: Calculate the percentage of correct predictions compared to the actual Y
    pass

def train_model():
    # TODO: Call function to load images and labels (X_train, Y_train)
    # TODO: Define alpha (learning rate) and number of epochs (iterations)
    
    # W1, b1, W2, b2 = init_params()
    
    # TODO: Write a for loop from 0 to iterations:
    #       - Call forward_prop
    #       - Call backward_prop
    #       - Call update_params
    #       - Every 50 epochs, print out the accuracy using get_accuracy()
    
    # TODO: After the loop finishes, use np.savez("saved_models/mlp_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    # TODO: Print a success message confirming the model has been saved
    pass