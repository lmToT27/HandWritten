import numpy as np
# TODO: Import forward_prop, get_predictions from src.model
# TODO: Import load_mapping from src.data_loader

def preprocess_image(image_path):
    """
    Preprocess user-drawn image to feed into the model.
    Might require OpenCV (cv2) or PIL.
    """
    # TODO: Read image from image_path
    # TODO: Convert image to Grayscale
    # TODO: Invert colors if necessary (ensure white text on black background like EMNIST)
    # TODO: Resize to 28x28
    # TODO: Flatten image into a vector of shape (784, 1) and divide by 255.0
    # TODO: Return this vector
    pass

def predict_image(image_path):
    # TODO: Use np.load() to read the file "saved_models/mlp_weights.npz"
    # TODO: Extract W1, b1, W2, b2 from the loaded file
    
    # TODO: Call preprocess_image(image_path) to get the input vector X
    # TODO: Call forward_prop(W1, b1, W2, b2, X) to get A2
    # TODO: Call get_predictions(A2) to get the predicted class_index
    
    # TODO: Call load_mapping() to get the dictionary
    # TODO: Get the actual character from class_index using the mapping dictionary
    # TODO: Print the prediction result to the console (e.g., "Model predicted: H")
    pass