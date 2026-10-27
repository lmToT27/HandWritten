import numpy as np
# TODO: Import forward_prop, get_predictions from src.model
import cv2
from src.model import forward_prop
from src.train import get_predictions
# TODO: Import load_mapping from src.data_loader
from src.data_loader import load_mapping

def preprocess_image(image_path):
    """
    Preprocess user-drawn image to feed into the model.
    Might require OpenCV (cv2) or PIL.
    """
    # TODO: Read image from image_path
    # TODO: Convert image to Grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

    # TODO: Invert colors if necessary (ensure white text on black background like EMNIST)
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # TODO: Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # TODO: Flatten image into a vector of shape (784, 1) and divide by 255.0
    X = img.reshape(784, 1) / 255.0
    # TODO: Return this vector
    return X

def predict_image(image_path):
    # TODO: Use np.load() to read the file "saved_models/mlp_weights.npz"
    data = np.load("saved_models/mlp_weights.npz")
    # TODO: Extract W1, b1, W2, b2 from the loaded file
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]

    # TODO: Call preprocess_image(image_path) to get the input vector X
    X = preprocess_image(image_path)
    # TODO: Call forward_prop(W1, b1, W2, b2, X) to get A2
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    # TODO: Call get_predictions(A2) to get the predicted class_index
    class_index = get_predictions(A2)[0]

    # TODO: Call load_mapping() to get the dictionary
    mapping = load_mapping("dataset/emnist-balanced-mapping.txt")
    # TODO: Get the actual character from class_index using the mapping dictionary
    character = mapping.get(int(class_index), "?")
    # TODO: Print the prediction result to the console (e.g., "Model predicted: H")
    print(f"Model predicted: '{character}'  (class index: {class_index})")