import numpy as np
import gzip

def load_emnist_images(filename):
    """
    Read .gz compressed file containing EMNIST images.
    """
    # TODO: Use gzip.open to read file in 'rb' mode
    # TODO: Use np.frombuffer to read data, remember to set offset=16 to skip the ubyte image file header
    # TODO: Reshape the read 1D array into a 3D array of shape (-1, 28, 28)
    # TODO: Rotate image 90 degrees using np.rot90(data, k=1, axes=(2, 1))
    # TODO: Flip image using np.flip(data, axis=2)
    # TODO: Reshape again to (-1, 784)
    # TODO: Return the transposed matrix (.T) and divide by 255.0 for normalization. Expected shape: (784, m)
    pass

def load_emnist_labels(filename):
    """
    Read .gz compressed file containing labels.
    """
    # TODO: Use gzip.open to read file in 'rb' mode
    # TODO: Use np.frombuffer to read data, set offset=8 to skip the label file header
    # TODO: Return 1D label array. Expected shape: (m,)
    pass

def load_mapping(filename):
    """
    Read emnist-balanced-mapping.txt to map from class index to ASCII character.
    """
    mapping = {}
    # TODO: Open mapping.txt file
    # TODO: Read line by line, extract column 1 (index) and column 2 (ASCII code)
    # TODO: Use chr(int(ascii_code)) to convert the numeric code to a character
    # TODO: Save to `mapping` dictionary and return it
    return mapping