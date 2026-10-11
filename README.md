# EMNIST Handwritten Character Recognition — Pure NumPy

An **Introduction to Artificial Intelligence** coursework project that implements a **Multi-Layer Perceptron (MLP)** neural network entirely from scratch using **NumPy**.

The project performs handwritten character recognition using the **EMNIST Balanced** dataset, supporting **47 character classes**, including digits and English letters.

> **No Deep Learning frameworks are used.**  
> The neural network, forward propagation, backpropagation, gradient descent, and activation functions are implemented using mathematical operations with NumPy.

---

## Key Features

- **Pure Mathematics**
  - Matrix-based forward propagation
  - Backpropagation
  - Gradient descent
  - ReLU activation
  - Softmax output layer
  - Cross-entropy loss

- **Memory Optimized**
  - Reads EMNIST data directly from compressed `.gz` files
  - Processes data using mini-batches instead of loading the entire dataset into memory

- **47 Character Classes**
  - Based on the EMNIST Balanced dataset
  - Supports digits and English alphabet characters

- **Command-Line Interface**
  - Easily switch between training and inference modes
  - Simple commands for running the project

- **Real-World Inference**
  - Predict characters from user-provided `.png` images
  - Supports images drawn using tools such as Microsoft Paint or Excalidraw

---

## Project Structure

```text
emnist_project/
├── dataset/
│   ├── emnist-balanced-train-images-idx3-ubyte.gz
│   ├── emnist-balanced-train-labels-idx1-ubyte.gz
│   └── emnist-balanced-mapping.txt
│
├── saved_models/
│   └── mlp_weights.npz
│
├── src/
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── model.py            # MLP architecture and mathematical logic
│   ├── train.py            # Training loop, loss, and accuracy
│   └── predict.py          # Inference on user-provided images
│
├── main.py                 # Main CLI entry point
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/emnist_project.git
cd emnist_project
```

### 2. Install Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

This project uses the **EMNIST Balanced** dataset.

Download the dataset in `.gz` format from the official NIST repository and place the required files inside the `dataset/` directory.

The following files are required:

```text
dataset/
├── emnist-balanced-train-images-idx3-ubyte.gz
├── emnist-balanced-train-labels-idx1-ubyte.gz
└── emnist-balanced-mapping.txt
```

> **Note:** The dataset files are not included in this repository because of their size. They should also be excluded from Git using `.gitignore`.

---

## Usage

### 1. Train the Model

Start the training process with:

```bash
python main.py --mode train
```

During training, the program will periodically display the current loss and accuracy.

Once training is completed, the trained model weights will automatically be saved to:

```text
saved_models/mlp_weights.npz
```

---

### 2. Test the Model

To perform inference on a handwritten character, first create an image containing a single character.

The recommended format is:

- **PNG** image
- White character
- Black background
- One character per image

For example:

```text
       ███
      █   █
      █   █
      █████
      █   █
      █   █
```

Save the image as a `.png` file and run:

```bash
python main.py --mode test --image path/to/your_image.png
```

Example:

```bash
python main.py --mode test --image examples/A.png
```

The model will preprocess the image and output its predicted character.

---

## Model Architecture

The project uses a **Multi-Layer Perceptron (MLP)** implemented entirely with NumPy.

The general pipeline is:

```text
EMNIST Image
     │
     ▼
Image Preprocessing
     │
     ▼
Flatten 28 × 28 → 784
     │
     ▼
Fully Connected Layer
     │
     ▼
ReLU Activation
     │
     ▼
Fully Connected Layer
     │
     ▼
Softmax
     │
     ▼
47 Character Classes
```

### Forward Propagation

The network computes the output using matrix operations:

```text
Z = XW + b
A = ReLU(Z)
```

The final layer applies Softmax to obtain class probabilities:

```text
P(y = i | x) = exp(zᵢ) / Σ exp(zⱼ)
```

### Backpropagation

The model calculates gradients using the chain rule and updates its parameters using gradient descent:

```text
W ← W - η ∂L/∂W
b ← b - η ∂L/∂b
```

where:

- `W` — weight matrix
- `b` — bias vector
- `L` — loss function
- `η` — learning rate

---

## Dataset

The project uses the **EMNIST Balanced** dataset.

Each image is:

```text
28 × 28 pixels
```

After flattening:

```text
28 × 28 = 784 input features
```

The output layer contains:

```text
47 classes
```

Each input pixel is normalized before being passed to the neural network.

---

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Matrix operations and neural network implementation |
| EMNIST Balanced | Handwritten character dataset |
| Pillow | Image preprocessing and inference |
| Git | Version control |

---

## Development Team

| Member | Responsibility |
|---|---|
| **Member 1** | Data Pipeline & Preprocessing |
| **Member 2** | Model Architecture & Algorithms |
| **Member 3** | Training Loop & Optimization |
| **Member 4** | Integration, CLI & Inference |

---

## Project Goals

This project aims to demonstrate the fundamental mathematical principles behind neural networks without relying on high-level Deep Learning frameworks.

The main concepts covered include:

- Data preprocessing
- Feature normalization
- Matrix multiplication
- Forward propagation
- Activation functions
- Softmax classification
- Cross-entropy loss
- Backpropagation
- Gradient descent
- Mini-batch training
- Model serialization
- Image inference

---

## License

This project was developed for educational purposes as part of an **Introduction to Artificial Intelligence** coursework project.