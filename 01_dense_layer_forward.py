import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Initialize NNFS (sets default random seed and float32 dtype)
nnfs.init()

# Dense Layer class
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with small random numbers
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases to zeros
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Dot product of inputs and weights + bias
        self.output = np.dot(inputs, self.weights) + self.biases

# Create a dataset with 100 samples per class, 3 classes → total 300 samples
X, y = spiral_data(samples=100, classes=3)

# Create a Dense Layer: 2 inputs → 3 neurons
dense1 = LayerDense(n_inputs=2, n_neurons=3)

# Forward pass of input X through dense layer
dense1.forward(X)

# Print shapes for understanding
print("Input X shape:", X.shape)            # Should be (300, 2)
print("Weights shape:", dense1.weights.shape)  # Should be (2, 3)
print("Biases shape:", dense1.biases.shape)    # Should be (1, 3)
print("Output shape:", dense1.output.shape)    # Should be (300, 3)

# Optionally print first few outputs
print("\nFirst 5 outputs:")
print(dense1.output[:5])
