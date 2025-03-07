import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# # Creating sample data
# features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# labels = torch.tensor([0, 1, 0])

# # Creating a TensorDataset
# dataset = TensorDataset(features, labels)
# print("Dataset sample:", dataset[1])  # Output: (tensor([1., 2.]), tensor(0))
# print(f"Dataset length: {len(dataset)}")
# print(f"Dataset features shape: {dataset[0][0].shape}")
# print(f"Dataset labels shape: {dataset[0][1].shape}")

# a = torch.tensor(np.array([[[1, 2], [3, 4], [5, 6]],[[7, 8], [9, 10], [11, 12]]]))
# print(a.shape)
import torch

# Mock data for testing
batch_size = 32
num_classes = 4

# Generate predictions (logits) from the model that perfectly match the labels
# For simplicity, we'll use one-hot encoded vectors for predictions
predictions = torch.eye(num_classes)[torch.randint(0, num_classes, (batch_size,))]

# Apply softmax to get probabilities (though it's not necessary for one-hot vectors)
predictions = torch.softmax(predictions, dim=1)

# Generate labels that perfectly match the predictions
labels = torch.argmax(predictions, dim=1)

# Print the mock data
print("Predictions (probabilities):")
print(predictions)
print("Labels (true class indices):")
print(labels)

# Define the calculate_accuracy function
def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

# Calculate and print the accuracy
accuracy = calculate_accuracy(predictions, labels)
print(f'Accuracy: {accuracy:.4f}')