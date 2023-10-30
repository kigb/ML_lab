import os
dataset_dir = "./data/lab1_dataset"
import pandas as pd
X_train = pd.read_csv(os.path.join(dataset_dir, 'X_train.csv'))
Y_train = pd.read_csv(os.path.join(dataset_dir, 'Y_train.csv'))
X_test = pd.read_csv(os.path.join(dataset_dir, 'X_test.csv'))
Y_test = pd.read_csv(os.path.join(dataset_dir, 'Y_test.csv'))
#读取数据
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize the model
input_size = X_train.shape[1]
hidden_size1 = 16
hidden_size2 = 8
output_size = 1  # Binary classification
model = MLP(input_size, hidden_size1, hidden_size2, output_size)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
Y_train_tensor = torch.FloatTensor(Y_train.values)
X_test_tensor = torch.FloatTensor(X_test.values)

# Training loop
num_epochs = 1100
for epoch in range(num_epochs):
    # Forward pass and loss
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training finished.")

# Test the model
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_predictions = (test_outputs > 0.5).float()

# Calculate accuracy
mlp_accuracy = accuracy_score(Y_test, test_predictions.numpy())
mlp_classification_report = classification_report(Y_test, test_predictions.numpy())
mlp_confusion_matrix = confusion_matrix(Y_test, test_predictions.numpy())

print(mlp_accuracy, mlp_classification_report, mlp_confusion_matrix)
