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
class LogisticRegressionGD:
    def __init__(self, learning_rate=0.005, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 1. Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Gradient Descent
        for _ in range(self.n_iterations):
            model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(model)
        return [1 if i > 0.5 else 0 for i in predictions]

# Training the Logistic Regression model using Gradient Descent
lr_gd = LogisticRegressionGD(learning_rate=0.001, n_iterations=5000)
lr_gd.fit(X_train, Y_train.values.ravel())

# Predicting on the test set
lr_gd_predictions = lr_gd.predict(X_test)

# Calculating the accuracy and other metrics
lr_gd_accuracy = accuracy_score(Y_test, lr_gd_predictions)
lr_gd_classification_report = classification_report(Y_test, lr_gd_predictions)
lr_gd_confusion_matrix = confusion_matrix(Y_test, lr_gd_predictions)

print(lr_gd_accuracy, lr_gd_classification_report, lr_gd_confusion_matrix)
