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
class SVMGD:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Convert y values to 1 and -1
        y_ = np.where(y <= 0, -1, 1)

        # 1. Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # 2. Gradient Descent
        for _ in range(self.n_iterations):
            for idx, xi in enumerate(X):
                condition = y_[idx] * (np.dot(xi, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(xi, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)

# Training the SVM model using Gradient Descent
svm_gd = SVMGD(learning_rate=0.0001, lambda_param=0.0005, n_iterations=10000)
svm_gd.fit(X_train.values, Y_train.values.ravel())  # Ensure numpy array input

# Predicting on the test set
svm_gd_predictions = svm_gd.predict(X_test.values)

# Convert predictions from -1,1 to 0,1
svm_gd_predictions = np.where(svm_gd_predictions == -1, 0, 1)

# Calculating the accuracy and other metrics
svm_gd_accuracy = accuracy_score(Y_test, svm_gd_predictions)
svm_gd_classification_report = classification_report(Y_test, svm_gd_predictions)
svm_gd_confusion_matrix = confusion_matrix(Y_test, svm_gd_predictions)

print(svm_gd_accuracy, svm_gd_classification_report, svm_gd_confusion_matrix)
