import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from sklearn.model_selection import KFold

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(output_size)
        self.lr = lr

    def sigmoid(self, x):
        with np.errstate(over='ignore'):
            return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        return self.sigmoid(output_layer_input)

    def train(self, inputs, labels, max_epochs):
        errors = []
        for epoch in range(max_epochs):
            total_error = 0
            for input_row, label in zip(inputs, labels):
                hidden_layer_input = np.dot(input_row, self.weights_input_hidden) + self.bias_hidden
                hidden_layer_output = self.sigmoid(hidden_layer_input)
                output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
                predicted_label = self.sigmoid(output_layer_input)

                error = label - predicted_label
                total_error += np.mean(np.abs(error))

                output_error = error * self.sigmoid_derivative(predicted_label)
                hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.sigmoid_derivative(hidden_layer_output)

                self.weights_hidden_output += self.lr * np.outer(hidden_layer_output, output_error)
                self.bias_output += self.lr * output_error
                self.weights_input_hidden += self.lr * np.outer(input_row, hidden_error)
                self.bias_hidden += self.lr * hidden_error

            errors.append(total_error)

            if total_error < 0.01:
                print(f"Converged at epoch {epoch}")
                break
        else:
            print("No convergence.")
        return errors

    def sigmoid_derivative(self, x):
        return x * (1 - x)

# Function to select a file using tkinter
def select_file(message):
    Tk().withdraw()
    print(message)
    filename = filedialog.askopenfilename()
    return filename

# Request the user to select the data file
data_file = select_file("Please select the data file")

# Load data from the CSV file
data = np.loadtxt(data_file, delimiter=',')

X = data[:, :3]  # Adjust the number of features according to your data
y = data[:, 3]   # Column containing the labels

# Initialize MultiLayerPerceptron
mlp = MultiLayerPerceptron(input_size=X.shape[1], hidden_size=3, output_size=3)  # Adjust the output size based on your problem

# Leave-k-out Cross Validation
k = 5  # Number of folds for leave-k-out
kf = KFold(n_splits=k)
k_errors = []
for train_index, test_index in kf.split(X):
    X_train_k, X_test_k = X[train_index], X[test_index]
    y_train_k, y_test_k = y[train_index], y[test_index]
    errors_k = mlp.train(X_train_k, y_train_k, max_epochs=1000)
    k_errors.append(errors_k[-1])  # Save the error of the last epoch

# Leave-one-out Cross Validation (LOOCV)
loocv_errors = []
for i in range(len(X)):
    X_train_loocv = np.delete(X, i, axis=0)
    y_train_loocv = np.delete(y, i, axis=0)
    errors_loocv = mlp.train(X_train_loocv, y_train_loocv, max_epochs=1000)
    loocv_errors.append(errors_loocv[-1])  # Save the error of the last epoch

# Calculate expected error, average, and standard deviation
expected_error_k = np.mean(k_errors)
expected_error_loocv = np.mean(loocv_errors)
average_error_k = np.mean(k_errors)
average_error_loocv = np.mean(loocv_errors)
std_deviation_k = np.std(k_errors)
std_deviation_loocv = np.std(loocv_errors)

# Print results
print("Leave-k-out Cross Validation:")
print(f"Expected Error: {expected_error_k}")
print(f"Average Error: {average_error_k}")
print(f"Standard Deviation: {std_deviation_k}")

print("\nLeave-one-out Cross Validation (LOOCV):")
print(f"Expected Error: {expected_error_loocv}")
print(f"Average Error: {average_error_loocv}")
print(f"Standard Deviation: {std_deviation_loocv}")
