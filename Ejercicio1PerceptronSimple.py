import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.lr = lr

    def sigmoid(self, x):
        with np.errstate(over='ignore'):
            return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights) + self.bias)

    def train(self, inputs, labels, max_epochs):
        errors = []
        for epoch in range(max_epochs):
            total_error = 0
            for input_row, label in zip(inputs, labels):
                prediction = self.predict(input_row)
                error = label - prediction
                self.weights += self.lr * error * input_row
                self.bias += self.lr * error
                total_error += abs(error)
            errors.append(total_error)
            if total_error == 0:
                print(f"Converged at epoch {epoch}")
                break
        else:
            print("No convergence.")
        return errors

# Function to select a file using tkinter
def select_file(message):
    Tk().withdraw()
    print(message)
    filename = filedialog.askopenfilename()
    return filename

# Request the user to select the training file
train_file = select_file("Please select the training file")

# Request the user to select the test file
test_file = select_file("Please select the test file")

# Load training and test data from CSV files
train_data = np.loadtxt(train_file, delimiter=',')
test_data = np.loadtxt(test_file, delimiter=',')

X_train = train_data[:, :2]
y_train = train_data[:, 2]
X_test = test_data[:, :2]
y_test = test_data[:, 2]

# Add a column of ones to inputs to represent the bias term
X_train = np.c_[X_train, np.ones(X_train.shape[0])]
X_test = np.c_[X_test, np.ones(X_test.shape[0])]

# Create and train the perceptron
perceptron = Perceptron(input_size=X_train.shape[1])
max_epochs = 1000
learning_rate = 0.1
errors = perceptron.train(X_train, y_train, max_epochs)

# Make predictions on test data
predictions = []
for input_row in X_test:
    prediction = perceptron.predict(input_row)
    predictions.append(prediction)

# Compare predictions with true labels
correct_predictions = (np.array(predictions) >= 0.5).astype(int)
accuracy = np.mean(correct_predictions == y_test)
print(f"Accuracy: {accuracy}")

# Create a mesh of points for the separation surface
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel(), np.ones(len(xx.ravel()))])
Z = Z.reshape(xx.shape)

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the separation surface
ax.plot_surface(xx, yy, Z, alpha=0.5)

# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c=y_train, cmap='viridis', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Label')

plt.title('Simple Perceptron')
plt.show()
