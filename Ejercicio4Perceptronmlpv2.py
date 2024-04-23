import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

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

# Request the user to select the training file
train_file = select_file("Please select the training file")

# Request the user to select the test file
test_file = select_file("Please select the test file")

# Load training and test data from CSV files
train_data = np.loadtxt(train_file, delimiter=',')
test_data = np.loadtxt(test_file, delimiter=',')

X_train = train_data[:, :3]  # Ajustar el número de características según tus datos
y_train = train_data[:, 3]   # Columna que contiene las etiquetas
X_test = test_data[:, :3]    # Ajustar el número de características según tus datos
y_test = test_data[:, 3]     # Columna que contiene las etiquetas

# Crear y entrenar el perceptrón
mlp = MultiLayerPerceptron(input_size=X_train.shape[1], hidden_size=3, output_size=3)  # Ajustar el tamaño de salida
max_epochs = 1000
learning_rate = 0.1
errors = mlp.train(X_train, y_train, max_epochs)

# Hacer predicciones en los datos de prueba
predictions = []
for input_row in X_test:
    prediction = mlp.predict(input_row)
    predictions.append(prediction)

# Comparar predicciones con las etiquetas reales
correct_predictions = np.argmax(predictions, axis=1)  # Obtener el índice de la salida más alta como predicción
accuracy = np.mean(correct_predictions == y_test)
print(f"Accuracy: {accuracy}")

# Graficar los errores durante el entrenamiento
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Total Error')
plt.title('Training Error')
plt.show()

# Graficar el resultado de la clasificación
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=correct_predictions, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Classification Result')
plt.colorbar(label='Predicted Label')
plt.show()
