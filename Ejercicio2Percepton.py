# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tkinter import Tk, filedialog

# Definir la clase Perceptrón
class Perceptron:
    def __init__(self, input_size, lr=0.1):
        # Inicializar los pesos y sesgo aleatoriamente
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.lr = lr

    def sigmoid(self, x):
        # Función de activación sigmoide
        with np.errstate(over='ignore'):
            return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        # Predicción utilizando la función de activación sigmoide
        return self.sigmoid(np.dot(inputs, self.weights) + self.bias)

    def train(self, inputs, labels, max_epochs):
        # Entrenamiento del perceptrón
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
            # Verificar si ha convergido
            if total_error == 0:
                print(f"Converged at epoch {epoch}")
                break
        else:
            print("No convergence.")
        return errors

# Función para seleccionar un archivo usando tkinter
def select_file(message):
    Tk().withdraw()
    print(message)
    filename = filedialog.askopenfilename()
    return filename

# Solicitar al usuario que seleccione el archivo de datos
file_path = select_file("Por favor, seleccione el archivo de datos")

# Cargar datos desde el archivo CSV
data = np.loadtxt(file_path, delimiter=',')

# Dividir los datos en conjuntos de entrenamiento y prueba
np.random.shuffle(data)
train_size = int(len(data) * 0.8)
train_data = data[:train_size]

# Dividir los datos de entrenamiento en diez particiones
X_train_partitions = []
y_train_partitions = []
partition_size = train_size // 10
for i in range(10):
    start_index = i * partition_size
    end_index = (i + 1) * partition_size if i < 9 else train_size
    X_train_partition = train_data[start_index:end_index, :-1]
    y_train_partition = train_data[start_index:end_index, -1]
    X_train_partitions.append(X_train_partition)
    y_train_partitions.append(y_train_partition)

# Crear y entrenar el perceptrón para cada partición
accuracies = []
for i in range(10):
    X_train = np.concatenate([X_train_partitions[j] for j in range(10) if j != i])
    y_train = np.concatenate([y_train_partitions[j] for j in range(10) if j != i])
    X_val = X_train_partitions[i]
    y_val = y_train_partitions[i]
    perceptron = Perceptron(input_size=X_train.shape[1], lr=0.1)
    max_epochs = 1000
    learning_rate = 0.1
    errors = perceptron.train(X_train, y_train, max_epochs)
    predictions = [perceptron.predict(input_row) for input_row in X_val]
    correct_predictions = (np.array(predictions) >= 1.0).astype(int)
    accuracy = np.mean(correct_predictions == y_val)
    accuracies.append(accuracy)

# Calcular y mostrar la precisión promedio
average_accuracy = np.mean(accuracies)
print(f"Precisión promedio: {average_accuracy}")

# Crear la figura y el eje 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
x = np.arange(10)
y = np.arange(10)
x, y = np.meshgrid(x, y)
z = np.array(accuracies).reshape((10, 1))

# Graficar la superficie de separación
ax.plot_surface(x, y, z, alpha=0.5)

# Graficar los puntos de entrenamiento
for i in range(10):
    ax.scatter(X_train_partitions[i][:, 0], X_train_partitions[i][:, 1], y_train_partitions[i], c=y_train_partitions[i], cmap='viridis', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Label')
plt.title('Perceptrón Simple')
plt.show()
