# Readme.md

Este código en Python implementa un modelo de clasificación utilizando el algoritmo K-Vecinos Cercanos (K-Nearest Neighbors, KNN) para un conjunto de datos de características de animales. A continuación, se explica el código por partes de una manera fácil de entender:

## Importación de librerías

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import tkinter as tk
from tkinter import filedialog
```

Aquí se importan las librerías necesarias para la implementación del modelo. `pandas` se utiliza para la manipulación y análisis de datos, `sklearn` proporciona herramientas para el procesamiento de datos y la construcción de modelos de aprendizaje automático, `joblib` se utiliza para guardar y cargar el modelo entrenado, y `tkinter` se utiliza para crear una interfaz gráfica de usuario (GUI) para seleccionar el archivo CSV.

## Función `load_and_train_model()`

```python
def load_and_train_model():
    # Abrir un cuadro de diálogo para seleccionar el archivo CSV
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        print("No file selected.")
        return
```

Esta función se encarga de cargar los datos del archivo CSV y entrenar el modelo KNN. Primero, se crea una ventana de Tkinter para abrir un cuadro de diálogo y seleccionar el archivo CSV. Si no se selecciona ningún archivo, se muestra un mensaje y se sale de la función.

```python
    # Cargar el dataset
    df = pd.read_csv(file_path)

    # Eliminar la columna animal_name
    df = df.drop(columns=['animal_name'])

    # Asumimos que la última columna es la etiqueta
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
```

Luego, se carga el conjunto de datos desde el archivo CSV utilizando `pd.read_csv`. Se elimina la columna `'animal_name'` del conjunto de datos, ya que no se utilizará para el entrenamiento del modelo. Se asume que la última columna del conjunto de datos contiene las etiquetas (`y`), y el resto de las columnas contienen las características (`X`).

```python
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar las características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
```

El conjunto de datos se divide en conjuntos de entrenamiento y prueba utilizando `train_test_split` de `sklearn`. El parámetro `test_size=0.2` indica que el 20% de los datos se utilizarán para pruebas, y el `random_state=42` garantiza que la división sea reproducible.

Luego, se realiza un escalamiento o normalización de las características utilizando `StandardScaler` de `sklearn`. Esto ayuda a que el algoritmo KNN funcione mejor, ya que las características se escalan a una misma escala.

```python
    # Crear y entrenar el modelo K-Vecinos Cercanos
    model = KNeighborsClassifier(n_neighbors=7)  # Usando k=5 como ejemplo
    model.fit(X_train, y_train)
```

Se crea una instancia del clasificador KNN de `sklearn` con `n_neighbors=7`, lo que significa que se utilizarán los 7 vecinos más cercanos para hacer las predicciones. Luego, se entrena el modelo utilizando los datos de entrenamiento (`X_train` y `y_train`).

```python
    # Hacer predicciones y evaluar el modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', pos_label=y_test.unique()[1])
    recall = recall_score(y_test, y_pred, average='weighted', pos_label=y_test.unique()[1])
    f1 = f1_score(y_test, y_pred, average='weighted', pos_label=y_test.unique()[1])

    # Obtener la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
```

Se realizan predicciones en el conjunto de prueba (`X_test`) utilizando el modelo entrenado. Luego, se calculan varias métricas de evaluación del modelo, como la precisión, la sensibilidad (recall), la especificidad y el puntaje F1, utilizando las funciones proporcionadas por `sklearn.metrics`.

También se calcula la matriz de confusión utilizando `confusion_matrix` de `sklearn`.

```python
    # Calcular la especificidad si es una matriz de 2x2
    print(f'Modelo K-Nearest Neighbors\n')
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        print(f'Specificity: {specificity:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Sensitivity (Recall): {recall:.2f}')
    print(f'Specificity: {specificity:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Precisión del modelo: {accuracy:.2f}')
    print('Reporte de clasificación:')
    print(classification_report(y_test, y_pred))
    print('Matriz de confusión:')
    print(cm)
```

Se imprimen las métricas de evaluación del modelo, como la precisión, la sensibilidad (recall), la especificidad, el puntaje F1 y la precisión general. Si la matriz de confusión tiene dimensiones 2x2 (es decir, es un problema de clasificación binaria), también se calcula e imprime la especificidad.

Además, se imprime el reporte de clasificación completo utilizando `classification_report` de `sklearn`, y la matriz de confusión.

```python
    # Guardar el modelo
    joblib.dump(model, 'knn_model.pkl')

    # Cargar el modelo (opcional)
    model = joblib.load('knn_model.pkl')
```

Finalmente, se guarda el modelo entrenado en un archivo llamado `'knn_model.pkl'` utilizando `joblib.dump`. Esto permite cargar y utilizar el modelo más tarde sin necesidad de entrenarlo nuevamente.

Opcionalmente, se muestra cómo cargar el modelo guardado utilizando `joblib.load`.

```python
if __name__ == "__main__":
    load_and_train_model()
```

Esta línea garantiza que la función `load_and_train_model` se ejecute solo cuando el script se ejecute directamente, y no cuando se importe como un módulo en otro script.

En resumen, este código permite cargar un conjunto de datos de características de animales desde un archivo CSV, dividirlo en conjuntos de entrenamiento y prueba, entrenar un modelo de clasificación KNN, evaluar el rendimiento del modelo utilizando varias métricas, guardar el modelo entrenado y cargarlo posteriormente si es necesario.
