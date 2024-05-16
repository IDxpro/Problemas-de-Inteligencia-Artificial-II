
Claro, aquí tienes una versión adaptada para un README en GitHub:

Modelo de Clasificación K-Vecinos Más Cercanos (KNN)
Este es un script en Python que muestra cómo construir, entrenar, evaluar y guardar un modelo de clasificación KNN utilizando la librería Scikit-Learn. También incluye funcionalidad para cargar un conjunto de datos desde un archivo CSV usando un cuadro de diálogo basado en Tkinter y preprocesar los datos antes de entrenar el modelo.

Requisitos
Python 3.x
pandas
scikit-learn
joblib
tkinter (para interfaz gráfica)
Instalación de Dependencias
Puedes instalar las librerías necesarias utilizando pip:

bash
Copier le code
pip install pandas scikit-learn joblib
Uso
Ejecuta el script knn_classifier.py en un entorno de Python.
Selecciona un archivo CSV cuando se solicite.
El script entrenará un modelo KNN y mostrará métricas de evaluación como precisión, sensibilidad, F1 score, reporte de clasificación y matriz de confusión.
El modelo entrenado se guardará en un archivo knn_model.pkl en el directorio actual.
Código de Ejemplo
python
Copier le code
# Importación de Librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import tkinter as tk
from tkinter import filedialog

# Función para cargar y entrenar el modelo
def load_and_train_model():
    # Abrir un cuadro de diálogo para seleccionar el archivo CSV
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if not file_path:
        print("No se seleccionó ningún archivo.")
        return

    # Cargar el dataset
    df = pd.read_csv(file_path)

    # Eliminar la columna animal_name
    df = df.drop(columns=['animal_name'])

    # Asumir que la última columna es la etiqueta
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar las características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Crear y entrenar el modelo KNN
    model = KNeighborsClassifier(n_neighbors=7)  # Usando k=5 como ejemplo
    model.fit(X_train, y_train)

    # Hacer predicciones y evaluar el modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Obtener la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Calcular la especificidad si es una matriz de 2x2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        print(f'Especificidad: {specificity:.2f}')

    print(f'Precisión del modelo: {accuracy:.2f}')
    print('Reporte de clasificación:')
    print(classification_report(y_test, y_pred))
    print('Matriz de confusión:')
    print(cm)

    # Guardar el modelo
    joblib.dump(model, 'knn_model.pkl')

    # Cargar el modelo (opcional)
    model = joblib.load('knn_model.pkl')

# Ejecución Principal
if __name__ == "__main__":
    load_and_train_model()
Contribuciones
Si deseas contribuir a este proyecto, ¡siéntete libre de hacerlo! Puedes enviar pull requests o abrir problemas para discutir nuevas ideas.
