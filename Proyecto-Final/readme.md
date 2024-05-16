Este código es un script en Python que muestra cómo construir, entrenar, evaluar y guardar un modelo de clasificación de Vecinos Más Cercanos (K-Nearest Neighbors, KNN) utilizando la librería Scikit-Learn. También incluye funcionalidad para cargar un conjunto de datos desde un archivo CSV usando un cuadro de diálogo basado en Tkinter y preprocesar los datos antes de entrenar el modelo.

Cómo Usar:

Dependencias:

Python 3.x
pandas
scikit-learn
joblib
tkinter
Instalación:
Asegúrate de tener instaladas las librerías necesarias. Si no, puedes instalarlas usando pip:

Copier le code
pip install pandas scikit-learn joblib
Ejecución:
Ejecuta el script en un entorno de Python. Al ejecutarlo, te pedirá seleccionar un archivo CSV que contenga el conjunto de datos.

Resumen de Funcionalidades:

El script utiliza pandas para cargar el conjunto de datos desde el archivo CSV seleccionado.
Preprocesa los datos eliminando una columna específica (animal_name en este caso), asumiendo que la última columna es la etiqueta objetivo.
Divide los datos en conjuntos de entrenamiento y prueba usando train_test_split de Scikit-Learn.
Normaliza las características utilizando StandardScaler.
Crea un clasificador KNN y lo entrena con los datos de entrenamiento.
Realiza predicciones en el conjunto de prueba y calcula diversas métricas de evaluación como precisión, sensibilidad, F1 score, reporte de clasificación y matriz de confusión.
Si la matriz de confusión es de 2x2, calcula la especificidad como una métrica adicional.
Guarda el modelo entrenado en un archivo (knn_model.pkl) usando joblib.
Desglose del Código:

Importación de Librerías:

python
Copier le code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import tkinter as tk
from tkinter import filedialog
Función load_and_train_model:

python
Copier le code
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
Ejecución Principal (if __name__ == "__main__":):

python
Copier le code
if __name__ == "__main__":
    load_and_train_model()
Notas:

Asegúrate de que tu archivo CSV tenga un formato adecuado para la tarea de clasificación.
Personaliza el código según tus datos y requisitos específicos.
Puedes ajustar los hiperparámetros del modelo KNN (por ejemplo, n_neighbors) para optimización.
Considera la validación cruzada o la sintonización de parámetros para mejorar aún más el modelo.
El modelo guardado se puede cargar más tarde para hacer predicciones sobre nuevos datos sin necesidad de volver a entrenarlo.
