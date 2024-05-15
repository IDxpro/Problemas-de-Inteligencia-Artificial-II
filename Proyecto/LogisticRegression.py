import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Carga el dataset
df = pd.read_csv('dataset.csv')

# Separar características y etiquetas
X = df.drop('label', axis=1)
y = df['label']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Hacer predicciones y evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')
print('Reporte de clasificación:')
print(classification_report(y_test, y_pred))
print('Matriz de confusión:')
print(confusion_matrix(y_test, y_pred))

# Guardar el modelo
joblib.dump(model, 'logistic_regression_model.pkl')

# Cargar el modelo (opcional)
model = joblib.load('logistic_regression_model.pkl')
