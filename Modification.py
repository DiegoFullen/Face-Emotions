import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Cargar datos
data = pd.read_csv('datasets/dataset_faces.csv')
X = data.iloc[:, 1:].values  
y = data.iloc[:, 0].values 

# Datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir los hiperparámetros para optimizar
param_grid = {
    'n_neighbors': [3, 5, 7, 10, 15],         # Número de vecinos
    'weights': ['uniform', 'distance'],      # Pesos
    'metric': ['euclidean', 'manhattan']     # Métrica de distancia
}

# Configuración de GridSearchCV
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Entrenar el modelo con validación cruzada
grid_search.fit(X_train, y_train.flatten())

# Obtener los mejores hiperparámetros
print("Mejores parámetros:", grid_search.best_params_)

# Usar el mejor modelo
best_model = grid_search.best_estimator_

# Validación con el conjunto de prueba
y_pred = best_model.predict(X_test)

# Evaluación
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Guardar el mejor modelo
joblib.dump(best_model, 'modelo_knn_face_recog.pkl')
