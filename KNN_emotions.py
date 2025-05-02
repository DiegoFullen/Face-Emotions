import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Cargar datos
data = pd.read_csv('datasets/dataset_emotions.csv')
X = data.iloc[:, 1:].values 
y = data.iloc[:, 0].values 

# Datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_neighbors = 9  # Num Vecinos
model = KNeighborsClassifier(n_neighbors)

y_train = y_train.flatten() 

model.fit(X_train, y_train)

# Validacion con el porcentaje fuera
y_pred = model.predict(X_test)
# Validacion cruzada
y_all_pred = model.predict(data.iloc[:, 1:].values)

# Evaluacion
print(confusion_matrix(y_test, y_pred))
# Validacion simple
#print(classification_report(y_test, y_pred))
# Validacion cruzada
print(classification_report(data.iloc[:, 0].values, y_all_pred))

joblib.dump(model, 'modelo_knn_emociones.pkl')