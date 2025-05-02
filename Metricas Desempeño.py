import pandas as pd
import seaborn as sb
import joblib
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


# ------------------ Cargar Datos ------------------
# Cargar datos
data = pd.read_csv('datasets/dataset_faces.csv')

# ------------------ Preprocesar Datos ----------------
# Codificar dinámicamente
data_encoded = data.copy()

# Codificar dinámicamente las columnas categóricas
for column in data_encoded.select_dtypes(include='object').columns:
    data_encoded[column] = LabelEncoder().fit_transform(data_encoded[column])

# Columna objetivo
objetivo = "User"  # Cambiar según corresponda
X = data_encoded.drop(columns=[objetivo])
y = data_encoded[objetivo]

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------ Entrenar Modelo KNN ------------------
model = KNeighborsClassifier(n_neighbors=9)  # Ajustar n_neighbors según corresponda
model.fit(X_train, y_train)

# ---------------------------------------------------------
# ------------------ Matriz de Confusión ------------------
# ---------------------------------------------------------
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
emotions = LabelEncoder().fit(data[objetivo])
clases = emotions.classes_

# Visualizar matriz de confusión
plt.figure(figsize=(8, 6))
sb.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clases, yticklabels=clases)
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.title("Matriz de Confusión")

#"""
# --------------------------------------------------------
# ------------------ Curva ROC-AUC -----------------------
# --------------------------------------------------------
# Binarizar las etiquetas
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))  # Etiquetas binarizadas
n_classes = y_test_bin.shape[1]

# Calcular ROC y AUC para cada clase
for i in range(n_classes):
    fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], model.predict_proba(X_test)[:, i])
    auc = roc_auc_score(y_test_bin[:, i], model.predict_proba(X_test)[:, i])

    # Graficar curva ROC para la clase actual
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (Clase {clases[i]}, AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guessing")
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title(f"Curva ROC para la Clase {clases[i]}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.5)

# --------------------------------------------------------
# ------------- Curva de Precisión Recall ----------------
# --------------------------------------------------------
# Binarizar las etiquetas de prueba
# Obtener las etiquetas originales desde el codificador
encoder = LabelEncoder()
encoder.fit(data[objetivo])  # Ajustamos con los datos originales
class_labels = encoder.classes_

# Binarizar las etiquetas de prueba
y_test_bin = label_binarize(y_test, classes=np.arange(len(class_labels)))

# Obtener probabilidades de predicción para cada clase
y_scores = model.predict_proba(X_test)

# Calcular la curva Precision-Recall para cada clase
for i, class_label in enumerate(class_labels):
    # Curva de precisión y recall para la clase actual
    precision, recall, thresholds = precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
    avg_precision = average_precision_score(y_test_bin[:, i], y_scores[:, i])

    # Graficar
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"{class_label} (AP = {avg_precision:.2f})")
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title(f"Curva Precisión-Recall para {class_label}", fontsize=16)
    plt.legend(loc="best", fontsize=12)
    plt.grid(alpha=0.5)
#"""
# --------------------------------------------------------
# --------------- Curva de Aprendizaje -------------------
# --------------------------------------------------------
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring="accuracy", n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Promedio y desviación estándar
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Puntaje de Entrenamiento")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="blue")
plt.plot(train_sizes, test_scores_mean, 'o-', color="green", label="Puntaje de Validación")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="green")
plt.title("Curva de Aprendizaje", fontsize=16)
plt.xlabel("Tamaño del Conjunto de Entrenamiento", fontsize=14)
plt.ylabel("Puntaje", fontsize=14)
plt.legend(loc="best", fontsize=12)
plt.grid(alpha=0.5)

# --------------------------------------------------------
# ---------------- Curva de Validación -------------------
# --------------------------------------------------------
param_name = "n_neighbors"
param_range = np.arange(1, 21)

train_scores, test_scores = validation_curve(
    model, X, y, param_name=param_name, param_range=param_range, cv=5, scoring="accuracy", n_jobs=-1
)

# Promedio y desviación estándar
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(param_range, train_scores_mean, 'o-', color="blue", label="Puntaje de Entrenamiento")
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="blue")
plt.plot(param_range, test_scores_mean, 'o-', color="green", label="Puntaje de Validación")
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="green")
plt.title("Curva de Validación", fontsize=16)
plt.xlabel(param_name, fontsize=14)
plt.ylabel("Puntaje", fontsize=14)
plt.legend(loc="best", fontsize=12)
plt.grid(alpha=0.5)

plt.show()