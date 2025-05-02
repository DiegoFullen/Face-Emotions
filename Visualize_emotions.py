import cv2
import mediapipe as mp
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=4)

model = joblib.load('modelo_knn_emociones.pkl')

def calcular_angulo(A, B, C):
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    
    # Producto punto y magnitud del vector
    cos_theta = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angulo = np.degrees(np.arccos(cos_theta))
    return angulo

angle_points = {
    'Angle1': [33, 133, 160],      # Boca
    'Angle2': [159, 158, 157],     # Boca
    'Angle3': [61, 78, 95],        # Boca
    'Angle4': [88, 178, 84],       # Boca
    'Angle5': [55, 65, 107],       # Boca
    'Angle6': [66, 105, 107],      # Boca
    'Angle7': [291, 375, 405],     # Ojo derecho
    'Angle8': [362, 263, 385],     # Ojo izquierdo
    'Angle9': [386, 387, 388],     # Ojo izquierdo
    'Angle10': [33, 246, 161],     # Cejas y ojos
    'Angle11': [130, 232, 247],    # Cejas y ojos
    'Angle12': [107, 108, 55],     # Boca y mejillas
    'Angle13': [370, 380, 374],    # Cejas y ojos
    'Angle14': [294, 301, 334],    # Frente
    'Angle15': [44, 48, 2],        # Cejas y frente
    'Angle16': [376, 379, 373],    # Frente
    'Angle17': [123, 205, 234],    # Mejillas
    'Angle18': [400, 379, 394],    # Frente
    'Angle19': [107, 118, 55],     # Mejillas
    'Angle20': [109, 107, 55],     # Boca
    'Angle21': [191, 78, 62],      # Boca
    'Angle22': [195, 6, 4],        # Boca
    'Angle23': [293, 300, 336],    # Frente
    'Angle24': [153, 145, 33],     # Mejillas y boca
    'Angle25': [294, 455, 357],    # Frente
    'Angle26': [265, 263, 362],    # Ojo izquierdo
    'Angle27': [61, 76, 37],       # Boca
    'Angle28': [2, 10, 338],       # Frente
    'Angle29': [393, 463, 341],    # Ojo derecho
    'Angle30': [164, 160, 61],     # Boca y mejillas
}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la imagen.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        h, w, _ = frame.shape

        for face_landmarks in result.multi_face_landmarks:

            angulos = []

            # Calcular los angulos
            for key, (p1, p2, p3) in angle_points.items():
                A = [face_landmarks.landmark[p1].x, face_landmarks.landmark[p1].y, face_landmarks.landmark[p1].z]
                B = [face_landmarks.landmark[p2].x, face_landmarks.landmark[p2].y, face_landmarks.landmark[p2].z]
                C = [face_landmarks.landmark[p3].x, face_landmarks.landmark[p3].y, face_landmarks.landmark[p3].z]
                
                angulo = calcular_angulo(A, B, C)
                angulos.append(angulo)

            # Convertir los ángulos en una fila de características
            caracteristicas = np.array(angulos).reshape(1, -1)

            # Hacer predicción con KNN
            prediccion = model.predict(caracteristicas)

            x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]

            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            label = f'Emocion: {prediccion[0]}'
            
            # Dibujar el rectangulo
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Colocar el texto 
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Detección de emociones', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()