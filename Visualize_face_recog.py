import cv2
import mediapipe as mp
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=4) 

model = joblib.load('modelo_knn_face_recog.pkl')

umbral_desconocido = 45

def calcular_angulo(A, B, C):
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    
    cos_theta = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angulo = np.degrees(np.arccos(cos_theta))
    return angulo

angle_points = {
    'Angle1': [33, 133, 160],
    'Angle2': [159, 158, 157],
    'Angle3': [173, 144, 145],
    'Angle4': [153, 145, 33],
    'Angle5': [362, 263, 385],
    'Angle6': [386, 387, 388],
    'Angle7': [390, 373, 374],
    'Angle8': [380, 374, 362],
    'Angle9': [1, 6, 197],
    'Angle10': [195, 5, 4],
    'Angle11': [98, 168, 2],
    'Angle12': [0, 13, 267],
    'Angle13': [269, 270, 409],
    'Angle14': [61, 78, 95],
    'Angle15': [88, 178, 84],
    'Angle16': [55, 65, 107],
    'Angle17': [66, 105, 107],
    'Angle18': [276, 283, 336],
    'Angle19': [293, 300, 336],
    'Angle20': [152, 176, 200],
    'Angle21': [172, 136, 150],
    'Angle22': [400, 379, 394],
    'Angle23': [33, 246, 161],
    'Angle24': [130, 232, 247],
    'Angle25': [352, 423, 398],
    'Angle26': [391, 387, 374],
    'Angle27': [169, 165, 133],
    'Angle28': [463, 388, 263],
    'Angle29': [420, 418, 421],
    'Angle30': [39, 73, 72],
    'Angle31': [338, 285, 296],
    'Angle32': [206, 428, 421],
    'Angle33': [204, 208, 203],
    'Angle34': [400, 379, 427],
    'Angle35': [181, 84, 17],
    'Angle36': [79, 97, 2],
    'Angle37': [250, 368, 366],
    'Angle38': [164, 133, 144],
    'Angle39': [294, 301, 334],
    'Angle40': [90, 78, 191],
    'Angle41': [307, 375, 405],
    'Angle42': [333, 298, 299],
    'Angle43': [325, 361, 405],
    'Angle44': [57, 61, 185],
    'Angle45': [55, 107, 108],
    'Angle46': [147, 148, 114],
    'Angle47': [3, 199, 4],
    'Angle48': [188, 209, 198],
    'Angle49': [196, 209, 237],
    'Angle50': [393, 463, 341],
    'Angle51': [76, 61, 37],
    'Angle52': [206, 182, 150],
    'Angle53': [152, 401, 394],
    'Angle54': [13, 10, 168],
    'Angle55': [170, 108, 109],
    'Angle56': [352, 451, 412],
    'Angle57': [294, 455, 357],
    'Angle58': [243, 190, 101],
    'Angle59': [171, 208, 195],
    'Angle60': [164, 160, 61],

    # Agregados
    'Angle61': [285, 277, 282],
    'Angle62': [336, 296, 334],
    'Angle63': [263, 374, 362],
    'Angle64': [133, 386, 159],
    'Angle65': [6, 168, 57],
    'Angle66': [6, 168, 287],
    'Angle67': [152, 377, 137],
    'Angle68': [152, 382, 377],
    'Angle69': [164, 133, 144],
    'Angle70': [50, 52, 15],
    'Angle71': [280, 282, 34],
    'Angle72': [277, 282, 334],
    'Angle73': [6, 277, 234],
    'Angle74': [10, 338, 285],
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

            for key, (p1, p2, p3) in angle_points.items():
                A = [face_landmarks.landmark[p1].x * w, face_landmarks.landmark[p1].y * h, face_landmarks.landmark[p1].z]
                B = [face_landmarks.landmark[p2].x * w, face_landmarks.landmark[p2].y * h, face_landmarks.landmark[p2].z]
                C = [face_landmarks.landmark[p3].x * w, face_landmarks.landmark[p3].y * h, face_landmarks.landmark[p3].z]
                
                angulo = calcular_angulo(A, B, C)
                angulos.append(angulo)

            caracteristicas = np.array(angulos).reshape(1, -1)

            prediccion = model.predict(caracteristicas)
            distancias, indices = model.kneighbors(caracteristicas)

            if distancias[0][0] > umbral_desconocido:
                label = "Desconocido"
            else:
                label = f'Usuario: {prediccion[0]}'

            x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]

            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            # Dibujar el rectangulo
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Colocar el texto 
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Detección de rostros', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Presiona 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()