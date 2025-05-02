import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def calculate_angle(point1, point2, point3):
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    angulo_cos = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angulo = np.arccos(angulo_cos)
    return np.degrees(angulo)

emotion = "Neutral"
dataset = 'datasets/dataset_emotions.csv'

# Puntos
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

with open(dataset, mode='a', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    cap = cv2.VideoCapture(0)
    previous_time = 0
    interval = 0.025

    angles = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(image_rgb)

        current_time = time.time() 
        if current_time - previous_time >= interval:
            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    h, w, _ = image.shape
                    points = []
                    for idx, lm in enumerate(face_landmarks.landmark):
                        x, y = int(lm.x * w), int(lm.y * h)
                        points.append([x, y, lm.z])

                    # Calcular ángulos
                    angles = []
                    for key, indices in angle_points.items():
                        angle = calculate_angle(points[indices[0]], points[indices[1]], points[indices[2]])
                        angles.append(angle)

                    # Guardar datos
                    writer.writerow([emotion] + angles)
                    file.flush()
                    print(f"Ángulos guardados, Emoción: {emotion}")

                    previous_time = current_time

                    #Cuadro Verde
                    x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
                    y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]

                    x_min = min(x_coords)
                    x_max = max(x_coords)
                    y_min = min(y_coords)
                    y_max = max(y_coords)

                    # Draw the rectangle
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow('FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Esc para salir
            break

    cap.release()
    cv2.destroyAllWindows()