import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

def calculate_angle(point1, point2, point3):
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

while cap.isOpened():
    success, image = cap.read()
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:

            point1 = landmarks.landmark[75]      # Verde - Base
            point2 = landmarks.landmark[20]     # Rojo  - Central
            point3 = landmarks.landmark[290]    # Azul  - Base

            h, w, _ = image.shape
            point1 = (int(point1.x * w), int(point1.y * h))
            point2 = (int(point2.x * w), int(point2.y * h))
            point3 = (int(point3.x * w), int(point3.y * h))

            cv2.circle(image, point1, 3, (0, 255, 0), -1)  
            cv2.circle(image, point2, 3, (255, 0, 0), -1)  
            cv2.circle(image, point3, 3, (0, 0, 255), -1)  

    cv2.imshow("FaceMesh", image)

    print(f"Ángulo: {calculate_angle(point1, point2, point3)}" , "point1:" ,  point1 , "point2:" , point2 , "point3:" , point3)
    time.sleep(0.2)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()