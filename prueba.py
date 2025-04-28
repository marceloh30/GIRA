import cv2
import mediapipe as mp
import pygame  # Para cargar animaciones del avatar
import numpy as np
import math

# Inicializar pygame
pygame.init()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Para dibujar puntos y líneas
pose = mp_pose.Pose()


# Cargar animaciones del avatar
# Crear una superficie de 200x200 píxeles y llenarla de un color
avatar_bored_surf = pygame.Surface((100, 100), pygame.SRCALPHA)
avatar_bored_surf.fill((255, 0, 0, 100))  # Rojo semitransparente

avatar_confused_surf = pygame.Surface((100, 100), pygame.SRCALPHA)
avatar_confused_surf.fill((0, 255, 0, 100))  # Verde semitransparente

# Convertir la superficie de pygame a un array de NumPy
def pygame_surf_to_cv2(surf):
    # Convierte la superficie a un array 3D (ancho x alto x canales)
    array3d = pygame.surfarray.array3d(surf)
    # Transponer para obtener (alto, ancho, canales)
    array3d = np.transpose(array3d, (1, 0, 2))
    # Convertir de RGB (pygame) a BGR (OpenCV)
    return cv2.cvtColor(array3d, cv2.COLOR_RGB2BGR)

avatar_bored = pygame_surf_to_cv2(avatar_bored_surf)
avatar_confused = pygame_surf_to_cv2(avatar_confused_surf)


# Abrir la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

def euclidean_distance(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def detectar_brazos_cruzados(landmarks, mp_pose, x_threshold=0.1):
    """
    Detecta brazos cruzados calculando distancia entre munecas y hombros contrarios
    """
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    
    # Condicion en eje x muneca-hombro: positivo si distancia en x son menores a (x_threshold)
    left_crossed = abs(right_wrist.x - left_shoulder.x) < (x_threshold)
    right_crossed = abs(left_wrist.x - right_shoulder.x) < (x_threshold)

    return left_crossed and right_crossed
'''
def detect_brazos_cruzados(landmarks, mp_pose, ratio_threshold=0.8):
    """
    Detecta si los brazos están cruzados basándose en los landmarks.
    Condiciones:
      - La muñeca izquierda debe estar a la derecha del hombro derecho.
      - La muñeca derecha debe estar a la izquierda del hombro izquierdo.
      - Ambas muñecas deben estar cerca de la línea media formada por los hombros.
    """
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2


    # Distancias para el brazo izquierdo
    d_left_opp = euclidean_distance(left_wrist, right_shoulder)   # distancia a hombro derecho
    d_left_same = euclidean_distance(left_wrist, left_shoulder)     # distancia a hombro izquierdo
    
    # Distancias para el brazo derecho
    d_right_opp = euclidean_distance(right_wrist, left_shoulder)    # distancia a hombro izquierdo
    d_right_same = euclidean_distance(right_wrist, right_shoulder)  # distancia a hombro derecho
    
    left_crossed = d_left_opp < ratio_threshold * d_left_same
    right_crossed = d_right_opp < ratio_threshold * d_right_same

    # Para debug: se pueden imprimir las distancias
    # print(f"Left: d_opp={d_left_opp:.3f}, d_same={d_left_same:.3f}; Right: d_opp={d_right_opp:.3f}, d_same={d_right_same:.3f}")
    
    return left_crossed and right_crossed
'''
def detectar_hombros_caidos(landmarks, mp_pose, posture_threshold=0.25):
    """
    Detecta una postura encogida (hombros caídos) comparando la posición vertical
    promedio de los hombros con la de las caderas.
    Si la diferencia es muy pequeña, se interpreta como una postura encogida.
    """
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_avg_y = (left_hip.y + right_hip.y) / 2

    return (hip_avg_y - shoulder_avg_y) < posture_threshold

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame")
        break
    print("Estoy capturando")
    # Procesar el frame con MediaPipe Pose
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Detectar brazos cruzados (ejemplo simplificado)
    if results.pose_landmarks:

        # Dibujar los landmarks y conexiones
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
        )

        # Detección de brazos cruzados
        if detectar_brazos_cruzados(results.pose_landmarks, mp_pose):
            cv2.putText(frame, "BRAZOS CRUZADOS", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Superponer avatar aburrido
            h, w, _ = avatar_bored.shape
            frame[0:h, 0:w] = avatar_bored

        # Detección de postura encogida
        if detectar_hombros_caidos(results.pose_landmarks, mp_pose):
            cv2.putText(frame, "POSTURA ENCOGIDA", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Opcionalmente, se puede superponer otro avatar (por ejemplo, avatar_confused)
            h, w, _ = avatar_confused.shape
            frame[0:h, w:w*2] = avatar_confused  # Se coloca en otra parte del frame


    cv2.imshow("Practica de presentacion", frame)
    
    #Presionar Escape para salir: 
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()