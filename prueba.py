import cv2
import mediapipe as mp
import pygame  # Para cargar animaciones del avatar
import numpy as np

# Inicializar pygame
pygame.init()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Para dibujar puntos y líneas
pose = mp_pose.Pose()


# Cargar animaciones del avatar
# Crear una superficie de 200x200 píxeles y llenarla de un color
avatar_bored_surf = pygame.Surface((200, 200), pygame.SRCALPHA)
avatar_bored_surf.fill((255, 0, 0, 100))  # Rojo semitransparente

avatar_confused_surf = pygame.Surface((200, 200), pygame.SRCALPHA)
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

def detect_brazos_cruzados(landmarks, mp_pose, wrist_threshold=0.15):
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

    left_wrist_crossed = left_wrist.x > right_shoulder.x and abs(left_wrist.x - shoulder_center_x) < wrist_threshold
    right_wrist_crossed = right_wrist.x < left_shoulder.x and abs(right_wrist.x - shoulder_center_x) < wrist_threshold

    return left_wrist_crossed and right_wrist_crossed

def detect_hombros_caidos(landmarks, mp_pose, posture_threshold=0.1):
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

        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Ejemplo simplificado: si la muñeca izquierda está cerca del hombro derecho
        if abs(left_wrist.x - right_shoulder.x) < 0.1:
            cv2.putText(frame, "BRAZOS CRUZADOS DETECTADOS", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Superponer el avatar aburrido en la esquina superior izquierda
            h, w, _ = avatar_bored.shape
            frame[0:h, 0:w] = avatar_bored

    cv2.imshow("Practica de presentacion", frame)
    
    #Presionar Escape para salir: 
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()