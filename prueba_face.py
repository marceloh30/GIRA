import cv2
import mediapipe as mp
import math

# Inicializar MediaPipe Pose (para detectar la inclinación de la cabeza)
mp_pose = mp.solutions.pose
mp_drawing_pose = mp.solutions.drawing_utils  # Para dibujar puntos y conexiones de Pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializar MediaPipe Face Mesh (para detectar el iris y el contacto visual)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_face = mp.solutions.drawing_utils  # Para dibujar landmarks faciales
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Habilita los landmarks del iris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def detect_head_tilt_down_v2(pose_landmarks, threshold=0.1):
    """
    Detecta si la cabeza está inclinada hacia abajo usando el landmark de la nariz.
    Se compara la posición vertical de la nariz (índice 0) con el promedio de los hombros (índices 11 y 12)
    y se normaliza por el ancho de los hombros.
    
    En una postura erguida, la diferencia (nariz.y - hombros_avg.y) suele ser más negativa.
    Si la cabeza se inclina hacia abajo, la diferencia se acerca a 0.
    Si norm_diff > threshold (por ejemplo, -0.1), se detecta inclinación hacia abajo.
    """
    try:
        nose = pose_landmarks.landmark[0]
        left_shoulder = pose_landmarks.landmark[11]
        right_shoulder = pose_landmarks.landmark[12]
    except Exception as e:
        return False

    shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
    shoulder_width = euclidean_distance(left_shoulder, right_shoulder)
    if shoulder_width == 0:
        return False

    diff = nose.y - shoulder_avg_y  # En coordenadas normalizadas: valores mayores indican posición más baja.
    norm_diff = diff / shoulder_width  # Normalizamos para compensar la distancia a la cámara.

    # Debug: descomenta para ver los valores en consola.
    print(f"nose.y: {nose.y:.3f}, shoulder_avg_y: {shoulder_avg_y:.3f}, norm_diff: {norm_diff:.3f}")

    return norm_diff > threshold

def detectar_cabeza_baja(pose_landmarks, threshold=-0.7):
    """
    Detecta si la cabeza está inclinada hacia abajo usando los landmarks de Pose.
    Se compara el promedio vertical (y) de los ojos (índices 2 y 5) con el promedio
    de los hombros (índices 11 y 12) y se normaliza con el ancho de los hombros.
    """
    # Verifica que existan los landmarks necesarios
    try:
        left_eye = pose_landmarks.landmark[2]   # ojo izquierdo (ajustar si es necesario)
        right_eye = pose_landmarks.landmark[5]    # ojo derecho
        left_shoulder = pose_landmarks.landmark[11]
        right_shoulder = pose_landmarks.landmark[12]
    except:
        return False

    eye_avg_y = (left_eye.y + right_eye.y) / 2
    shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2

    shoulder_width = euclidean_distance(left_shoulder, right_shoulder)
    if shoulder_width == 0:
        return False

    diff = eye_avg_y - shoulder_avg_y
    norm_diff = diff / shoulder_width

    # Debug: descomenta para ver los valores
    print(f"Head Tilt norm_diff: {norm_diff:.3f}")

    return norm_diff > threshold

def average_landmarks(face_landmarks, indices):
    """Calcula el promedio (x, y) de los landmarks indicados."""
    x, y = 0, 0
    for i in indices:
        x += face_landmarks.landmark[i].x
        y += face_landmarks.landmark[i].y
    count = len(indices)
    return x / count, y / count

def detect_eye_contact_iris(face_landmarks, threshold_ratio=0.1):
    """
    Determina el contacto visual calculando la diferencia relativa entre el centro del iris y el centro del ojo.
    Para el ojo izquierdo se usan los índices 33 y 133 para las esquinas y 468 para el iris.
    Para el ojo derecho se usan 362 y 263 para las esquinas y 473 para el iris.
    Se normaliza la diferencia dividiéndola por el ancho del ojo.
    """
    # Ojo izquierdo
    left_eye_left_corner = face_landmarks.landmark[33]
    left_eye_right_corner = face_landmarks.landmark[133]
    left_eye_width = abs(left_eye_right_corner.x - left_eye_left_corner.x)
    left_iris_center = face_landmarks.landmark[468]
    left_eye_center = ( (left_eye_left_corner.x + left_eye_right_corner.x) / 2 )
    diff_left = abs(left_iris_center.x - left_eye_center)
    normalized_left = diff_left / left_eye_width if left_eye_width else float('inf')

    # Ojo derecho
    right_eye_left_corner = face_landmarks.landmark[362]
    right_eye_right_corner = face_landmarks.landmark[263]
    right_eye_width = abs(right_eye_right_corner.x - right_eye_left_corner.x)
    right_iris_center = face_landmarks.landmark[473]
    right_eye_center = ( (right_eye_left_corner.x + right_eye_right_corner.x) / 2 )
    diff_right = abs(right_iris_center.x - right_eye_center)
    normalized_right = diff_right / right_eye_width if right_eye_width else float('inf')

    # Debug
    #print(f"Normalized Left: {normalized_left:.3f}, Normalized Right: {normalized_right:.3f}")

    return normalized_left < threshold_ratio and normalized_right < threshold_ratio

# Abrir la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen con ambas soluciones
    results_pose = pose.process(frame_rgb)
    results_face = face_mesh.process(frame_rgb)
    
    # Detección de inclinación de cabeza usando Pose
    if results_pose.pose_landmarks:
        mp_drawing_pose.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_pose.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing_pose.DrawingSpec(color=(255,0,0), thickness=2)
        )
        if detectar_cabeza_baja(results_pose.pose_landmarks):
            cv2.putText(frame, "LEVANTA LA CABEZA", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    # Detección de contacto visual usando Face Mesh
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing_face.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing_face.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing_face.DrawingSpec(color=(0,0,255), thickness=1)
            )
            mp_drawing_face.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=mp_drawing_face.DrawingSpec(color=(255,255,0), thickness=1, circle_radius=1)
            )
            if detect_eye_contact_iris(face_landmarks, threshold_ratio=0.1):
                cv2.putText(frame, "CONTACTO VISUAL", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame, "SIN CONTACTO VISUAL", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow("Deteccion de Cabeza y Contacto Visual", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
face_mesh.close()