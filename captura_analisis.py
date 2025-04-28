import cv2
import mediapipe as mp
import math
import traceback

### Captura con OpenCV:
def abrir_camara(index: int = 0) -> cv2.VideoCapture:
    """
    Abre la cámara y comprueba que esté disponible.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise IOError("No se pudo abrir la cámara")
    return cap


### Analisis de Gestos con MediaPipe:

# Inicializa MediaPipe Pose y Face Mesh
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def safe_detect(func, *args, **kwargs):
    """Ejecuta func(*args) y atrapa excepciones, devolviendo False si falla."""
    try:
        return func(*args, **kwargs)
    except Exception:
        print(f"[ERROR] en {func.__name__}:\n", traceback.format_exc())
        return False

def distancia_euclidea(p1, p2) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def detectar_brazos_cruzados(landmarks, umbral_x = 0.1) -> bool:
    """
    Detecta brazos cruzados comparando distancias horizontales muneca–hombro.
    """
    mun_izq = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    mun_der = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    hombro_izq = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    hombro_der = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    cruzado_izq = abs(mun_der.x - hombro_izq.x) < umbral_x
    cruzado_der = abs(mun_izq.x - hombro_der.x) < umbral_x
    return cruzado_izq and cruzado_der

def detectar_hombros_caidos(landmarks, umbral_postura = 0.25) -> bool:
    """
    Detecta postura encogida comparando hombros vs caderas
    """
    hombro_izq = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    hombro_der = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    cadera_izq = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    cadera_der = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    y_hombros = (hombro_izq.y + hombro_der.y) / 2
    y_caderas = (cadera_izq.y + cadera_der.y) / 2
    return (y_caderas - y_hombros) < umbral_postura

def detectar_cabeza_baja(landmarks, umbral = -0.1) -> bool:
    """
    Detecta inclinación de cabeza hacia abajo usando nariz vs hombros
    """
    nariz = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    hombro_izq = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    hombro_der = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    y_hombros = (hombro_izq.y + hombro_der.y) / 2
    largo_hombros = distancia_euclidea(hombro_izq, hombro_der)
    if largo_hombros == 0:
        return False

    dif_normalizada = (nariz.y - y_hombros) / largo_hombros
    return dif_normalizada > umbral

def detectar_contacto_visual(face_landmarks, umbral = 0.1) -> bool:
    """
    Detecta contacto visual comparando posición del iris vs centro del ojo
    """
    # Ojo izquierdo
    ojo_izq_extizq = face_landmarks.landmark[33]
    ojo_izq_extder = face_landmarks.landmark[133]
    largo_ojo_izq = abs(ojo_izq_extder.x - ojo_izq_extizq.x)
    centro_iris_izq = face_landmarks.landmark[468]
    dif_izq = (ojo_izq_extizq.x + ojo_izq_extder.x) / 2
    izq_normalizado = abs(centro_iris_izq.x - dif_izq) / (largo_ojo_izq if largo_ojo_izq else 1)

    # Ojo derecho
    ojo_der_extizq = face_landmarks.landmark[362]
    ojo_der_extder = face_landmarks.landmark[263]
    largo_ojo_der = abs(ojo_der_extder.x - ojo_der_extizq.x)
    centro_iris_der = face_landmarks.landmark[473]
    dif_der = (ojo_der_extizq.x + ojo_der_extder.x) / 2
    der_normalizado = abs(centro_iris_der.x - dif_der) / (largo_ojo_der if largo_ojo_der else 1)

    return izq_normalizado < umbral and der_normalizado < umbral

def procesar_frame(frame: cv2.Mat) -> dict:
    """
    Devuelve un dict con:
      - brazos_cruzados
      - hombros_caidos
      - cabeza_baja
      - contacto_visual
    """
    out = {
        'brazos_cruzados': False,
        'hombros_caidos': False,
        'cabeza_baja': False,
        'contacto_visual': True
    }
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res_p = pose.process(rgb)
    if res_p.pose_landmarks:
        lm = res_p.pose_landmarks
        out['brazos_cruzados'] = safe_detect(detectar_brazos_cruzados, lm)
        out['hombros_caidos']  = safe_detect(detectar_hombros_caidos, lm)
        out['cabeza_baja']     = safe_detect(detectar_cabeza_baja, lm)
    res_f = face_mesh.process(rgb)
    if res_f.multi_face_landmarks:
        out['contacto_visual'] = safe_detect(detectar_contacto_visual, res_f.multi_face_landmarks[0])
    return out