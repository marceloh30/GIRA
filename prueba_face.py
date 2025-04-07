import cv2
import mediapipe as mp
import math

# Inicializar Face Mesh con refine_landmarks=True para obtener los landmarks del iris.
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Esto habilita los landmarks del iris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)



def average_landmarks(face_landmarks, indices):
    """Calcula el promedio (x, y) de los landmarks indicados."""
    x, y = 0, 0
    for i in indices:
        x += face_landmarks.landmark[i].x
        y += face_landmarks.landmark[i].y
    count = len(indices)
    return x / count, y / count

def detect_eye_contact_iris(face_landmarks, threshold_ratio=0.1):
    # Para el ojo izquierdo:
    left_eye_left_corner = face_landmarks.landmark[33]
    left_eye_right_corner = face_landmarks.landmark[133]
    left_eye_width = abs(left_eye_right_corner.x - left_eye_left_corner.x)
    left_iris_center = face_landmarks.landmark[468]
    left_eye_center = (left_eye_left_corner.x + left_eye_right_corner.x) / 2
    diff_left = abs(left_iris_center.x - left_eye_center)
    normalized_left = diff_left / left_eye_width

    # Para el ojo derecho:
    right_eye_left_corner = face_landmarks.landmark[362]
    right_eye_right_corner = face_landmarks.landmark[263]
    right_eye_width = abs(right_eye_right_corner.x - right_eye_left_corner.x)
    right_iris_center = face_landmarks.landmark[473]
    right_eye_center = ((right_eye_left_corner.x + right_eye_right_corner.x) / 2)
    diff_right = abs(right_iris_center.x - right_eye_center)
    normalized_right = diff_right / right_eye_width

    # Debug
    print(f"Normalized Left: {normalized_left:.3f}, Normalized Right: {normalized_right:.3f}")

    return normalized_left < threshold_ratio and normalized_right < threshold_ratio
'''
def detect_eye_contact_iris(face_landmarks, threshold=0.1):
    """
    Determina el contacto visual calculando el centro de cada iris y comparándolo
    con el centro del ojo (definido como el promedio de las esquinas internas y externas).
    Si la diferencia es menor que 'threshold' para ambos ojos, se considera que hay contacto visual.
    """
    # Para el ojo izquierdo:
    # Esquinas: índice 33 (externa) y 133 (interna)
    left_eye_center = ((face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2,
                       (face_landmarks.landmark[33].y + face_landmarks.landmark[133].y) / 2)
    # Iris: índices 474, 475, 476, 477
    left_iris_center = average_landmarks(face_landmarks, [474, 475, 476, 477])
    
    # Para el ojo derecho:
    # Esquinas: índice 263 (externa) y 362 (interna)
    right_eye_center = ((face_landmarks.landmark[263].x + face_landmarks.landmark[362].x) / 2,
                        (face_landmarks.landmark[263].y + face_landmarks.landmark[362].y) / 2)
    # Iris: índices 469, 470, 471, 472
    right_iris_center = average_landmarks(face_landmarks, [469, 470, 471, 472])
    
    # Calcular la diferencia absoluta en la dirección x (horizontal) para cada ojo
    diff_left = abs(left_iris_center[0] - left_eye_center[0])
    diff_right = abs(right_iris_center[0] - right_eye_center[0])
    
    print(f"Izquierdo: diff={diff_left:.3f}, Derecho: diff={diff_right:.3f}")
    
    # También se puede evaluar la diferencia vertical si se desea (aquí nos centramos en la horizontal)
    # diff_left_y = abs(left_iris_center[1] - left_eye_center[1])
    # diff_right_y = abs(right_iris_center[1] - right_eye_center[1])
    
    # Debug: Puedes imprimir los valores para ajustar el umbral
    # print(f"Izquierdo: diff={diff_left:.3f}, Derecho: diff={diff_right:.3f}")
    
    return diff_left < threshold and diff_right < threshold
'''
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
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujar la malla facial (opcional)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))
            
            # Dibujar la conexión de los iris (opcional)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,0), thickness=1, circle_radius=1))
            
            # Detectar contacto visual usando la posición del iris
            if detect_eye_contact_iris(face_landmarks):
                cv2.putText(frame, "CONTACTO VISUAL", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "SIN CONTACTO VISUAL", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Deteccion de Contacto Visual con Iris", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()