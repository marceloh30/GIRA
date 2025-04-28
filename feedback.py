import cv2
import numpy as np
import os

ASSETS_DIR = "assets"

''' ###Intento realizado con pygame que no funcionó
import pygame
# Inicializar Pygame (necesario para crear superficies)
pygame.init()
def crear_avatar(color: tuple, tamano: tuple = (100, 100), alpha: int = 128) -> np.ndarray:
    """
    Crea un avatar semitransparente de un color dado y lo convierte a array OpenCV.
    """
    surf = pygame.Surface(tamano, pygame.SRCALPHA)
    surf.fill((*color, alpha))
    array3d = pygame.surfarray.array3d(surf)
    array3d = np.transpose(array3d, (1, 0, 2))
    return cv2.cvtColor(array3d, cv2.COLOR_RGB2BGR)

# Genera los avatares base
AVATAR_BORED    = crear_avatar((255, 0,   0), alpha=100)  # rojo
AVATAR_CONFUSED = crear_avatar((0,   255, 0), alpha=100)  # verde
AVATAR_WARN     = crear_avatar((255,255, 0), alpha=100)  # amarillo
'''

def cargar_imagen(nombre: str):
    """
    Carga un PNG de assets/name.png con canal alfa y devuelve (img_bgr, alpha_mask).
    """
    path = os.path.join(ASSETS_DIR, nombre + ".png")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"No se encontró {path}")
    # img.shape = (h, w, 4)
    bgr = img[:, :, :3]
    alpha = img[:, :, 3] / 255.0  # máscara 0.0–1.0
    return bgr, alpha

# Precargo los avatares
AVATAR_ABURRIDO,   ALPHA_ABURRIDO   = cargar_imagen("aburrido")
AVATAR_CONFUNDIDO, ALPHA_CONFUNDIDO = cargar_imagen("distraido")
AVATAR_PREOCUPADO, ALPHA_PREOCUPADO = cargar_imagen("preocupado")

def superponer_imagen(frame, img_bgr, alpha, x, y, tamano=(150,150)):
    """
    Superpone el avatar (BGR) en la esquina o posición indicada del frame.
    """
    # Redimensiona avatar a 'size'
    img_bgr = cv2.resize(img_bgr, tamano, interpolation=cv2.INTER_AREA)
    alpha   = cv2.resize(alpha,   tamano, interpolation=cv2.INTER_AREA)
    # (luego el mismo código para superponer)
    h, w = tamano[1], tamano[0]
    fh, fw, _ = frame.shape
    if x + w > fw or y + h > fh:
        w = min(w, fw - x); h = min(h, fh - y)
        img_bgr = img_bgr[0:h, 0:w]; alpha = alpha[0:h, 0:w]
    roi = frame[y:y+h, x:x+w]
    for c in range(3):
        roi[:, :, c] = (alpha * img_bgr[:, :, c] + (1 - alpha) * roi[:, :, c])
    frame[y:y+h, x:x+w] = roi

def superponer_texto(frame: np.ndarray, texto: str, posicion: tuple, color: tuple=(0,0,255), escala: float=1, thickness: int=2):
    """
    Escribe texto sobre el frame en la posición dada.
    """
    cv2.putText(frame, texto, posicion, cv2.FONT_HERSHEY_SIMPLEX, escala, color, thickness)
    
def feedback(frame, detecciones: dict):
    """
    Superpone texto y avatar según detecciones:
      - brazos_cruzados → AVATAR_BORED
      - hombros_caidos  → AVATAR_CONFUSED
      - cabeza_baja     → AVATAR_WARN
      - contacto_visual False → marca en rojo
    """
    if detecciones['brazos_cruzados']:
        superponer_texto(frame, "BRAZOS CRUZADOS", (50,50))
        superponer_imagen(frame, AVATAR_ABURRIDO, ALPHA_ABURRIDO, 0, 0, tamano=(150,150))
    if detecciones['hombros_caidos']:
        superponer_texto(frame, "POSTURA ENCOGIDA", (50,100))
        superponer_imagen(frame, AVATAR_CONFUNDIDO, ALPHA_CONFUNDIDO, 0, 0, tamano=(150,150))
    if detecciones['cabeza_baja']:
        superponer_texto(frame, "LEVANTA LA CABEZA", (50,150))
        superponer_imagen(frame, AVATAR_PREOCUPADO, ALPHA_PREOCUPADO, 0, 0, tamano=(150,150))
    if not detecciones['contacto_visual']:
        superponer_texto(frame, "SIN CONTACTO VISUAL", (50,200), color=(0,0,255))
        superponer_imagen(frame, AVATAR_ABURRIDO, ALPHA_ABURRIDO, 0, 0, tamano=(150,150))
    else:
        superponer_texto(frame, "CONTACTO VISUAL", (50,200), color=(0,255,0))