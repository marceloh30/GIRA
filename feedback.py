import pygame
import cv2
import numpy as np

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
AVATAR_BORED   = crear_avatar((255, 0,   0), alpha=100)  # rojo
AVATAR_CONFUSED= crear_avatar((0,   255, 0), alpha=100)  # verde
AVATAR_WARN    = crear_avatar((255,255, 0), alpha=100)  # amarillo

def superponer_avatar(frame: np.ndarray, avatar: np.ndarray, posicion: tuple = (0,0)):
    """
    Superpone el avatar (BGR) en la esquina o posición indicada del frame.
    """
    h, w, _ = avatar.shape
    x, y = posicion
    frame[y:y+h, x:x+w] = avatar

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
        superponer_avatar(frame, AVATAR_BORED, 0, 0)
    if detecciones['hombros_caidos']:
        superponer_texto(frame, "POSTURA ENCOGIDA", (50,100))
        superponer_avatar(frame, AVATAR_CONFUSED, 110, 0)
    if detecciones['cabeza_baja']:
        superponer_texto(frame, "LEVANTA LA CABEZA", (50,150))
        superponer_avatar(frame, AVATAR_WARN, 220, 0)
    if not detecciones['contacto_visual']:
        superponer_texto(frame, "SIN CONTACTO VISUAL", (50,200), color=(0,0,255))
    else:
        superponer_texto(frame, "CONTACTO VISUAL", (50,200), color=(0,255,0))