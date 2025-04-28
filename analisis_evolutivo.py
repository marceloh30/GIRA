import cv2
import json
from datetime import datetime

class Sesion:
    """
    Graba métricas de gestos por sesión y permite un análisis evolutivo.
    """
    def __init__(self, out_video_path=None):
        self.frames = 0
        self.counts = {
            'brazos_cruzados': 0,
            'hombros_caidos':  0,
            'cabeza_baja':     0,
            'sin_contacto_visual': 0
        }
        self.out = None
        if out_video_path:
            # Configura escritor de vídeo (asume 640x480 a 20 FPS)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(out_video_path, fourcc, 20.0, (640,480))

    def grabar(self, frame, detecciones: dict):
        self.frames += 1
        if self.out:
            self.out.write(frame)
        # Acumula eventos
        for k in ['brazos_cruzados','hombros_caidos','cabeza_baja']:
            if detecciones[k]:
                self.counts[k] += 1
        if not detecciones['contacto_visual']:
            self.counts['sin_contacto_visual'] += 1

    def resumen(self) -> dict:
        """
        Devuelve métricas porcentuales de la sesión.
        """
        if self.frames == 0:
            return {}
        return {
            k: round(v / self.frames * 100, 2)
            for k,v in self.counts.items()
        }

    def guardar_reporte(self, path: str):
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_frames': self.frames,
            'metrics': self.resumen()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        if self.out:
            self.out.release()