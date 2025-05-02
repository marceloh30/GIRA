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
        try:
            if detecciones.get('brazos_cruzados', False):
                self.counts['brazos_cruzados'] += 1
                # print(f"[DEBUG] Frame {self.frames}: brazos_cruzados=True")
            if detecciones.get('hombros_caidos', False):
                self.counts['hombros_caidos'] += 1
                # print(f"[DEBUG] Frame {self.frames}: hombros_caidos=True")
            if detecciones.get('cabeza_baja', False):
                self.counts['cabeza_baja'] += 1
                # print(f"[DEBUG] Frame {self.frames}: cabeza_baja=True")
            # Para contacto visual, contamos los negativos
            if not detecciones.get('contacto_visual', True):
                self.counts['sin_contacto_visual'] += 1
                # print(f"[DEBUG] Frame {self.frames}: sin_contacto_visual=True")
        except Exception as e:
            # Nunca dejamos que un fallo de conteo interrumpa la grabación
            print(f"[ERROR] Sesion.grabar fallo al actualizar counts: {e}")

    def resumen(self) -> dict:
        """
        Devuelve métricas porcentuales de la sesión.
        """
        if self.frames == 0:
            return { k: 0.0 for k in self.counts }
        return {
            k: round(v / self.frames * 100, 2)
            for k, v in self.counts.items()
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