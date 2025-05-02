from captura_analisis import abrir_camara, procesar_frame
from feedback import feedback
from analisis_evolutivo import Sesion
import cv2

def main():
    cap = abrir_camara()
    recorder = Sesion("sesion.avi")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            det = procesar_frame(frame)
            feedback(frame, det)
            recorder.grabar(frame, det)

            cv2.imshow("GIRA Prototipo", frame)
            if cv2.waitKey(1) == 27: break
    except Exception as e:
        print(e)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nResultados de la sesión:\n","Frames totales: ",recorder.frames)
        print("Numero de frames de cada gesto inadecuado:\n")
        for gesto, numeroFrames in recorder.counts.items():
            print(f"{gesto}: Encontrado en {numeroFrames} frames")
            print(f"Detectado en un {(100*numeroFrames/recorder.frames):.2f}% del tiempo.")
        recorder.guardar_reporte("reporte.json")
        

if __name__ == "__main__":
    main()