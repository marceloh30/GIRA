from captura_analisis import abrir_camara, process_frame
from feedback import feedback
from analisis_evolutivo import SessionRecorder
import cv2

def main():
    cap = abrir_camara()
    recorder = SessionRecorder("sesion.avi")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            det = process_frame(frame)
            feedback(frame, det)
            recorder.record(frame, det)

            cv2.imshow("GIRA Prototipo", frame)
            if cv2.waitKey(1) == 27: break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recorder.save_report("reporte.json")

if __name__ == "__main__":
    main()