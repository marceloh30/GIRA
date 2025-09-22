# GIRA: Gestión Interactiva de Retroalimentación en Audiencias

GIRA es una herramienta que utiliza tu webcam para analizar en tiempo real tu lenguaje corporal durante una presentación y darte feedback visual para mejorar tu comunicación.

---
##  Características principales

* **Análisis en tiempo real**: Detecta brazos cruzados, hombros caídos, cabeza baja y contacto visual.
* **Feedback visual**: Muestra "avatares" (versión actual solo muestra una imagen) y texto en pantalla para indicar áreas de mejora.
* **Grabación de la sesión**: Guarda un video de la presentación en un archivo `sesion.avi`.
* **Reporte final**: Genera un archivo `reporte.json` con estadísticas del rendimiento.

---
## ⚙️ Instalación

Se recomienda utilizar un entorno virtual para este proyecto.

1.  **Clona el repositorio y navega al directorio.**
    ```bash
    git clone [https://github.com/tu-usuario/GIRA.git](https://github.com/tu-usuario/GIRA.git)
    cd GIRA
    ```

2.  **Crea y activa el entorno virtual.**
    ```bash
    # Para Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```
Se recomienda el uso de Python 3.9.

3.  **Actualiza las herramientas de instalación.**
    ```bash
    python -m pip install --upgrade pip setuptools wheel
    ```

4.  **Instala las dependencias necesarias.**
    El proyecto utiliza **MediaPipe** y **OpenCV**.
    ```bash
    pip install mediapipe opencv-python
    ```

---
## ▶️ Uso

1.  **Ejecuta el script principal** desde la terminal.
    ```bash
    python main.py
    ```

2.  **Comienza tu presentación**. Se abrirá una ventana mostrando la cámara y el feedback en vivo.

3.  **Presiona la tecla `Esc`** para finalizar la sesión.

4.  **Revisa los resultados**. Al terminar, encontrarás los archivos `sesion.avi` y `reporte.json` en la carpeta del proyecto.
