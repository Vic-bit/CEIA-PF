# Desarrollo de un sistema de posicionamiento y localización mediante odometría monocular

Proyecto Final de Estudios - CEIA  
Autor: Víctor David Silva

## Descripción

Este proyecto implementa un sistema de Visual SLAM (Simultaneous Localization and Mapping) utilizando odometría monocular. El objetivo es estimar la trayectoria de una cámara y reconstruir un mapa 3D del entorno a partir de una secuencia de imágenes.  
Esta versión corresponde a la implementación clásica basada en SIFT (Scale-Invariant Feature Transform).

## Estructura del Proyecto

- `src/sift_classic/`: Implementación principal del sistema SLAM con SIFT clásico.
  - `main.py`: Script principal de ejecución.
  - `features.py`: Extracción y emparejamiento de características SIFT.
  - `pointmap.py`: Estructuras de datos para el mapa y puntos 3D.
  - `display.py`: Visualización de trayectoria, puntos y métricas.
  - `utils.py`: Utilidades para calibración y manejo de datos.
  - `config.py`: Parámetros de configuración.
- `src/sift_kornia/`: Implementación del sistema SLAM con SIFT usando Kornia.
  - `main.py`: Script principal de ejecución.
  - `features.py`: Extracción y emparejamiento de características SIFT.
  - `pointmap.py`: Estructuras de datos para el mapa y puntos 3D.
  - `display.py`: Visualización de trayectoria, puntos y métricas.
  - `utils.py`: Utilidades para calibración y manejo de datos.
  - `config.py`: Parámetros de configuración.
- `src/sift_rpi/`: Implementación del sistema SLAM con SIFT usando Raspberry Pi.
- `dataset/`: Secuencias de imágenes y archivos de calibración (formato KITTI).
- `notebooks/`: Análisis exploratorio y pruebas.
- `outputs/`: Resultados y visualizaciones generadas.

## Versiones

El proyecto cuenta con tres versiones, cada una explorando diferentes técnicas de extracción de características y frameworks.  

Las 4 versiones a desarrollar son:
- SIFT classic
- SIFT kornia
- SIFT Raspberry Pi 

## Requisitos

- Python 3.8+
- OpenCV (`opencv-python`)
- PyQt5
- pyqtgraph
- numpy

## SIFT classic

### Ejecución

1. Asegúrate de tener el dataset en la ruta `dataset/00/` con imágenes y archivo de calibración. El cual se obtiene de https://www.cvlibs.net/datasets/kitti/
2. Modifica los parámetros en `src/sift_classic/config.py` si es necesario.
3. Ejecuta el sistema desde la carpeta raíz del proyecto:

```bash
python src/sift_classic/main.py
```

Se abrirá una interfaz gráfica mostrando la imagen actual, la trayectoria estimada y los puntos 3D reconstruidos.

## SIFT con Raspberry Pi
### Instalación 

Se debe instalar primero desde el sistema, es decir, sin ningún venv activado:
```bash
pip install opencv-python-headless
sudo apt install python3-libgpiod python3-lgpio -y
sudo apt install python3-picamera2 -y
```

Crear el entorno virtual con acceso al sistema:
```bash
python3 -m venv --system-site-packages voenv
```

Activar el entorno virtual:
```bash
source voenv/bin/activate
```

Instala las dependencias con:
```bash 
pip install -r requirements.txt
```

### Ejecución SIFT Raspberry Pi

Debido a que se dificultó con la interfaz gráfica al tratar de ejecutarlos desde SSH, se debe acceder a la Raspberry Pi mediante VNC Viewer. Una vez conectado en la terminal se deben ejecutar estos comandos
```bash 
cd ~/PF/Github/CEIA-PF/src/sift_rpi
source voenv/bin/activate
python main.py
```

Una vez hecho esto, se ejecutará el programa en la Raspberry Pi.

## Estados de avance de las versiones

[X] SIFT classic
[X] SIFT kornia
[X] SIFT Raspberry Pi 
