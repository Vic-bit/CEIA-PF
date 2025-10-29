# Desarrollo de un sistema de posicionamiento y localización mediante odometría monocular

Proyecto Final de Estudios - CEIA  
Autor: Víctor David Silva
Fecha: Octubre 2025

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
- `dataset/`: Secuencias de imágenes y archivos de calibración (formato KITTI).
- `notebooks/`: Análisis exploratorio y pruebas.
- `outputs/`: Resultados y visualizaciones generadas.

## Versiones

El proyecto cuenta con cuatro versiones, cada una explorando diferentes técnicas de extracción de características y frameworks.  

Las 4 versiones a desarrollar son:
- SIFT classic
- SIFT kornia
- SIFT classic en Raspberry Pi 
- SIFT kornia en Raspberry Pi

## Requisitos

- Python 3.8+
- OpenCV (`opencv-python`)
- PyQt5
- pyqtgraph
- numpy

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Ejecución

1. Asegúrate de tener el dataset en la ruta `dataset/00/` con imágenes y archivo de calibración. El cual se obtiene de https://www.cvlibs.net/datasets/kitti/
2. Modifica los parámetros en `src/sift_classic/config.py` si es necesario.
3. Ejecuta el sistema desde la carpeta raíz del proyecto:

```bash
python src/sift_classic/main.py
```

Se abrirá una interfaz gráfica mostrando la imagen actual, la trayectoria estimada y los puntos 3D reconstruidos.

## Estados de avance de las versiones

[X] SIFT classic
[ ] SIFT kornia
[ ] SIFT classic en Raspberry Pi 
[ ] SIFT kornia en Raspberry Pi
