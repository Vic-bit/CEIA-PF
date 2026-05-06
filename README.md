# Visual SLAM Monocular - Comparación SIFT Classic vs Kornia

Proyecto Final de Estudios - CEIA  
Autor: Víctor David Silva

⏰ **Si retomaste después de tiempo:** Ve a [🚀 Guía de Ejecución](#-guía-de-ejecución---paso-a-paso)

## 📋 Descripción

Sistema de **Visual SLAM** (Simultaneous Localization and Mapping) monocular que compara dos implementaciones de extracción de características:
- **SIFT Classic**: Usando OpenCV tradicional (`cv2.SIFT()`)
- **SIFT Kornia**: Usando Kornia (GPU-optimizado, soporta MPS en Apple Silicon)

El proyecto estima la **trayectoria de cámara** y reconstruye un **mapa 3D** a partir de secuencias de imágenes KITTI, alineando resultados contra ground truth mediante transformación Sim(3) (7-DoF).

## 🗂️ Estructura del Proyecto

```
src/
├── analysis/                  # ✨ NUEVO: Módulo reutilizable
│   ├── __init__.py           # Imports limpios
│   ├── ground_truth.py       # Clase GroundTruthAnalyzer
│   └── trajectory.py         # Clase TrajectoryComparison + helpers
│
├── sift_classic/              # Implementación con OpenCV SIFT clásico
│   ├── main.py               # Script de ejecución con GUI Qt5
│   ├── features.py           # Extracción y emparejamiento SIFT
│   ├── pointmap.py           # Mapa de puntos 3D
│   ├── display.py            # Visualización interactiva
│   ├── utils.py              # Utilidades de calibración
│   └── config.py             # Parámetros (MAX_FRAMES=250)
│
├── sift_kornia/               # Implementación con Kornia (GPU)
│   ├── main.py
│   ├── features.py
│   ├── pointmap.py
│   ├── display.py
│   ├── utils.py
│   └── config.py
│
├── sift_kornia_copy/          # Experimental (backup, no usar)
│
└── alignment/                    # Módulos reutilizables
    ├── __init__.py
    └── alignment.py           # Alineación Sim(3) (reutilizable)

scripts/                       # ✨ NUEVO: Scripts ejecutables
├── __init__.py
├── run_ground_truth.py        # Generar GT desde KITTI
└── evaluate_odometry.py       # Evaluar trayectorias SLAM

notebooks/                     # Análisis exploratorio
├── EDA.ipynb
├── EDA_kornia.ipynb
└── Benchmark_Analysis_SIFT_Classic_vs_Kornia.ipynb

dataset/00/                    # KITTI dataset
├── calib.txt                 # Calibración de cámara
├── poses.txt                 # Ground truth (4×4 poses)
├── times.txt                 # Timestamps
└── image_0/                  # Imágenes

outputs/benchmarks/            # Resultados
├── ground_truth_trajectory.json
├── sift_classic_trajectory.json
├── sift_kornia_trajectory.json
├── sift_classic.json         # Métricas de performance
├── sift_kornia.json
├── trajectory_comparison_aligned.png
├── trajectory_evaluation_sim3.json
└── qualitative_analysis/

main.py                        # ✨ NUEVO: Orquestador principal
requirements.txt               # Dependencias con versiones exactas
README.md                      # Este archivo
```

**¿Qué cambió?**
- ✨ **`src/analysis/`**: Módulo reutilizable (antes eran archivos sueltos en raíz)
- ✨ **`scripts/`**: Scripts ejecutables (antes eran archivos en raíz)
- ✨ **`main.py`**: Orquestador (point of entry único)
- ✨ **Cleaner imports**: Todos los módulos son importables desde `src`

## 📦 Requisitos

- Python 3.8+
- OpenCV (clásico y contrib)
- PyQt5 + pyqtgraph
- NumPy, SciPy, Matplotlib
- PyTorch
- Kornia

Instala todas las dependencias con versiones exactas:

```bash
pip install -r requirements.txt
```

## 🚀 Guía de Ejecución - Paso a Paso

### **OPCIÓN A: Pipeline Completo (Recomendado para empezar)**

Ejecuta TODO en una sola línea (abre 2 GUIs interactivas):

```bash
python main.py full
```

**Esto hace:**
1. Genera ground truth
2. Abre GUI de SIFT Classic (haz clic para interactuar, cierra para guardar)
3. Abre GUI de SIFT Kornia  (haz clic para interactuar, cierra para guardar)
4. Compara resultados automáticamente
5. Genera gráficos y reportes

**Outputes:**
- `outputs/benchmarks/trajectory_comparison_aligned.png`
- `outputs/benchmarks/trajectory_evaluation_sim3.json`

---

### **OPCIÓN B: Paso a Paso Manual**

Si prefieres más control, ejecuta cada paso:

#### **PASO 1: Generar Ground Truth**

```bash
python scripts/run_ground_truth.py
```

**Output:**
- `outputs/benchmarks/ground_truth_trajectory.json` ← truth
- Información: 250 frames (configurable en `src/sift_classic/config.py`)

#### **PASO 2: Ejecutar SIFT Classic (GUI Interactiva)**

```bash
python src/sift_classic/main.py
```

**Interfaz:**
- Visualización en tiempo real de trayectoria
- Puntos 3D detectados en el mapa
- **Al cerrar la ventana**: guarda logs automáticamente

**Outputs:**
- `outputs/benchmarks/sift_classic_trajectory.json` ← trayectoria estimada
- `outputs/benchmarks/sift_classic.json` ← métricas (FPS, matches, tiempo)

#### **PASO 3: Ejecutar SIFT Kornia (GPU - GUI Interactiva)**

```bash
python src/sift_kornia/main.py
```

**Interfaz:** Idéntica a SIFT Classic (pero puede usar GPU/MPS)

**Outputs:**
- `outputs/benchmarks/sift_kornia_trajectory.json`
- `outputs/benchmarks/sift_kornia.json` ← métricas GPU

#### **PASO 4: Evaluar y Comparar**

```bash
python scripts/evaluate_odometry.py
```

**Qué hace:**
1. Carga: Ground Truth + ambas trayectorias SLAM
2. Alinea ambas con Sim(3) contra GT (ajusta escala, rotación, traslación)
3. Calcula:
   - **ATE** (Absolute Trajectory Error)
   - **RPE** (Relative Pose Error)
   - Factor de escala (s)
4. Genera gráficos de comparación

**Outputs:**
- `outputs/benchmarks/trajectory_comparison_aligned.png` ← gráficos
- `outputs/benchmarks/trajectory_evaluation_sim3.json` ← resultados detallados

#### **PASO 5: Análisis Interactivo (Notebook - Opcional)**

```bash
jupyter notebook notebooks/Benchmark_Analysis_SIFT_Classic_vs_Kornia.ipynb
```

**4 Secciones Temáticas:**
1. **Carga de Datos**: Lee JSONs + visualiza metadata
2. **Trayectorias**: Superpone GT vs SLAM alineadas
3. **Análisis de Errores**: Distribuciones ATE/RPE
4. **Métricas de Performance**: FPS, RAM, matches

---

## ⚙️ Configuración

**Edita `src/sift_classic/config.py` y `src/sift_kornia/config.py`:**

```python
MAX_FRAMES = 250           # Número de frames a procesar
MATCHER_TYPE = "BruteForce"  # O "FLANN"
RANSAC_THRESHOLD = 1.0    # Threshold para RANSAC
TRIANGULATION_MIN_DEPTH = 0.1  # Profundidad mínima aceptada
```

## 🔍 Interpretación de Resultados

### ATE (Absolute Trajectory Error)
- Diferencia euclidiana entre poses estimadas vs ground truth
- **Unidad**: metros
- **Bueno**: < 0.1m (muy preciso), < 1m (aceptable)

### RPE (Relative Pose Error)
- Error de movimiento relativo entre frames consecutivos (Δ5 frames)
- Captura consistencia de odometría
- **Unidad**: metros

### Scale Factor (s)
- Factor de escala uniforme aplicado: `trayectoria_slam × s ≈ metros_reales`
- Si s ≈ 1.0 → SLAM detectó escala correctamente
- Si s >> 1 → SLAM reportó distancias muy pequeñas

## � Debugging & Troubleshooting

**Si `scripts/evaluate_odometry.py` falla:**
1. ✓ Verifica que exista `outputs/benchmarks/ground_truth_trajectory.json`
   - Si no: ejecuta `python scripts/run_ground_truth.py`
2. ✓ Verifica que existan `sift_classic_trajectory.json` y `sift_kornia_trajectory.json`
   - Si no: ejecuta los main.py respectivos

**Si GUI de SIFT no carga:**
- Verifica PyQt5: `python -c "import PyQt5; print(PyQt5.__version__)"`
- Verifica calib.txt existe: `ls dataset/00/calib.txt`

**Si Kornia no usa GPU:**
- Verificá: `python -c "import torch; print(torch.backends.mps.is_available())"` (Apple Silicon)
- O `torch.cuda.is_available()` (NVIDIA)

**Si imports fallan:**
- `from src.analysis import GroundTruthAnalyzer` debe funcionar
- `from src.alignment.alignment import align_sim3_umeyama` debe funcionar
- Si no funciona: verifica que estés en la carpeta raíz del proyecto

## 📖 Versiones de Dependencias

ver [requirements.txt](requirements.txt) para versiones exactas (distancia en 2+ meses)

## 🔗 Referencias

- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [SIFT Paper (Lowe 2004)](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [OpenCV SIFT](https://docs.opencv.org/latest/da/df5/tutorial_py_sift_intro.html)
- [Kornia Framework](https://kornia.readthedocs.io/)
- [Sim(3) Alignment (Umeyama 1991)](https://ieeexplore.ieee.org/document/70844)

## 👤 Contacto

Víctor David Silva - CEIA

## Instalación
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
