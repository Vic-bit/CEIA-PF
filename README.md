# Visual SLAM Monocular — SIFT Classic vs SIFT Kornia - ORB (Raspberry Pi)

**Proyecto Final de Especialización — CEIA**  
Autor: Víctor David Silva

---

## Descripción

Sistema de **Visual SLAM monocular** que compara dos implementaciones de extracción de características sobre secuencias del dataset KITTI, con una tercera implementación sobre hardware real:

- **SIFT Classic**: Implementación tradicional vía OpenCV (`cv2.SIFT`)
- **SIFT Kornia**: Implementación optimizada para GPU (compatible con MPS en Apple Silicon)
- **ORB RPi**: Implementación con ORB sobre Raspberry Pi 5 con cámara y robot móvil de 2 ruedas motoras

El sistema estima la trayectoria de cámara y reconstruye un mapa 3D a partir de imágenes monoculares, alineando los resultados contra el ground truth mediante transformación Sim(3) (7-DoF).

---

## Estructura del Proyecto

```
CEIA-PF/
├── src/
│   ├── slam/
│   │   ├── sift_classic/          # Implementación OpenCV SIFT
│   │   │   ├── main.py
│   │   │   ├── features.py
│   │   │   ├── pointmap.py
│   │   │   ├── display.py
│   │   │   ├── utils.py
│   │   │   └── config.py
│   │   ├── sift_kornia/           # Implementación Kornia (GPU)
│   │   │   ├── main.py
│   │   │   ├── features.py
│   │   │   ├── pointmap.py
│   │   │   ├── display.py
│   │   │   ├── utils.py
│   │   │   └── config.py
│   │   └── orb_rpi/               # Implementación ORB sobre Raspberry Pi
│   │       ├── main.py
│   │       ├── features.py
│   │       ├── pointmap.py
│   │       ├── display.py
│   │       ├── utils.py
│   │       ├── config.py
│   │       ├── camera.py          # Interfaz Picamera2
│   │       ├── camera_calib.py    # Calibración con tablero de ajedrez
│   │       ├── motor_controller.py # Control PWM/GPIO de motores DC
│   │       └── calibration/
│   │           └── calibration.npz
│   ├── ground_truth/
│   │   ├── generate_ground_truth.py
│   │   └── ground_truth.py
│   ├── evaluation/
│   │   ├── evaluate_slam.py
│   │   ├── alignment.py           # Alineación Sim(3) — Umeyama
│   │   └── trajectory.py          # Clase TrajectoryComparison
│   └── utils/
│       └── benchmark_logger.py
├── notebooks/
│   ├── EDA.ipynb
│   ├── EDA_kornia.ipynb
├── dataset/00/                    # Dataset KITTI (no incluido en el repo)
│   ├── calib.txt
│   ├── poses.txt
│   ├── times.txt
│   └── image_0/
├── outputs/benchmarks/            # Resultados generados
├── pyproject.toml
├── requirements.txt
└── requirements_rpi.txt           # Dependencias exclusivas de Raspberry Pi
```

---

## Requisitos

### PC (SIFT Classic y SIFT Kornia)

- Python 3.8+
- OpenCV (clásico y contrib)
- PyQt5 + pyqtgraph
- NumPy, SciPy, Matplotlib
- PyTorch
- Kornia

El dataset KITTI (secuencia 00) debe descargarse por separado desde [cvlibs.net](https://www.cvlibs.net/datasets/kitti/) y colocarse en `dataset/00/`.

### Raspberry Pi (ORB RPi)

- Raspberry Pi 5 con Raspberry Pi OS
- Cámara compatible con Picamera2
- Python 3.8+
- OpenCV
- PyQt5 + pyqtgraph
- Picamera2, rpi-hardware-pwm, gpiod

---

## Instalación

### PC

```bash
git clone https://github.com/tu-usuario/CEIA-PF
cd CEIA-PF
pip install -r requirements.txt
pip install -e .
```

### Raspberry Pi

```bash
git clone https://github.com/tu-usuario/CEIA-PF
cd CEIA-PF
pip install -r requirements_rpi.txt
pip install -e .
```

El comando `pip install -e .` registra el paquete `src` en el entorno Python, permitiendo ejecutar cualquier script desde la raíz del proyecto sin manipular paths manualmente.

---

## Ejecución

Todos los comandos deben ejecutarse desde la **raíz del proyecto** (`CEIA-PF/`).

---

### SIFT Classic y SIFT Kornia (PC)

**Paso 1 — Generar Ground Truth**

```bash
python src/ground_truth/generate_ground_truth.py
```

Output: `outputs/benchmarks/ground_truth_trajectory.json`

---

**Paso 2 — Ejecutar SIFT Classic**

```bash
python src/slam/sift_classic/main.py
```

Se abre una GUI interactiva con la trayectoria estimada y el mapa 3D en tiempo real. Al cerrar la ventana se guardan los resultados automáticamente. Se puede cerrar presionando la tecla q, pero se recomienda dejarlo hasta que se cierre en el número de frame máximo configurado.

Output: `outputs/benchmarks/sift_classic_trajectory.json`

---

**Paso 3 — Ejecutar SIFT Kornia**

```bash
python src/slam/sift_kornia/main.py
```

Idéntico al paso anterior. Si hay GPU disponible (CUDA o MPS), se utiliza automáticamente.

Output: `outputs/benchmarks/sift_kornia_trajectory.json`

---

**Paso 4 — Evaluar y comparar trayectorias**

```bash
python src/evaluation/evaluate_slam.py
```

Alinea ambas trayectorias contra el ground truth mediante Sim(3), calcula métricas ATE y RPE, y genera gráficos comparativos.

Outputs:
- `outputs/benchmarks/trajectory_comparison_aligned.png`
- `outputs/benchmarks/trajectory_evaluation_sim3.json`

---

### ORB RPi (Raspberry Pi)

**Paso 1 — Calibrar la cámara** (solo la primera vez)

Colocar imágenes del tablero de ajedrez en `src/slam/orb_rpi/calibration/` y ejecutar:

```bash
python src/slam/orb_rpi/camera_calib.py
```

Output: `src/slam/orb_rpi/calibration/calibration.npz`

---

**Paso 2 — Ejecutar el sistema**

```bash
python src/slam/orb_rpi/main.py
```

Se abre una GUI con visualización de video en tiempo real, trayectoria 2D (plano XZ) y controles de movimiento del robot (WASD o botones en pantalla).

---

## Configuración

Los parámetros principales se encuentran en el `config.py` de cada implementación.

**PC** (`src/slam/sift_classic/config.py` y `sift_kornia/config.py`):

```python
MAX_FRAMES = 250
MATCHER_TYPE = "BruteForce"
RANSAC_THRESHOLD = 1.0
TRIANGULATION_MIN_DEPTH = 0.1
```

**Raspberry Pi** (`src/slam/orb_rpi/config.py`):

```python
SIFT_N_FEATURES = 250    # Features ORB por frame
WIDTH, HEIGHT = 320, 240 # Resolución de captura
SKIP_RATE = 2            # Procesar 1 de cada N frames
```

---

## Métricas de Evaluación

**ATE (Absolute Trajectory Error):** diferencia euclidiana entre poses estimadas y ground truth. Unidad: metros. Valores < 0.1 m se consideran precisos.

**RPE (Relative Pose Error):** error de movimiento relativo entre frames consecutivos (ventana de 5 frames). Captura la consistencia local de la odometría.

**Factor de escala (s):** escala uniforme aplicada en la alineación Sim(3). Un valor cercano a 1.0 indica que el sistema estimó la escala correctamente.

> **Nota sobre la implementación RPi:** No se calculan métricas ATE/RPE para esta variante.

---

## Referencias

- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [SIFT — Lowe (2004)](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [Kornia](https://kornia.readtedia.io/)
- [Sim(3) Alignment — Umeyama (1991)](https://ieeexplore.ieee.org/document/70844)
- [ORB — Rublee et al. (2011)](https://ieeexplore.ieee.org/document/6126544)