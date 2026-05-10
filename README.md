# Visual SLAM Monocular — SIFT Classic vs SIFT Kornia

**Proyecto Final de Especialización — CEIA**  
Autor: Víctor David Silva

---

## Descripción

Sistema de **Visual SLAM monocular** que compara dos implementaciones de extracción de características sobre secuencias del dataset KITTI:

- **SIFT Classic**: Implementación tradicional vía OpenCV (`cv2.SIFT`)
- **SIFT Kornia**: Implementación optimizada para GPU (compatible con MPS en Apple Silicon)

El sistema estima la trayectoria de cámara y reconstruye un mapa 3D a partir de imágenes monoculares, alineando los resultados contra el ground truth mediante transformación Sim(3) (7-DoF).

---

## Estructura del Proyecto

```
CEIA-PF/
├── src/
│   ├── analysis/                  # Módulo de análisis reutilizable
│   │   ├── __init__.py
│   │   ├── alignment.py           # Alineación Sim(3) — Umeyama
│   │   └── trajectory.py          # Clase TrajectoryComparison
│   ├── slam/
│   │   ├── sift_classic/          # Implementación OpenCV SIFT
│   │   │   ├── main.py
│   │   │   ├── features.py
│   │   │   ├── pointmap.py
│   │   │   ├── display.py
│   │   │   ├── utils.py
│   │   │   └── config.py
│   │   └── sift_kornia/           # Implementación Kornia (GPU)
│   │       ├── main.py
│   │       ├── features.py
│   │       ├── pointmap.py
│   │       ├── display.py
│   │       ├── utils.py
│   │       └── config.py
│   ├── ground_truth/
│   │   ├── generate_ground_truth.py
│   │   └── ground_truth.py
│   ├── evaluation/
│   │   └── evaluate_slam.py
│   └── utils/
│       └── benchmark_logger.py
├── notebooks/
│   ├── EDA.ipynb
│   ├── EDA_kornia.ipynb
│   └── Benchmark_Analysis_SIFT_Classic_vs_Kornia.ipynb
├── dataset/00/                    # Dataset KITTI (no incluido en el repo)
│   ├── calib.txt
│   ├── poses.txt
│   ├── times.txt
│   └── image_0/
├── outputs/benchmarks/            # Resultados generados
├── main.py                        # Orquestador principal
├── pyproject.toml
└── requirements.txt
```

---

## Requisitos

- Python 3.8+
- OpenCV (clásico y contrib)
- PyQt5 + pyqtgraph
- NumPy, SciPy, Matplotlib
- PyTorch
- Kornia

El dataset KITTI (secuencia 00) debe descargarse por separado desde [cvlibs.net](https://www.cvlibs.net/datasets/kitti/) y colocarse en `dataset/00/`.

---

## Instalación

```bash
git clone https://github.com/tu-usuario/CEIA-PF
cd CEIA-PF
pip install -r requirements.txt
pip install -e .
```

El comando `pip install -e .` registra el paquete `src` en el entorno Python, permitiendo ejecutar cualquier script desde la raíz del proyecto sin manipular paths manualmente.

---

## Ejecución

Todos los comandos deben ejecutarse desde la **raíz del proyecto** (`CEIA-PF/`).

### Pipeline completo (recomendado)

```bash
python main.py full
```

Ejecuta los 4 pasos en secuencia: ground truth → SIFT Classic → SIFT Kornia → evaluación.

---

### Ejecución paso a paso

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

Se abre una GUI interactiva con la trayectoria estimada y el mapa 3D en tiempo real. Al cerrar la ventana se guardan los resultados automáticamente.

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

## Configuración

Los parámetros principales se encuentran en `src/slam/sift_classic/config.py` y `src/slam/sift_kornia/config.py`:

```python
MAX_FRAMES = 250              # Frames a procesar
MATCHER_TYPE = "BruteForce"   # Alternativa: "FLANN"
RANSAC_THRESHOLD = 1.0
TRIANGULATION_MIN_DEPTH = 0.1
```

---

## Métricas de Evaluación

**ATE (Absolute Trajectory Error):** diferencia euclidiana entre poses estimadas y ground truth. Unidad: metros. Valores < 0.1 m se consideran precisos.

**RPE (Relative Pose Error):** error de movimiento relativo entre frames consecutivos (ventana de 5 frames). Captura la consistencia local de la odometría.

**Factor de escala (s):** escala uniforme aplicada en la alineación Sim(3). Un valor cercano a 1.0 indica que el sistema estimó la escala correctamente.

---

## Análisis en Notebook

```bash
jupyter notebook notebooks/Benchmark_Analysis_SIFT_Classic_vs_Kornia.ipynb
```

El notebook contiene 4 secciones: carga de datos, comparación de trayectorias, distribución de errores ATE/RPE, y métricas de performance (FPS, RAM, matches por frame).

---

## Referencias

- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [SIFT — Lowe (2004)](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [Kornia](https://kornia.readthedocs.io/)
- [Sim(3) Alignment — Umeyama (1991)](https://ieeexplore.ieee.org/document/70844)