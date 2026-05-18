import os
from pathlib import Path

# Calcular paths basados en la ubicación de este archivo
_CONFIG_DIR = Path(__file__).parent  # directorio de config.py (src/sift_kornia/)
_PROJECT_ROOT = _CONFIG_DIR.parent.parent.parent  # raíz del proyecto (CEIA-PF/)

SIFT_N_FEATURES = 150
WIDTH = 1241
HEIGHT = 376
CAMERA_ID = 'P0'

# Path (absolutas para funcionar desde cualquier directorio de ejecución)
CALIB_PATH = str(_PROJECT_ROOT / "dataset" / "00" / "calib.txt")
IMG_PATH = str(_PROJECT_ROOT / "dataset" / "00" / "image_0" / "*.png")

# Limit frames for development/testing (None = use all)
MAX_FRAMES = 250  # Set to None to process all frames

# Logging
ENABLE_LOGGING = True
OUTPUT_DIR = str(_PROJECT_ROOT / "outputs" / "benchmarks")

