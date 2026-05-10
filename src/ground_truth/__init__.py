# src/__init__.py
from pathlib import Path
import sys

def setup_path():
    """Agregar raíz del proyecto al path automáticamente."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

setup_path()