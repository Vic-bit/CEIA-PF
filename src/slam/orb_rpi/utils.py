# utils.py
import numpy as np
import os
from config import CALIB_PATH

def load_calibration_npz(calib_file_path: str) -> np.ndarray:
    """
    Carga la matriz intrínseca K desde el archivo de calibración .npz
    generado por camera_calib.py
    
    Args:
        calib_file_path (str): Ruta al archivo calibration.npz

    Returns:
        np.ndarray: Matriz intrínseca K de 3x3
    """
    if not os.path.exists(calib_file_path):
        raise FileNotFoundError(f"Archivo de calibración no encontrado: {calib_file_path}")
    
    calib_data = np.load(calib_file_path)
    K = calib_data['camMatrix'][:3, :3] if calib_data['camMatrix'].shape[1] == 4 else calib_data['camMatrix']
    return K


def get_intrinsic_matrix_from_npz(calib_dir: str = None) -> np.ndarray:
    """
    Obtiene la matriz intrínseca K desde el archivo calibration.npz
    en el directorio especificado (o desde CALIB_PATH del config)
    
    Args:
        calib_dir (str): Ruta al directorio con calibration.npz. 
                        Si es None, usa CALIB_PATH del config

    Returns:
        np.ndarray: Matriz intrínseca K de 3x3
    """
    if calib_dir is None:
        calib_dir = CALIB_PATH
    
    calib_path = os.path.join(calib_dir, 'calibration.npz')
    return load_calibration_npz(calib_path)

def main():
    try:
        K = get_intrinsic_matrix_from_npz()
        print("Intrinsic Matrix (K) desde calibration.npz:")
        print(K)
    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
