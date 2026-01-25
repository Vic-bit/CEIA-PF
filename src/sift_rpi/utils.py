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
    en el directorio especificado (o del mismo directorio de este archivo)
    
    Args:
        calib_dir (str): Ruta al directorio con calibration.npz. 
                        Si es None, usa el directorio de este archivo

    Returns:
        np.ndarray: Matriz intrínseca K de 3x3
    """
    if calib_dir is None:
        calib_dir = os.path.dirname(os.path.abspath(__file__))
    
    calib_path = os.path.join(calib_dir, 'calibration/calibration.npz')
    return load_calibration_npz(calib_path)


def read_calibration_file(calib_file_path: str) -> list:
    """
    Lee un archivo de calibración en formato texto (ej: KITTI format)
    y devuelve sus líneas
    
    Args:
        calib_file_path (str): Ruta al archivo de calibración

    Returns:
        list: Líneas del archivo de calibración
    """
    with open(calib_file_path, 'r') as f:
        lines = f.readlines()
    return lines


def extract_intrinsic_matrix(calib_lines: list, camera_id: str = "P0") -> np.ndarray:
    """
    Extrae la matriz intrínseca K de las líneas de un archivo de calibración
    en formato KITTI (ej: P0: fx 0 cx 0 fy cy 0 0 1 0)
    
    Args:
        calib_lines (list): Líneas del archivo de calibración
        camera_id (str): ID de la cámara a extraer (por defecto 'P0')
        
    Returns:
        np.ndarray: Matriz intrínseca K de 3x3
    """
    for line in calib_lines:
        if line.startswith(camera_id):
            # Separa la cámara y la convierte en float
            values = line.strip().split()[1:]
            values = [float(val) for val in values]
            P = np.array(values).reshape(3, 4)
            K = P[:3, :3]
            return K
    return None


def main():
    # Ejemplo 1: Leer calibración desde archivo .npz (camera_calib.py)
    try:
        K = get_intrinsic_matrix_from_npz()
        print("Intrinsic Matrix (K) desde calibration.npz:")
        print(K)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Ejemplo 2: Leer calibración desde archivo de texto (KITTI format)
    # calib_file_path = CALIB_PATH
    # calib_lines = read_calibration_file(calib_file_path)
    # intrinsic_matrix = extract_intrinsic_matrix(calib_lines, camera_id="P0")
    # if intrinsic_matrix is not None:
    #     print("Intrinsic Matrix (K) desde archivo KITTI:")
    #     print(intrinsic_matrix)


if __name__ == "__main__":
    main()
