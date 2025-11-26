# utils.py
import torch

from config import CALIB_PATH, CAMERA_ID

def read_calibration_file(calib_file_path: str) -> list:
    """
    Lee el archivo de calibración y devuelve sus líneas
    
    Args:
        calib_file_path (str): Ruta al archivo de calibración

    Returns:
        list: Líneas del archivo de calibración
    """
    with open(calib_file_path, 'r') as f:
        lines = f.readlines()
    return lines

def extract_intrinsic_matrix(calib_lines: list, device: torch.device, camera_id: str = CAMERA_ID) -> torch.Tensor:
    """
    Extrae la matriz intrínseca K de las líneas del archivo de calibración
    
    Args:
        calib_lines (list): Líneas del archivo de calibración
        device (torch.device): Dispositivo donde crear el tensor
        camera_id (str): ID de la cámara a extraer (por defecto 'P0')
    
    Returns:
        torch.Tensor: Matriz intrínseca K de la cámara como tensor
    """
    for line in calib_lines:
        if line.startswith(camera_id):
            values = line.strip().split()[1:]
            values = [float(val) for val in values]
            P = torch.tensor(values, device=device, dtype=torch.float32).reshape(3, 4)
            K = P[:3, :3]
            return K
    return None

def main():
    calib_file_path = CALIB_PATH
    calib_lines = read_calibration_file(calib_file_path)
    intrinsic_matrix = extract_intrinsic_matrix(calib_lines, camera_id = CAMERA_ID)

    if intrinsic_matrix is not None:
        print("Intrinsic Matrix (K):")
        print(intrinsic_matrix)
    else:
        print("Intrinsic matrix not found for the specified camera ID.")

if __name__ == "__main__":
    main()
