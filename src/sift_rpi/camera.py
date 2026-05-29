# camera.py

from picamera2 import Picamera2
from time import sleep
import cv2
import numpy as np
from config import WIDTH, HEIGHT, F
from utils import get_intrinsic_matrix_from_npz


class Camera:
    """Wrapper para Picamera2."""
    
    def __init__(self):
        self.picam2 = Picamera2()
        cfg = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (WIDTH, HEIGHT)}
        )
        self.picam2.configure(cfg)
        self.picam2.start()
        sleep(1)  # Esperar a que la cámara se estabilice
        
        # Cargar matriz de intrínsecos calibrada
        self.K = get_intrinsic_matrix_from_npz()
        
        # Cargar coeficientes de distorsión
        from config import CALIB_PATH
        import os
        calib_file = os.path.join(CALIB_PATH, 'calibration.npz')
        if os.path.exists(calib_file):
            calib_data = np.load(calib_file)
            self.dist_coeffs = calib_data['distCoeff']
            print("[Camera] Coeficientes de distorsión cargados")
        else:
            print(f"[Camera] Advertencia: Archivo de calibración no encontrado en {calib_file}")
            self.dist_coeffs = np.zeros(5)  # Fallback: sin distorsión
        
        print("[Camera] Matriz K calibrada cargada")

    def read(self):
        """Lee un frame y lo devuelve en formato OpenCV (BGR).
        Aplica corrección de distorsión si está disponible.
        Devuelve (ret, img) compatible con cv2.VideoCapture.
        """
        try:
            # Capturar como RGB
            rgb_array = self.picam2.capture_array()
            # Convertir RGB a BGR para OpenCV
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            # Aplicar corrección de distorsión
            bgr_undistorted = cv2.undistort(bgr_array, self.K, self.dist_coeffs)
            return True, bgr_undistorted
        except Exception as e:
            print(f"Error capturando frame: {e}")
            return False, None
    
    def capture_array(self):
        """Captura un frame como array RGB."""
        return self.picam2.capture_array()
    
    def close(self):
        """Cierra la cámara."""
        self.picam2.close()