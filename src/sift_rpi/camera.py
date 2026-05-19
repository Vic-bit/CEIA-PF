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
        #self.K = np.array([[F, 0, WIDTH//2], [0, F, HEIGHT//2], [0, 0, 1]])
        print("[Camera] Matriz K calibrada cargada")

    def read(self):
        """Lee un frame y lo devuelve en formato OpenCV (BGR).
        Devuelve (ret, img) compatible con cv2.VideoCapture.
        """
        try:
            # Capturar como RGB
            rgb_array = self.picam2.capture_array()
            # Convertir RGB a BGR para OpenCV
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            return True, bgr_array
        except Exception as e:
            print(f"Error capturando frame: {e}")
            return False, None
    
    def capture_array(self):
        """Captura un frame como array RGB."""
        return self.picam2.capture_array()
    
    def close(self):
        """Cierra la cámara."""
        self.picam2.close()