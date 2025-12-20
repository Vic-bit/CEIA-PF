# camera.py

from picamera2 import Picamera2
from time import sleep
from config import WIDTH, HEIGHT


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
    
    def capture_array(self):
        """Captura un frame como array RGB."""
        return self.picam2.capture_array()
    
    def close(self):
        """Cierra la cámara."""
        self.picam2.close()