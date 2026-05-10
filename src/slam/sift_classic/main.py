# main.py
import sys
import cv2
import glob
import time
import numpy as np
import os
import signal

# Agregar directorio padre (src/) al path para importar benchmark_logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace('/sift_classic', ''))
# Agregar el directorio actual (sift_classic/) para importaciones locales
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import Frame, match_frames, add_ones
from pointmap import Map
from display import Display
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from utils import read_calibration_file, extract_intrinsic_matrix
from config import (WIDTH, HEIGHT, CALIB_PATH, IMG_PATH, CAMERA_ID, ENABLE_LOGGING, OUTPUT_DIR, MAX_FRAMES)
from src.slam.utils.benchmark_logger import BenchmarkLogger

# Variable global para almacenar la instancia SLAM
_slam_instance = None

def signal_handler(sig, frame):
    """Manejador para Ctrl+C que exporta el logger antes de salir"""
    global _slam_instance
    if _slam_instance:
        _slam_instance.export_logger()
        print("\n[Info] Benchmark exportado antes de salir (Ctrl+C).")
    sys.exit(0)

# Registrar el manejador de señales
signal.signal(signal.SIGINT, signal_handler)


class VisualSLAM:
    def __init__(self, img_files, display):
        self.image_files = img_files
        self.frame_idx = 0
        self.map = Map()
        self.display = display

        calib_lines = read_calibration_file(CALIB_PATH)
        self.K = extract_intrinsic_matrix(calib_lines, camera_id=CAMERA_ID)
        
        # Logger (si está habilitado)
        self.logger = BenchmarkLogger("sift_classic") if ENABLE_LOGGING else None

        # Timer sin límite de fps
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)
    
    def export_logger(self):
        """Exporta el logger y guarda análisis cualitativo (imágenes) + trayectoria"""
        if self.logger is not None:
            import os
            output_file = os.path.join(OUTPUT_DIR, "sift_classic.json")
            self.logger.export_summary(output_file)
        
        # Guardar imágenes para análisis cualitativo
        self.display.save_camera_frame(OUTPUT_DIR)
        self.display.save_trajectory_plot(OUTPUT_DIR)
        
        # Guardar trayectoria estimada para validación contra Ground Truth
        self._save_trajectory(OUTPUT_DIR)

    def _save_trajectory(self, output_dir):
        """Guarda la trayectoria estimada en formato JSON para validación"""
        import os
        import json
        from datetime import datetime
        
        # Obtener datos de trayectoria
        trajectory_data = self.display.get_trajectory_data()
        
        # Crear estructura de datos
        output_data = {
            "metadata": {
                "implementation": "sift_classic",
                "timestamp": datetime.now().isoformat(),
                "num_frames": len(trajectory_data['x'])
            },
            "trajectory": {
                "x": trajectory_data['x'],
                "z": trajectory_data['z']
            },
            "statistics": {
                "num_frames": len(trajectory_data['x']),
                "total_distance": float(np.sum(np.sqrt(np.diff(trajectory_data['x'])**2 + np.diff(trajectory_data['z'])**2)))
            }
        }
        
        # Guardar
        output_file = os.path.join(output_dir, "sift_classic_trajectory.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Trayectoria estimada guardada: {output_file}")

    def update_frame(self):
        # Al procesar todas las imágenes o llegar a MAX_FRAMES se detiene el programa
        if self.frame_idx >= len(self.image_files):
            self.timer.stop()
            self.export_logger()
            return
        
        # Limitar a MAX_FRAMES si está configurado
        if MAX_FRAMES is not None and self.frame_idx >= MAX_FRAMES:
            self.timer.stop()
            self.export_logger()
            return

        frame_start_time = time.perf_counter()

        # Si no encuentra la imagen, se salta el índice
        img = cv2.imread(self.image_files[self.frame_idx])
        if img is None:
            self.frame_idx += 1
            return

        #img = cv2.resize(img, (self.W, self.H))
        img = cv2.resize(img, (WIDTH, HEIGHT))
        frame = Frame(self.map, img, self.K)
        
        num_matches = 0
        if frame.id > 0:
            f1, f2 = self.map.frames[-1], self.map.frames[-2]
            try:
                idx1, idx2, Rt = match_frames(f1, f2)
                num_matches = len(idx1)

                f1.pose = np.dot(f2.pose, Rt)

                pts4d = triangulate(f1.pose, f2.pose,
                                    f1.pts[idx1], f2.pts[idx2])
                pts4d /= pts4d[:, 3:]

                valid_mask = filter_points_behind_camera(pts4d, f1.pose)
                valid_points = pts4d[valid_mask]

                # Agregar puntos válidos al mapa
                good_points = []
                for i, point_3d in enumerate(valid_points):
                    original_idx1 = idx1[valid_mask][i]
                    original_idx2 = idx2[valid_mask][i]
                    
                    map_point = self.map.add_or_update_point(
                        point_3d, f1, original_idx1
                    )
                    map_point.add_observation(f2, original_idx2)
                    
                    # Solo guardar si el punto es bueno
                    if map_point.is_good():
                        good_points.append(map_point)

                # Actualizar trayectoria
                t = f1.pose[:3, 3]
                self.display.update_trajectory(t[0], t[2])

                # Actualizar visualización del mapa
                good_map_points = [p for p in self.map.points if p.is_good()]
                self.display.update_map_visualization(good_map_points)
                
                # Actualizar métricas
                self.display.update_metrics(frame.id + 1, len(self.map.points))

            except Exception as e:
                print(f"[Warning] SLAM falló en frame {frame.id}: {e}")

        # Dibujar keypoints
        if frame.id > 0:
            self.display.update_frame_display(img, self.map.frames[-1].kps)
        else:
            # Primer frame sin keypoints previos
            self.display.update_frame_display(img, np.array([]))

        # Logging de benchmark
        if self.logger is not None:
            frame_elapsed_ms = (time.perf_counter() - frame_start_time) * 1000
            self.logger.log_frame(frame.id, num_matches, frame_elapsed_ms)

        self.frame_idx += 1


# Funciones auxiliares
def triangulate(pose1, pose2, pts1, pts2):
    """
    Triangula puntos 3D a partir de dos poses y puntos correspondientes en ambos frames.
    """
    ret = np.zeros((pts1.shape[0], 4))
    pose1_inv = np.linalg.inv(pose1)
    pose2_inv = np.linalg.inv(pose2)
    
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1_inv[2] - pose1_inv[0]
        A[1] = p[0][1] * pose1_inv[2] - pose1_inv[1]
        A[2] = p[1][0] * pose2_inv[2] - pose2_inv[0]
        A[3] = p[1][1] * pose2_inv[2] - pose2_inv[1]
        _, _, V = np.linalg.svd(A)
        ret[i] = V[3]
    return ret


def filter_points_behind_camera(points, 
                                cam_pose, 
                                z_min_threshold=0.1,
                                z_max_threshold=30.0,
                                xy_threshold=15.0):
    """
    Filtra puntos que están detrás de la cámara y devuelve máscara de puntos válidos
    """
    if len(points) == 0:
        return np.array([], dtype=bool)
    
    inv_pose = np.linalg.inv(cam_pose)
    valid_mask = []
    
    for point in points:
        # Transformar punto al sistema de coordenadas de la cámara
        point_cam = inv_pose.dot(point)
        # Verificar que esté delante de la cámara (z > threshold)
        # y que no esté demasiado lejos (filtro adicional)
        is_valid = (point_cam[2] > z_min_threshold and 
                   point_cam[2] < z_max_threshold and  # No muy lejos
                   abs(point_cam[0]) < xy_threshold and  # No muy a los lados
                   abs(point_cam[1]) < xy_threshold)    
        valid_mask.append(is_valid)
    
    return np.array(valid_mask, dtype=bool)


if __name__ == "__main__":
    files = sorted(glob.glob(IMG_PATH))
    
    # Limitar cantidad de frames si MAX_FRAMES está configurado
    if MAX_FRAMES is not None and MAX_FRAMES > 0:
        files = files[:MAX_FRAMES]
        print(f"[Info] Limitado a {MAX_FRAMES} frames de {len(glob.glob(IMG_PATH))}")
    
    app = QApplication(sys.argv)
    
    # Crear display
    display = Display(WIDTH, HEIGHT)
    display.show()
    
    # Crear sistema SLAM y asignarlo a la variable global
    _slam_instance = VisualSLAM(files, display)
    
    # Conectar callback de cierre de display para exportar logger
    display.set_on_close_callback(_slam_instance.export_logger)
    
    sys.exit(app.exec_())