# main.py
import sys
import cv2
import glob
import time
import torch
import kornia
import os
import signal
import numpy as np

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from features import Frame, match_frames, add_ones
from pointmap import Map
from display import Display
from utils import read_calibration_file, extract_intrinsic_matrix
from config import WIDTH, HEIGHT, CALIB_PATH, IMG_PATH, CAMERA_ID, ENABLE_LOGGING, OUTPUT_DIR, MAX_FRAMES
from src.slam.utils.benchmark_logger import BenchmarkLogger

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='kornia')

# Device
device = kornia.utils.get_mps_device_if_available()

# Umbrales de movimiento
MIN_TRANSLATION = 0.05
MAX_TRANSLATION = 2.0

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
        self.start_time = time.time()
        self.image_files = img_files
        self.frame_idx = 0
        self.map = Map(device)
        self.display = display

        # Cargar calibración
        calib_lines = read_calibration_file(CALIB_PATH)
        self.K = extract_intrinsic_matrix(calib_lines, device, CAMERA_ID)
        
        # Logger (si está habilitado)
        self.logger = BenchmarkLogger("sift_kornia") if ENABLE_LOGGING else None

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)
    
    def export_logger(self):
        """Exporta el logger y guarda análisis cualitativo (imágenes) + trayectoria"""
        if self.logger is not None:
            import os
            output_file = os.path.join(OUTPUT_DIR, "sift_kornia.json")
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
                "implementation": "sift_kornia",
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
        output_file = os.path.join(output_dir, "sift_kornia_trajectory.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Trayectoria estimada guardada: {output_file}")

    def update_frame(self):
        # Detener al terminar o llegar a MAX_FRAMES
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

        # Leer imagen
        img = cv2.imread(self.image_files[self.frame_idx])
        if img is None:
            self.frame_idx += 1
            return

        img = cv2.resize(img, (WIDTH, HEIGHT))
        
        # Convertir a tensor Kornia
        img_tensor = kornia.image_to_tensor(img, keepdim=False).float() / 255.0
        img_tensor = kornia.color.rgb_to_grayscale(img_tensor)
        img_tensor = img_tensor.to(device)
        
        # Crear frame
        frame = Frame(self.map, img_tensor, self.K, device)
        
        num_matches = 0
        if frame.id > 0:
            f1, f2 = self.map.frames[-1], self.map.frames[-2]
            
            try:
                idx1, idx2, Rt = match_frames(f1, f2)
                num_matches = len(idx1)
                
                # Validar traslación
                tnorm = torch.norm(Rt[:3, 3]).item()
                
                if tnorm < MIN_TRANSLATION:
                    Rt[:3, 3] = 0.0
                
                if tnorm > MAX_TRANSLATION:
                    print(f"[WARN] Traslación muy grande ({tnorm:.2f})")
                    self.frame_idx += 1
                    return
                
                # Actualizar pose
                f1.pose = torch.matmul(f2.pose, Rt)
                
                # Triangulación
                pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
                pts4d = pts4d / pts4d[:, 3:4]
                
                # Filtrar puntos válidos
                valid_mask = filter_points_behind_camera(pts4d, f1.pose)
                valid_points = pts4d[valid_mask]
                
                # Añadir al mapa
                for i, point_3d in enumerate(valid_points):
                    original_idx1 = idx1[valid_mask][i]
                    original_idx2 = idx2[valid_mask][i]
                    
                    map_point = self.map.add_or_update_point(
                        point_3d, f1, original_idx1.item()
                    )
                    map_point.add_observation(f2, original_idx2.item())

                # Actualizar trayectoria
                t = f1.pose[:3, 3]
                self.display.update_trajectory(t[0], t[2])
                
                # Actualizar mapa
                good_map_points = [p for p in self.map.points if p.is_good()]
                self.display.update_map_visualization(good_map_points)
                
                # Métricas
                self.display.update_metrics(frame.id + 1, len(self.map.points))
                
            except Exception as e:
                print(f"[WARN] SLAM falló en frame {frame.id}: {e}")
        
        # Visualizar frame
        if frame.id > 0:
            self.display.update_frame_display(img, frame.kps)
        else:
            self.display.update_frame_display(img, torch.empty((0, 2)))
        
        # Logging de benchmark
        if self.logger is not None:
            frame_elapsed_ms = (time.perf_counter() - frame_start_time) * 1000
            self.logger.log_frame(frame.id, num_matches, frame_elapsed_ms)

        self.frame_idx += 1

        # Métricas de tiempo
        elapsed_total = time.time() - self.start_time
        mins = int(elapsed_total // 60)
        secs = int(elapsed_total % 60)
        
        self.display.update_metrics(
            frame.id + 1, 
            len(self.map.points),
            f"{mins}:{secs:02d}"  # Formato MM:SS
        )


def triangulate(pose1, pose2, pts1, pts2):
    """Triangulación de puntos 3D (tensores)"""
    N = pts1.shape[0]
    ret = torch.zeros((N, 4), dtype=torch.float32, device=pts1.device)
    pose1_inv = torch.inverse(pose1)
    pose2_inv = torch.inverse(pose2)
    
    pts1_h = add_ones(pts1)
    pts2_h = add_ones(pts2)
    
    A_batch = torch.zeros((N, 4, 4), device=pts1.device)
    A_batch[:, 0] = pts1_h[:, 0:1] * pose1_inv[2:3] - pose1_inv[0:1]
    A_batch[:, 1] = pts1_h[:, 1:2] * pose1_inv[2:3] - pose1_inv[1:2]
    A_batch[:, 2] = pts2_h[:, 0:1] * pose2_inv[2:3] - pose2_inv[0:1]
    A_batch[:, 3] = pts2_h[:, 1:2] * pose2_inv[2:3] - pose2_inv[1:2]
    
    _, _, V = torch.svd(A_batch)
    ret = V[:, :, -1]
    
    return ret


def filter_points_behind_camera(points, cam_pose, 
                                z_min=0.1, z_max=30.0, xy_thresh=15.0):
    """Filtra puntos detrás de la cámara"""
    if len(points) == 0:
        return torch.empty(0, dtype=torch.bool, device=points.device)
    
    inv_pose = torch.inverse(cam_pose)
    points_cam = torch.mm(points, inv_pose.T)
    
    valid_mask = (
        (points_cam[:, 2] > z_min) & 
        (points_cam[:, 2] < z_max) &
        (torch.abs(points_cam[:, 0]) < xy_thresh) &
        (torch.abs(points_cam[:, 1]) < xy_thresh)
    )
    
    return valid_mask


if __name__ == "__main__":
    files = sorted(glob.glob(IMG_PATH))
    
    # Limitar cantidad de frames si MAX_FRAMES está configurado
    if MAX_FRAMES is not None and MAX_FRAMES > 0:
        files = files[:MAX_FRAMES]
        print(f"[Info] Limitado a {MAX_FRAMES} frames de {len(sorted(glob.glob(IMG_PATH)))}")
    
    app = QApplication(sys.argv)
    
    display = Display(WIDTH, HEIGHT)
    display.show()
    
    # Crear sistema SLAM y asignarlo a la variable global
    _slam_instance = VisualSLAM(files, display)
    
    # Conectar callback de cierre de display para exportar logger
    display.set_on_close_callback(_slam_instance.export_logger)
    
    sys.exit(app.exec_())