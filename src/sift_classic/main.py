# main.py
import sys
import cv2
import glob
import numpy as np
from features import Frame, match_frames, add_ones
from pointmap import Map
from display import Display
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from utils import read_calibration_file, extract_intrinsic_matrix
from config import (WIDTH, HEIGHT, CALIB_PATH, IMG_PATH, CAMERA_ID)


class VisualSLAM:
    def __init__(self, img_files, display):
        self.image_files = img_files
        self.frame_idx = 0
        self.map = Map()
        self.display = display

        calib_file_path = CALIB_PATH
        calib_lines = read_calibration_file(calib_file_path)
        self.K = extract_intrinsic_matrix(calib_lines, camera_id=CAMERA_ID)

        # Timer sin límite de fps
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)

    def update_frame(self):
        # Al procesar todas las imágenes se detiene el programa
        if self.frame_idx >= len(self.image_files):
            self.timer.stop()
            return

        # Si no encuentra la imagen, se salta el índice
        img = cv2.imread(self.image_files[self.frame_idx])
        if img is None:
            self.frame_idx += 1
            return

        #img = cv2.resize(img, (self.W, self.H))
        img = cv2.resize(img, (WIDTH, HEIGHT))
        frame = Frame(self.map, img, self.K)
        
        if frame.id > 0:
            f1, f2 = self.map.frames[-1], self.map.frames[-2]
            try:
                idx1, idx2, Rt = match_frames(f1, f2)

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
    app = QApplication(sys.argv)
    
    # Crear display
    display = Display(WIDTH, HEIGHT)
    display.show()
    
    # Crear sistema SLAM
    slam = VisualSLAM(files, display)
    
    sys.exit(app.exec_())