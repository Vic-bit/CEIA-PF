# main.py - CORREGIDO para evitar pérdida de tracking
import sys
import signal
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from camera import Camera
from display import MainWindow
from features import Frame, match_frames, add_ones
from pointmap import Map
from motor_controller import MotorController
from config import WIDTH, HEIGHT, TIMER_INTERVAL_MS, SKIP_RATE, MIN_TRANSLATION

signal.signal(signal.SIGINT, lambda *args: sys.exit(0))
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))


class VisualSLAM:
    def __init__(self, camera, motor_ctrl, display):
        self.camera = camera
        self.motor_ctrl = motor_ctrl
        self.display = display
        self.map = Map()
        self.frame_skip_counter = 0
        
        # Usar matriz de intrínsecos de la cámara
        self.K = camera.K
        
        # NUEVO: Variables de estado para tracking
        self.last_good_pose = np.eye(4)
        self.frames_without_tracking = 0
        self.MAX_FRAMES_WITHOUT_TRACKING = 10

        # Timer para procesar frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(TIMER_INTERVAL_MS)

    def update_frame(self):
        """Procesa un frame para Visual Odometry."""
        ret, img = self.camera.read()
        if not ret:
            return

        # Frame skipping
        self.frame_skip_counter += 1
        if self.frame_skip_counter % SKIP_RATE != 0:
            return

        img = cv2.resize(img, (WIDTH, HEIGHT))
        frame = Frame(self.map, img, self.K)
        
        # Mostrar primer frame
        if frame.id == 0:
            self.display.update_frame_display(img, frame.kps)
            return
        
        # TRACKING
        f1, f2 = self.map.frames[-1], self.map.frames[-2]
        
        try:
            idx1, idx2, Rt = match_frames(f1, f2)

            # CASO 1: No hay suficientes matches
            if len(idx1) < 6:
                print(f"⚠️ Frame {frame.id}: Pocos matches ({len(idx1)})")
                self.frames_without_tracking += 1
                
                if self.frames_without_tracking > self.MAX_FRAMES_WITHOUT_TRACKING:
                    # Perdimos tracking completamente
                    print("🔴 TRACKING PERDIDO - Reiniciando")
                    f1.pose = self.last_good_pose.copy()
                else:
                    # Mantener última pose
                    f1.pose = f2.pose.copy()
                
                self.display.update_frame_display(img, frame.kps)
                self.display.update_metrics(frame.id + 1, len(self.map.points))
                return
            
            # CASO 2: Hay matches - validar movimiento
            translation_norm = np.linalg.norm(Rt[:3, 3])
            
            # Movimiento muy pequeño = ruido
            if translation_norm < MIN_TRANSLATION:
                print(f"⚠️ Frame {frame.id}: Movimiento mínimo ({translation_norm:.4f})")
                f1.pose = f2.pose.copy()
                self.display.update_frame_display(img, frame.kps)
                return
            
            # Movimiento muy grande = error
            if translation_norm > 2.0:  # NUEVO: detectar saltos
                print(f"🔴 Frame {frame.id}: Movimiento EXCESIVO ({translation_norm:.4f}) - IGNORANDO")
                f1.pose = f2.pose.copy()
                self.frames_without_tracking += 1
                self.display.update_frame_display(img, frame.kps)
                return
            
            # CASO 3: Validar rotación
            R = Rt[:3, :3]
            det_R = np.linalg.det(R)
            if abs(det_R - 1.0) > 0.1:
                print(f"🔴 Frame {frame.id}: Rotación inválida (det={det_R:.4f})")
                f1.pose = f2.pose.copy()
                self.frames_without_tracking += 1
                self.display.update_frame_display(img, frame.kps)
                return
            
            # ✅ TRACKING EXITOSO
            f1.pose = f2.pose.dot(Rt)  # Actualizar pose
            self.last_good_pose = f1.pose.copy()  # Guardar como buena
            self.frames_without_tracking = 0  # Reset contador
            
            # Triangular puntos
            pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
            pts4d /= pts4d[:, 3:]

            valid_mask = filter_points_behind_camera(pts4d, f1.pose)
            valid_points = pts4d[valid_mask]

            # Agregar puntos al mapa
            for i, point_3d in enumerate(valid_points):
                original_idx1 = idx1[valid_mask][i]
                original_idx2 = idx2[valid_mask][i]
                
                map_point = self.map.add_or_update_point(
                    point_3d, f1, original_idx1
                )
                map_point.add_observation(f2, original_idx2)

            # Actualizar trayectoria
            t = f1.pose[:3, 3]
            self.display.update_trajectory(t[0], t[2])

            # Actualizar visualización
            good_map_points = [p for p in self.map.points if p.is_good()]
            self.display.update_map_visualization(good_map_points)
            self.display.update_metrics(frame.id + 1, len(self.map.points))
            
            # Info de debug
            print(f"✅ Frame {frame.id}: {len(idx1)} matches, "
                  f"trans={translation_norm:.3f}, "
                  f"pose: x={t[0]:.2f}, z={t[2]:.2f}")

        except Exception as e:
            print(f"❌ Frame {frame.id}: Error en SLAM: {e}")
            f1.pose = f2.pose.copy()
            self.frames_without_tracking += 1

        # Actualizar display
        self.display.update_frame_display(img, f1.kps)


def triangulate(pose1, pose2, pts1, pts2):
    """
    Triangula puntos 3D - VERSIÓN CORREGIDA
    """
    ret = np.zeros((pts1.shape[0], 4))
    pose1_inv = np.linalg.inv(pose1)
    pose2_inv = np.linalg.inv(pose2)
    
    for i, (p1, p2) in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        A = np.vstack([
            p1[0] * pose1_inv[2] - pose1_inv[0],
            p1[1] * pose1_inv[2] - pose1_inv[1],
            p2[0] * pose2_inv[2] - pose2_inv[0],
            p2[1] * pose2_inv[2] - pose2_inv[1]
        ])
        _, _, V = np.linalg.svd(A)
        ret[i] = V[-1]  # Última fila de V
    
    return ret


def filter_points_behind_camera(points, 
                                cam_pose, 
                                z_min_threshold=0.05,    # MÁS BAJO
                                z_max_threshold=50.0,    # MÁS ALTO
                                xy_threshold=25.0):      # MÁS PERMISIVO
    """
    Filtra puntos - VERSIÓN MÁS PERMISIVA
    """
    if len(points) == 0:
        return np.array([], dtype=bool)
    
    inv_pose = np.linalg.inv(cam_pose)
    valid_mask = []
    
    for point in points:
        point_cam = inv_pose.dot(point)
        
        is_valid = (
            point_cam[2] > z_min_threshold and 
            point_cam[2] < z_max_threshold and
            abs(point_cam[0]) < xy_threshold and
            abs(point_cam[1]) < xy_threshold
        )
        valid_mask.append(is_valid)
    
    return np.array(valid_mask, dtype=bool)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    camera = Camera()
    motor_ctrl = MotorController()
    display = MainWindow(motor_ctrl)
    display.show()
    
    # Conectar botones
    display.btn_forward.clicked.connect(motor_ctrl.forward)
    display.btn_backward.clicked.connect(motor_ctrl.backward)
    display.btn_left.clicked.connect(motor_ctrl.turn_left)
    display.btn_right.clicked.connect(motor_ctrl.turn_right)
    display.btn_stop.clicked.connect(motor_ctrl.stop)
    display.btn_quit.clicked.connect(app.quit)
    
    slam = VisualSLAM(camera, motor_ctrl, display)
    
    sys.exit(app.exec_())