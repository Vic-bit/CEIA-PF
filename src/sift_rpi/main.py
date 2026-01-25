# main.py - Mejorado con mejor tracking y gestión de mapa
import sys
import atexit
import signal
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication
from camera import Camera
from display import MainWindow
from extractor import Frame, match_frames, add_ones
from pointmap import Map
from motor_controller import MotorController
from config import (
    WIDTH, HEIGHT, F, MIN_TRANSLATION
)
from utils import get_intrinsic_matrix_from_npz

signal.signal(signal.SIGINT, lambda *args: sys.exit(0))
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))

# VO parameters
#K = np.array([[F, 0, WIDTH//2], [0, F, HEIGHT//2], [0, 0, 1]])
K = get_intrinsic_matrix_from_npz()

mapp = Map()
trajectory_x, trajectory_z = [], []
trajectory = {'x': trajectory_x, 'z': trajectory_z}

# Variables para tracking mejorado
last_good_pose = np.eye(4)
frames_since_good_match = 0
MAX_FRAMES_WITHOUT_MATCH = 5


def triangulate(pose1, pose2, pts1, pts2):
    """Triangula puntos 3D a partir de dos vistas."""
    ret = np.zeros((pts1.shape[0], 4))
    p1i, p2i = np.linalg.inv(pose1), np.linalg.inv(pose2)
    for i, (a, b) in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        A = np.vstack([
            a[0]*p1i[2] - p1i[0],
            a[1]*p1i[2] - p1i[1],
            b[0]*p2i[2] - p2i[0],
            b[1]*p2i[2] - p2i[1]
        ])
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret


def filter_points_behind_camera(pts, pose, z_thr=0.1):
    """Filtra puntos que están detrás de la cámara."""
    if pts.size == 0:
        return np.empty((0, 4))
    inv = np.linalg.inv(pose)
    good = [p for p in pts if inv.dot(p)[2] > z_thr]
    return np.array(good) if good else np.empty((0, 4))


def check_pose_validity(Rt):
    """Verifica que la pose sea razonable."""
    # Verificar rotación
    R = Rt[:3, :3]
    det = np.linalg.det(R)
    if abs(det - 1.0) > 0.1:
        return False
    
    # Verificar traslación no sea demasiado grande
    t = Rt[:3, 3]
    if np.linalg.norm(t) > 5.0:  # Más de 5 unidades es sospechoso
        return False
    
    return True


def process_vo(img):
    """Procesa un frame para Visual Odometry con mejor manejo de errores."""
    global last_good_pose, frames_since_good_match
    
    img = cv2.resize(img, (WIDTH, HEIGHT))
    frame = Frame(mapp, img, K)
    
    if frame.id < 1:
        return img, None

    f1, f2 = mapp.frames[-1], mapp.frames[-2]
    idx1, idx2, Rt = match_frames(f1, f2)
    
    # Si no hay matches
    if idx1.size == 0:
        print(f"⚠️ No matches - Frame {frame.id}")
        frames_since_good_match += 1
        
        # Si llevamos muchos frames sin match, mantener última pose conocida
        if frames_since_good_match > MAX_FRAMES_WITHOUT_MATCH:
            print("🔴 Demasiados frames sin match - manteniendo pose")
            f1.pose = last_good_pose.copy()
        else:
            # Mantener la pose del frame anterior
            f1.pose = f2.pose.copy()
        
        # Dibujar keypoints aunque no haya matches
        img_k = img.copy()
        for x, y in frame.kps.astype(int):
            cv2.circle(img_k, (x, y), 2, (0, 0, 255), -1)  # Rojo = sin matches
        return img_k, None
    
    # Verificar movimiento mínimo
    translation_norm = np.linalg.norm(Rt[:3, 3])
    if translation_norm < MIN_TRANSLATION:
        print(f"⚠️ Movimiento mínimo: {translation_norm:.4f}")
        Rt[:3, 3] = 0
        f1.pose = f2.pose.copy()
        
        img_k = img.copy()
        for x, y in frame.kps.astype(int):
            cv2.circle(img_k, (x, y), 2, (255, 255, 0), -1)  # Amarillo = poco movimiento
        return img_k, None
    
    # Verificar validez de la pose
    if not check_pose_validity(Rt):
        print(f"⚠️ Pose inválida detectada - Frame {frame.id}")
        f1.pose = f2.pose.copy()
        frames_since_good_match += 1
        
        img_k = img.copy()
        for x, y in frame.kps.astype(int):
            cv2.circle(img_k, (x, y), 2, (255, 0, 255), -1)  # Magenta = pose inválida
        return img_k, None
    
    # Actualizar pose
    f1.pose = f2.pose.dot(Rt)
    
    # Triangular puntos
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3, None]
    valid = filter_points_behind_camera(pts4d, f1.pose)
    
    # Agregar puntos válidos al mapa
    for i, pt in enumerate(valid):
        mapp.add_or_update_point(pt, f1, idx1[i], threshold=0.2)
    
    # Actualizar trayectoria
    T = f1.pose[:3, 3]
    trajectory_x.append(T[0])
    trajectory_z.append(T[2])
    
    # Guardar como última pose buena
    last_good_pose = f1.pose.copy()
    frames_since_good_match = 0
    
    # Dibujar keypoints
    img_k = img.copy()
    for x, y in frame.kps.astype(int):
        cv2.circle(img_k, (x, y), 2, (0, 255, 0), -1)  # Verde = todo OK
    
    # Mostrar info de tracking
    cv2.putText(img_k, f"Matches: {len(idx1)}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img_k, f"Points: {len(mapp.points)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img_k, f"Trans: {translation_norm:.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return img_k, valid


def main():
    # Inicializar componentes
    camera = Camera()
    motor_ctrl = MotorController()

    # Qt application
    app = QApplication(sys.argv)
    window = MainWindow(camera, motor_ctrl, process_vo, trajectory)
    
    # Registrar limpieza al salir
    atexit.register(window.cleanup)
    
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()