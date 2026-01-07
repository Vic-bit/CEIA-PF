# main.py
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
    WIDTH, HEIGHT, F, MIN_TRANSLATION, TURN_REDUCTION, GUI_UPDATE_MS
)


signal.signal(signal.SIGINT, lambda *args: sys.exit(0))
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))

# — VO parameters —
K = np.array([[F,0,WIDTH//2],[0,F,HEIGHT//2],[0,0,1]])

mapp = Map()
trajectory_x, trajectory_z = [], []
trayectory = {'x': trajectory_x, 'z': trajectory_z}

def triangulate(pose1, pose2, pts1, pts2):
    """Triangula puntos 3D a partir de dos vistas."""
    ret = np.zeros((pts1.shape[0],4))
    p1i, p2i = np.linalg.inv(pose1), np.linalg.inv(pose2)
    for i,(a,b) in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        A = np.vstack([a[0]*p1i[2]-p1i[0],
                       a[1]*p1i[2]-p1i[1],
                       b[0]*p2i[2]-p2i[0],
                       b[1]*p2i[2]-p2i[1]])
        _,_,vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret

def filter_points_behind_camera(pts, pose, z_thr=0.0):
    """Filtra puntos que están detrás de la cámara."""
    inv = np.linalg.inv(pose)
    good = [p for p in pts if inv.dot(p)[2] > z_thr]
    return np.array(good) if good else np.empty((0,4))

def process_vo(img):
    """Procesa un frame para Visual Odometry."""
    img = cv2.resize(img, (WIDTH, HEIGHT))
    frame = Frame(mapp, img, K)
    if frame.id < 1:
        return img, None

    f1, f2 = mapp.frames[-1], mapp.frames[-2]
    idx1, idx2, Rt = match_frames(f1, f2)
    if idx1.size == 0:
        return img, None

    if np.linalg.norm(Rt[:3, 3]) < MIN_TRANSLATION:
        Rt[:3, 3] = 0
    f1.pose = f2.pose.dot(Rt)

    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3, None]
    valid = filter_points_behind_camera(pts4d, f1.pose)

    # update trajectory
    T = f1.pose[:3, 3]
    trajectory_x.append(T[0])
    trajectory_z.append(T[2])

    # draw keypoints
    img_k = img.copy()
    for x, y in frame.kps.astype(int):
        cv2.circle(img_k, (x, y), 2, (0, 255, 0), -1)
    return img_k, valid


def main():
    # Inicializar componentes
    camera = Camera()
    motor_ctrl = MotorController()

    # Qt application
    app = QApplication(sys.argv)
    window = MainWindow(camera, motor_ctrl, process_vo, trayectory)
    
    # Registrar limpieza al salir
    atexit.register(window.cleanup)
    
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
