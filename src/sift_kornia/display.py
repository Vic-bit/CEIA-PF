# display.py
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg


class Display(QMainWindow):
    def __init__(self, W, H):
        super().__init__()
        self.setWindowTitle("Visual SLAM Dashboard - Kornia")
        self.W = W
        self.H = H
        self.traj_x, self.traj_z = [], []
        
        self._setup_ui()

    def _setup_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        main_layout = QVBoxLayout(cw)

        # Métricas
        metrics_layout = QHBoxLayout()
        self.lbl_frames = QLabel("Frames: 0")
        self.lbl_frames.setAlignment(Qt.AlignCenter)
        self.lbl_points = QLabel("Puntos: 0")
        self.lbl_points.setAlignment(Qt.AlignCenter)
        metrics_layout.addWidget(self.lbl_frames)
        metrics_layout.addWidget(self.lbl_points)
        main_layout.addLayout(metrics_layout)

        # Video
        self.video_label = QLabel()
        self.video_label.setFrameShape(QFrame.Box)
        main_layout.addWidget(self.video_label, stretch=2, alignment=Qt.AlignHCenter)

        # Gráfico de trayectoria
        self.pg_plot = pg.PlotWidget(title="Trayectoria y Mapeo (X,Z)")
        self.map_scatter = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(255, 0, 0, 80))
        self.pg_plot.addItem(self.map_scatter)
        self.curve = self.pg_plot.plot(pen='y', symbol='o', symbolSize=5)
        self.pg_plot.setLabel('bottom', 'X')
        self.pg_plot.setLabel('left', 'Z')
        self.pg_plot.setFixedSize(400, 400)
        self.pg_plot.setXRange(-300, 300)
        self.pg_plot.setYRange(0, 600)
        self.pg_plot.showGrid(x=True, y=True)
        main_layout.addWidget(self.pg_plot, stretch=1, alignment=Qt.AlignHCenter)

        # Tiempo 
        self.lbl_time = QLabel("Time: 0:00")
        metrics_layout.addWidget(self.lbl_time)

    def update_frame_display(self, img, keypoints):
        """Actualiza visualización con keypoints (tensores o numpy)"""
        disp = cv2.resize(img, (self.W, self.H)).copy()
        
        # Convertir keypoints si es tensor
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.detach().cpu().numpy()
        
        for x, y in keypoints.astype(int):
            cv2.circle(disp, (x, y), 2, (0, 255, 0), -1)

        h, w, _ = disp.shape
        qimg = QImage(disp.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def update_trajectory(self, x, z):
        """Actualiza trayectoria (acepta tensores o floats)"""
        if isinstance(x, torch.Tensor):
            x = x.item()
        if isinstance(z, torch.Tensor):
            z = z.item()
        
        self.traj_x.append(x)
        self.traj_z.append(z)
        self.curve.setData(self.traj_x, self.traj_z)

    def update_map_visualization(self, points):
        """Actualiza puntos del mapa con optimización batch"""
        if len(points) > 0:
            try:
                # Batch conversion: todos los puntos de una vez
                tensor_points = [p.pt for p in points if isinstance(p.pt, torch.Tensor)]
                
                if tensor_points:
                    all_points = torch.stack(tensor_points)
                    points_cpu = all_points[:, [0, 2]].detach().cpu().numpy()
                    
                    # Filtrado vectorizado
                    mask = ((np.abs(points_cpu[:, 0]) < 300) & 
                           (np.abs(points_cpu[:, 1]) < 600) & 
                           (points_cpu[:, 1] > 0))
                    
                    filtered_points = points_cpu[mask]
                    
                    if len(filtered_points) > 0:
                        pts = [{'pos': pt} for pt in filtered_points]
                        self.map_scatter.setData(pts)
            except Exception as e:
                pass

    def update_metrics(self, frame_count, point_count, time_str="0:00"):
        """Actualiza métricas"""
        self.lbl_frames.setText(f"Frames: {frame_count}")
        self.lbl_points.setText(f"Puntos: {point_count}")
        self.lbl_time.setText(f"Time: {time_str}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()