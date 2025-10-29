# display.py
import cv2
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
        self.setWindowTitle("Visual SLAM Dashboard")
        self.traj_x, self.traj_z = [], []
        self._setup_ui()
        self.W = W
        self.H = H

    def _setup_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        # Layout vertical principal
        main_layout = QVBoxLayout(cw)

        # Métricas en fila horizontal arriba
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

        # Gráfico de trayectoria debajo
        self.pg_plot = pg.PlotWidget(title="Trayectoria y Mapeo (X,Z)")

        # Scatter para puntos del mapa global
        self.map_scatter = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(255, 0, 0, 80))
        self.pg_plot.addItem(self.map_scatter)

        # Curva para la trayectoria
        self.curve = self.pg_plot.plot(pen='y', symbol='o', symbolSize=5)
        self.pg_plot.setLabel('bottom', 'X')
        self.pg_plot.setLabel('left', 'Z')
        self.pg_plot.setFixedSize(400, 400)
        self.pg_plot.setXRange(-300, 300)
        self.pg_plot.setYRange(0, 600)
        self.pg_plot.showGrid(x=True, y=True)
        main_layout.addWidget(self.pg_plot, stretch=1, alignment=Qt.AlignHCenter)

    def update_frame_display(self, img, keypoints):
        """Actualiza la visualización del frame con los keypoints"""
        disp = cv2.resize(img, (self.W, self.H)).copy()
        
        for x, y in keypoints.astype(int):
            cv2.circle(disp, (x, y), 2, (0, 255, 0), -1)

        h, w, _ = disp.shape
        qimg = QImage(disp.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def update_trajectory(self, x, z):
        """Actualiza la trayectoria con una nueva posición"""
        self.traj_x.append(x)
        self.traj_z.append(z)
        self.curve.setData(self.traj_x, self.traj_z)

    def update_map_visualization(self, points):
        """Actualiza la visualización de los puntos del mapa en PyQtGraph"""
        if len(points) > 0:
            # Extraer coordenadas X,Z de todos los puntos buenos
            map_points_xz = []
            for point in points:
                x, z = point.pt[0], point.pt[2]
                map_points_xz.append((x, z))
            
            if map_points_xz:
                # PyQtGraph quiere una lista de diccionarios {'pos': (x,z)}
                pts = [{'pos': pt} for pt in map_points_xz]
                self.map_scatter.setData(pts)

    def update_metrics(self, frame_count, point_count):
        """Actualiza las métricas mostradas
        
        Args:
            frame_count (int): Número de frames procesados
            point_count (int): Número de puntos 3D en el mapa
        """
        self.lbl_frames.setText(f"Frames: {frame_count}")
        self.lbl_points.setText(f"Puntos: {point_count}")

    def keyPressEvent(self, event):
        """Maneja eventos de teclado
        
        Args:
            event: Evento de teclado
        """
        if event.key() == Qt.Key_Q:
            self.close()