# display.py - Interfaz PyQt5 con controles SLAM + Motores
import cv2
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QFrame, QGridLayout, QGroupBox, QFormLayout, QSlider
)
from PyQt5.QtGui import QImage, QPixmap
from config import WIDTH, HEIGHT, PLOT_X_MIN, PLOT_X_MAX, PLOT_Z_MIN, PLOT_Z_MAX



class MainWindow(QMainWindow):
    """Ventana principal con SLAM + controles de motor."""
    
    def __init__(self, motor_controller):
        super().__init__()
        self.setWindowTitle("Visual SLAM + Motor Control - Raspberry Pi")
        self.motor_ctrl = motor_controller
        self.traj_x, self.traj_z = [], []
        
        self._setup_ui()
        self._setup_shortcuts()
        
    def _setup_ui(self):
        """Configura la interfaz de usuario."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal horizontal: SLAM a la izquierda, controles a la derecha
        main_layout = QHBoxLayout(central_widget)
        
        # ========== PANEL IZQUIERDO: SLAM ==========
        slam_layout = QVBoxLayout()
        
        # Métricas
        metrics_layout = QHBoxLayout()
        self.lbl_frames = QLabel("Frames: 0")
        self.lbl_frames.setAlignment(Qt.AlignCenter)
        self.lbl_points = QLabel("Puntos: 0")
        self.lbl_points.setAlignment(Qt.AlignCenter)
        metrics_layout.addWidget(self.lbl_frames)
        metrics_layout.addWidget(self.lbl_points)
        slam_layout.addLayout(metrics_layout)
        
        # Video
        self.video_label = QLabel()
        self.video_label.setFrameShape(QFrame.Box)
        self.video_label.setFixedSize(WIDTH, HEIGHT)
        slam_layout.addWidget(self.video_label, stretch=1, alignment=Qt.AlignCenter)
        
        # Gráfico de trayectoria
        self.pg_plot = pg.PlotWidget(title="Trayectoria y Mapeo (X,Z)")
        self.map_scatter = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(255, 0, 0, 80))
        self.pg_plot.addItem(self.map_scatter)
        self.curve = self.pg_plot.plot(pen='y', symbol='o', symbolSize=5)
        self.pg_plot.setLabel('bottom', 'X')
        self.pg_plot.setLabel('left', 'Z')
        self.pg_plot.setFixedSize(400, 400)
        self.pg_plot.setXRange(PLOT_X_MIN, PLOT_X_MAX)
        self.pg_plot.setYRange(PLOT_Z_MIN, PLOT_Z_MAX)
        self.pg_plot.showGrid(x=True, y=True)
        slam_layout.addWidget(self.pg_plot, stretch=0, alignment=Qt.AlignCenter)
        
        # ========== PANEL DERECHO: CONTROLES ==========
        control_layout = QVBoxLayout()
        
        # === CONTROLES DE MOVIMIENTO ===
        movement_group = QGroupBox("Movimiento (WSAD)")
        movement_layout = QGridLayout(movement_group)
        
        # Botón W (adelante)
        self.btn_forward = QtWidgets.QPushButton("W - Adelante")
        self.btn_forward.setFixedSize(120, 50)
        movement_layout.addWidget(self.btn_forward, 0, 1)
        
        # Botón A (izquierda)
        self.btn_left = QtWidgets.QPushButton("A - Izquierda")
        self.btn_left.setFixedSize(120, 50)
        movement_layout.addWidget(self.btn_left, 1, 0)
        
        # Botón S (atrás)
        self.btn_backward = QtWidgets.QPushButton("S - Atrás")
        self.btn_backward.setFixedSize(120, 50)
        movement_layout.addWidget(self.btn_backward, 1, 1)
        
        # Botón D (derecha)
        self.btn_right = QtWidgets.QPushButton("D - Derecha")
        self.btn_right.setFixedSize(120, 50)
        movement_layout.addWidget(self.btn_right, 1, 2)
        
        # Botón E (detener)
        self.btn_stop = QtWidgets.QPushButton("E - DETENER")
        self.btn_stop.setFixedSize(120, 50)
        self.btn_stop.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        movement_layout.addWidget(self.btn_stop, 2, 1)
        
        control_layout.addWidget(movement_group)
        
        # === CONTROL DE VELOCIDAD ===
        speed_group = QGroupBox("Velocidad Motor A (ENA)")
        speed_layout = QFormLayout(speed_group)
        
        self.slider_motor_a = QSlider(Qt.Horizontal)
        self.slider_motor_a.setMinimum(0)
        self.slider_motor_a.setMaximum(100)
        self.slider_motor_a.setValue(50)
        self.slider_motor_a.setTickPosition(QSlider.TicksBelow)
        self.slider_motor_a.setTickInterval(10)
        self.lbl_motor_a_val = QLabel("50%")
        self.lbl_motor_a_val.setAlignment(Qt.AlignCenter)
        
        self.slider_motor_a.valueChanged.connect(lambda v: self._update_motor_a_label(v))
        self.slider_motor_a.valueChanged.connect(lambda v: self.motor_ctrl.set_duty_ena(v))
        
        speed_layout.addRow("Duty Cycle ENA:", self.slider_motor_a)
        speed_layout.addRow("Valor:", self.lbl_motor_a_val)
        
        control_layout.addWidget(speed_group)
        
        # === CONTROL DE VELOCIDAD MOTOR B ===
        speed_group_b = QGroupBox("Velocidad Motor B (ENB)")
        speed_layout_b = QFormLayout(speed_group_b)
        
        self.slider_motor_b = QSlider(Qt.Horizontal)
        self.slider_motor_b.setMinimum(0)
        self.slider_motor_b.setMaximum(100)
        self.slider_motor_b.setValue(50)
        self.slider_motor_b.setTickPosition(QSlider.TicksBelow)
        self.slider_motor_b.setTickInterval(10)
        self.lbl_motor_b_val = QLabel("50%")
        self.lbl_motor_b_val.setAlignment(Qt.AlignCenter)
        
        self.slider_motor_b.valueChanged.connect(lambda v: self._update_motor_b_label(v))
        self.slider_motor_b.valueChanged.connect(lambda v: self.motor_ctrl.set_duty_enb(v))
        
        speed_layout_b.addRow("Duty Cycle ENB:", self.slider_motor_b)
        speed_layout_b.addRow("Valor:", self.lbl_motor_b_val)
        
        control_layout.addWidget(speed_group_b)
        
        # Espaciador
        control_layout.addStretch()
        
        # === SALIR ===
        self.btn_quit = QtWidgets.QPushButton("Q - SALIR")
        self.btn_quit.setFixedSize(120, 50)
        self.btn_quit.setStyleSheet("background-color: darkred; color: white; font-weight: bold;")
        control_layout.addWidget(self.btn_quit, alignment=Qt.AlignCenter)
        
        # === AGREGAR PANELES AL LAYOUT PRINCIPAL ===
        left_panel = QWidget()
        left_panel.setLayout(slam_layout)
        main_layout.addWidget(left_panel, stretch=2)
        
        right_panel = QWidget()
        right_panel.setLayout(control_layout)
        right_panel.setMaximumWidth(300)
        main_layout.addWidget(right_panel, stretch=1)

    def _setup_shortcuts(self):
        """Configura shortcuts de teclado."""
        # W - Adelante
        QtWidgets.QShortcut(Qt.Key_W, self, self.motor_ctrl.forward)
        # A - Izquierda
        QtWidgets.QShortcut(Qt.Key_A, self, self.motor_ctrl.turn_left)
        # S - Atrás
        QtWidgets.QShortcut(Qt.Key_S, self, self.motor_ctrl.backward)
        # D - Derecha
        QtWidgets.QShortcut(Qt.Key_D, self, self.motor_ctrl.turn_right)
        # E - Detener
        QtWidgets.QShortcut(Qt.Key_E, self, self.motor_ctrl.stop)
        # Q - Salir
        QtWidgets.QShortcut(Qt.Key_Q, self, self.close)

    def _update_motor_a_label(self, value):
        """Actualiza el label del motor A."""
        self.lbl_motor_a_val.setText(f"{value}%")

    def _update_motor_b_label(self, value):
        """Actualiza el label del motor B."""
        self.lbl_motor_b_val.setText(f"{value}%")

    def update_frame_display(self, img, keypoints):
        """Actualiza la visualización del frame con los keypoints en escala de grises."""
        disp = cv2.resize(img, (WIDTH, HEIGHT)).copy()
        
        for x, y in keypoints.astype(int):
            cv2.circle(disp, (x, y), 2, 255, -1)  # Blanco en grayscale
        
        h, w = disp.shape[:2]
        qimg = QImage(disp.data, w, h, w, QImage.Format_Grayscale8)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def update_trajectory(self, x, z):
        """Actualiza la trayectoria con una nueva posición."""
        self.traj_x.append(x)
        self.traj_z.append(z)
        self.curve.setData(self.traj_x, self.traj_z)

    def update_map_visualization(self, points):
        """Actualiza la visualización de los puntos del mapa."""
        if len(points) > 0:
            map_points_xz = []
            for point in points:
                x, z = point.pt[0], point.pt[2]
                map_points_xz.append((x, z))
            
            if map_points_xz:
                pts = [{'pos': pt} for pt in map_points_xz]
                self.map_scatter.setData(pts)

    def update_metrics(self, frame_count, point_count):
        """Actualiza las métricas mostradas."""
        self.lbl_frames.setText(f"Frames: {frame_count}")
        self.lbl_points.setText(f"Puntos: {point_count}")

    def closeEvent(self, event):
        """Limpia recursos al cerrar."""
        try:
            self.motor_ctrl.cleanup()
        except:
            pass
        event.accept()
