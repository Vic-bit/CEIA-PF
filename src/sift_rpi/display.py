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
from config import (
    WIDTH, HEIGHT, PLOT_X_MIN, PLOT_X_MAX, PLOT_Z_MIN, PLOT_Z_MAX,
    PWM_FORWARD_DUTY_A, PWM_FORWARD_DUTY_B,
    PWM_BACKWARD_DUTY_A, PWM_BACKWARD_DUTY_B,
    PWM_TURN_DUTY
)



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
        movement_layout.setSpacing(5)  # Reducir espacio entre botones
        movement_layout.setContentsMargins(5, 5, 5, 5)  # Reducir margen
        
        # Botón W (adelante)
        self.btn_forward = QtWidgets.QPushButton("W\nAdelante")
        self.btn_forward.setFixedSize(70, 40)
        self.btn_forward.clicked.connect(lambda: self._handle_movement(self.motor_ctrl.forward, 'forward'))
        movement_layout.addWidget(self.btn_forward, 0, 1)
        
        # Botón A (izquierda)
        self.btn_left = QtWidgets.QPushButton("A\nIzq")
        self.btn_left.setFixedSize(70, 40)
        self.btn_left.clicked.connect(lambda: self._handle_movement(self.motor_ctrl.turn_left, 'turn_left'))
        movement_layout.addWidget(self.btn_left, 1, 0)
        
        # Botón S (atrás)
        self.btn_backward = QtWidgets.QPushButton("S\nAtrás")
        self.btn_backward.setFixedSize(70, 40)
        self.btn_backward.clicked.connect(lambda: self._handle_movement(self.motor_ctrl.backward, 'backward'))
        movement_layout.addWidget(self.btn_backward, 1, 1)
        
        # Botón D (derecha)
        self.btn_right = QtWidgets.QPushButton("D\nDer")
        self.btn_right.setFixedSize(70, 40)
        self.btn_right.clicked.connect(lambda: self._handle_movement(self.motor_ctrl.turn_right, 'turn_right'))
        movement_layout.addWidget(self.btn_right, 1, 2)
        
        # Botón E (detener)
        self.btn_stop = QtWidgets.QPushButton("E\nDETENER")
        self.btn_stop.setFixedSize(70, 40)
        self.btn_stop.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.btn_stop.clicked.connect(self.motor_ctrl.stop)
        movement_layout.addWidget(self.btn_stop, 2, 1)
        
        control_layout.addWidget(movement_group)
        
        # === CONTROL DE VELOCIDAD - FORWARD ===
        forward_group = QGroupBox("Forward (W)")
        forward_layout = QFormLayout(forward_group)
        
        # Forward Motor A
        self.slider_forward_a = QSlider(Qt.Horizontal)
        self.slider_forward_a.setMinimum(0)
        self.slider_forward_a.setMaximum(100)
        self.slider_forward_a.setValue(PWM_FORWARD_DUTY_A)
        self.slider_forward_a.setTickPosition(QSlider.TicksBelow)
        self.slider_forward_a.setTickInterval(10)
        self.lbl_forward_a_val = QLabel(f"{PWM_FORWARD_DUTY_A}%")
        self.lbl_forward_a_val.setAlignment(Qt.AlignCenter)
        self.slider_forward_a.valueChanged.connect(lambda v: self._update_label(self.lbl_forward_a_val, v))
        
        forward_layout.addRow("Motor A (ENA):", self.slider_forward_a)
        forward_layout.addRow("Valor:", self.lbl_forward_a_val)
        
        # Forward Motor B
        self.slider_forward_b = QSlider(Qt.Horizontal)
        self.slider_forward_b.setMinimum(0)
        self.slider_forward_b.setMaximum(100)
        self.slider_forward_b.setValue(PWM_FORWARD_DUTY_B)
        self.slider_forward_b.setTickPosition(QSlider.TicksBelow)
        self.slider_forward_b.setTickInterval(10)
        self.lbl_forward_b_val = QLabel(f"{PWM_FORWARD_DUTY_B}%")
        self.lbl_forward_b_val.setAlignment(Qt.AlignCenter)
        self.slider_forward_b.valueChanged.connect(lambda v: self._update_label(self.lbl_forward_b_val, v))
        
        forward_layout.addRow("Motor B (ENB):", self.slider_forward_b)
        forward_layout.addRow("Valor:", self.lbl_forward_b_val)
        
        control_layout.addWidget(forward_group)
        
        # === CONTROL DE VELOCIDAD - BACKWARD ===
        backward_group = QGroupBox("Backward (S)")
        backward_layout = QFormLayout(backward_group)
        
        # Backward Motor A
        self.slider_backward_a = QSlider(Qt.Horizontal)
        self.slider_backward_a.setMinimum(0)
        self.slider_backward_a.setMaximum(100)
        self.slider_backward_a.setValue(PWM_BACKWARD_DUTY_A)
        self.slider_backward_a.setTickPosition(QSlider.TicksBelow)
        self.slider_backward_a.setTickInterval(10)
        self.lbl_backward_a_val = QLabel(f"{PWM_BACKWARD_DUTY_A}%")
        self.lbl_backward_a_val.setAlignment(Qt.AlignCenter)
        self.slider_backward_a.valueChanged.connect(lambda v: self._update_label(self.lbl_backward_a_val, v))
        
        backward_layout.addRow("Motor A (ENA):", self.slider_backward_a)
        backward_layout.addRow("Valor:", self.lbl_backward_a_val)
        
        # Backward Motor B
        self.slider_backward_b = QSlider(Qt.Horizontal)
        self.slider_backward_b.setMinimum(0)
        self.slider_backward_b.setMaximum(100)
        self.slider_backward_b.setValue(PWM_BACKWARD_DUTY_B)
        self.slider_backward_b.setTickPosition(QSlider.TicksBelow)
        self.slider_backward_b.setTickInterval(10)
        self.lbl_backward_b_val = QLabel(f"{PWM_BACKWARD_DUTY_B}%")
        self.lbl_backward_b_val.setAlignment(Qt.AlignCenter)
        self.slider_backward_b.valueChanged.connect(lambda v: self._update_label(self.lbl_backward_b_val, v))
        
        backward_layout.addRow("Motor B (ENB):", self.slider_backward_b)
        backward_layout.addRow("Valor:", self.lbl_backward_b_val)
        
        control_layout.addWidget(backward_group)
        
        # === CONTROL DE VELOCIDAD - TURN ===
        turn_group = QGroupBox("Turn (A/D)")
        turn_layout = QFormLayout(turn_group)
        
        self.slider_turn = QSlider(Qt.Horizontal)
        self.slider_turn.setMinimum(0)
        self.slider_turn.setMaximum(100)
        self.slider_turn.setValue(PWM_TURN_DUTY)
        self.slider_turn.setTickPosition(QSlider.TicksBelow)
        self.slider_turn.setTickInterval(10)
        self.lbl_turn_val = QLabel(f"{PWM_TURN_DUTY}%")
        self.lbl_turn_val.setAlignment(Qt.AlignCenter)
        self.slider_turn.valueChanged.connect(lambda v: self._update_label(self.lbl_turn_val, v))
        
        turn_layout.addRow("Motor A/B:", self.slider_turn)
        turn_layout.addRow("Valor:", self.lbl_turn_val)
        
        control_layout.addWidget(turn_group)
        
        # Espaciador
        control_layout.addStretch()
        
        # === SALIR ===
        self.btn_quit = QtWidgets.QPushButton("Q - SALIR")
        self.btn_quit.setFixedSize(70, 40)
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
        QtWidgets.QShortcut(Qt.Key_W, self, lambda: self._handle_movement(self.motor_ctrl.forward, 'forward'))
        # A - Izquierda
        QtWidgets.QShortcut(Qt.Key_A, self, lambda: self._handle_movement(self.motor_ctrl.turn_left, 'turn_left'))
        # S - Atrás
        QtWidgets.QShortcut(Qt.Key_S, self, lambda: self._handle_movement(self.motor_ctrl.backward, 'backward'))
        # D - Derecha
        QtWidgets.QShortcut(Qt.Key_D, self, lambda: self._handle_movement(self.motor_ctrl.turn_right, 'turn_right'))
        # E - Detener
        QtWidgets.QShortcut(Qt.Key_E, self, self.motor_ctrl.stop)
        # Q - Salir
        QtWidgets.QShortcut(Qt.Key_Q, self, self.close)

    def _update_label(self, label, value):
        """Actualiza el label con el valor del slider."""
        label.setText(f"{value}%")

    def _handle_movement(self, movement_func, movement_type):
        """Ejecuta una función de movimiento con duty cycles del slider correspondiente.
        
        Args:
            movement_func: Función del motor controller (forward, backward, turn_left, turn_right)
            movement_type: String del tipo de movimiento ('forward', 'backward', 'turn_left', 'turn_right')
        """
        if movement_type == 'forward':
            duty_a = self.slider_forward_a.value()
            duty_b = self.slider_forward_b.value()
            movement_func(duty_a, duty_b)
        elif movement_type == 'backward':
            duty_a = self.slider_backward_a.value()
            duty_b = self.slider_backward_b.value()
            movement_func(duty_a, duty_b)
        elif movement_type == 'turn_left':
            duty = self.slider_turn.value()
            movement_func(duty)
        elif movement_type == 'turn_right':
            duty = self.slider_turn.value()
            movement_func(duty)
        
        self._sync_sliders_with_motor()

    def _sync_sliders_with_motor(self):
        """Sincroniza los sliders con los valores actuales del motor controller.
        Los sliders mantienen sus valores independientemente.
        """
        pass  # No es necesario sincronizar, cada slider es independiente

    def update_frame_display(self, img, keypoints):
        """Actualiza la visualización del frame con los keypoints.
        Procesa en grayscale pero muestra en RGB con keypoints verdes para mejor contraste.
        """
        disp = cv2.resize(img, (WIDTH, HEIGHT)).copy()
        # Convertir grayscale a BGR para poder mostrar puntos verdes
        disp_bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        
        for x, y in keypoints.astype(int):
            cv2.circle(disp_bgr, (x, y), 2, (0, 255, 0), -1)  # Verde brillante
        
        h, w, _ = disp_bgr.shape
        qimg = QImage(disp_bgr.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
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
