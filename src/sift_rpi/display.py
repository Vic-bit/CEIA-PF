import cv2
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QGridLayout,
    QGroupBox, QFormLayout, QSlider, QShortcut, QAction, QSplitter
)
from PyQt5.QtGui import QKeySequence
from config import (
    WIDTH, HEIGHT, INIT_DUTY, PLOT_X_MIN, PLOT_X_MAX, PLOT_Z_MIN, PLOT_Z_MAX,
    SLIDER_MIN, SLIDER_MAX, SKIP_RATE
)


class MainWindow(QMainWindow):
    """Ventana principal con visualización y controles del robot."""
    
    def __init__(self, camera, motor_controller, process_vo_callback, trajectory_data):
        """
        Inicializa la ventana principal.
        
        Args:
            camera: Instancia de Camera para capturar frames
            motor_controller: Instancia de MotorController para controlar motores
            process_vo_callback: Función que procesa frames de VO
            trajectory_data: Dict con listas 'x' y 'z' para la trayectoria
        """
        super().__init__()
        
        # Referencias externas
        self.camera = camera
        self.motor_ctrl = motor_controller
        self.process_vo = process_vo_callback
        self.trajectory_data = trajectory_data
        
        # Estado interno
        self.skip_rate = SKIP_RATE
        self._skip_counter = 0
        self.frame_count = 0
        
        # Configurar UI
        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._setup_timer()


    def _setup_ui(self):
        """Configura la interfaz de usuario."""
        self.setWindowTitle("VO + Control")
        
        # Central widget & layout
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        grid.setContentsMargins(5, 5, 5, 5)
        grid.setSpacing(10)

        # Splitter left: video & plot
        splitter = QSplitter(Qt.Vertical)
        grid.addWidget(splitter, 0, 0, 2, 1)

        # Video label
        self.video_label = QLabel()
        self.video_label.setFixedSize(WIDTH, HEIGHT)
        splitter.addWidget(self.video_label)

        # PyQtGraph plot
        self.pg_plot = pg.PlotWidget(title="Trayectoria (X,Z)")
        self.pg_plot.setLabel('bottom', 'X [m]')
        self.pg_plot.setLabel('left', 'Z [m]')
        self.pg_plot.showGrid(x=True, y=True)
        self.pg_plot.enableAutoRange(False, False)
        self.pg_plot.setXRange(PLOT_X_MIN, PLOT_X_MAX, padding=0)
        self.pg_plot.setYRange(PLOT_Z_MIN, PLOT_Z_MAX, padding=0)
        self.curve = self.pg_plot.plot(pen='y', symbol='o')
        splitter.addWidget(self.pg_plot)

        # Controls panel (right)
        control = self._create_control_panel()
        grid.addWidget(control, 0, 1)

        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 1)

 def _create_control_panel(self):
        """Crea el panel de controles y estado."""
        control = QGroupBox("Control & Status")
        form = QFormLayout()
        control.setLayout(form)

        # Labels de estado
        self.lbl_frame = QLabel("0")
        self.lbl_command = QLabel("-")
        self.lbl_dutyA = QLabel(f"{INIT_DUTY}%")
        self.lbl_dutyB = QLabel(f"{INIT_DUTY}%")
        form.addRow("Frames:", self.lbl_frame)
        form.addRow("Command:", self.lbl_command)
        form.addRow("Duty ENA:", self.lbl_dutyA)
        form.addRow("Duty ENB:", self.lbl_dutyB)

        # Slider for ENA
        self.sliderA = QSlider(Qt.Horizontal)
        self.sliderA.setRange(SLIDER_MIN, SLIDER_MAX)
        self.sliderA.setValue(INIT_DUTY)
        self.sliderA.valueChanged.connect(self.on_sliderA)
        form.addRow("Adjust ENA:", self.sliderA)

        # Slider for ENB
        self.sliderB = QSlider(Qt.Horizontal)
        self.sliderB.setRange(SLIDER_MIN, SLIDER_MAX)
        self.sliderB.setValue(INIT_DUTY)
        self.sliderB.valueChanged.connect(self.on_sliderB)
        form.addRow("Adjust ENB:", self.sliderB)
        
        return control

    def _setup_menu(self):
        """Configura menú y toolbar."""
        exitAct = QAction("Exit", self, shortcut="Q", triggered=self.close)
        menu = self.menuBar().addMenu("File")
        menu.addAction(exitAct)
        toolbar = self.addToolBar("Main")
        toolbar.addAction(exitAct)
    
    def _setup_shortcuts(self):
        """Configura atajos de teclado."""
        keys = {
            Qt.Key_W: self.forward,
            Qt.Key_S: self.backward,
            Qt.Key_A: self.turn_left,
            Qt.Key_D: self.turn_right,
            Qt.Key_E: self.stop_motors,
            Qt.Key_Q: self.close,
            Qt.Key_Plus: lambda: self.adjustDutyA(+5),
            Qt.Key_Minus: lambda: self.adjustDutyA(-5),
            Qt.Key_BracketRight: lambda: self.adjustDutyB(+5),
            Qt.Key_BracketLeft: lambda: self.adjustDutyB(-5),
        }
        for k, fn in keys.items():
            sc = QShortcut(QKeySequence(k), self, activated=fn)
            sc.setContext(Qt.ApplicationShortcut)
    
    def _setup_timer(self):
        """Configura el timer de actualización."""
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms → ~33 FPS

    # ===== Callbacks de sliders =====
    
    def on_sliderA(self, v):
        """Actualiza duty cycle del motor A."""
        self.motor_ctrl.set_duty_ena(v)
        self.lbl_dutyA.setText(f"{v}%")

    def on_sliderB(self, v):
        """Actualiza duty cycle del motor B."""
        self.motor_ctrl.set_duty_enb(v)
        self.lbl_dutyB.setText(f"{v}%")

    def adjustDutyA(self, delta):
        """Ajusta duty cycle del motor A con incremento."""
        self.sliderA.setValue(self.sliderA.value() + delta)

    def adjustDutyB(self, delta):
        """Ajusta duty cycle del motor B con incremento."""
        self.sliderB.setValue(self.sliderB.value() + delta)

    # ===== Comandos de movimiento =====
    
    def forward(self):
        """Comando: avanzar."""
        self.motor_ctrl.forward()
        self.lbl_command.setText("Forward")

    def backward(self):
        """Comando: retroceder."""
        self.motor_ctrl.backward()
        self.lbl_command.setText("Backward")

    def turn_right(self):
        """Comando: girar a la derecha."""
        self.motor_ctrl.turn_right()
        self.lbl_command.setText("Turn right")

    def turn_left(self):
        """Comando: girar a la izquierda."""
        self.motor_ctrl.turn_left()
        self.lbl_command.setText("Turn left")

    def stop_motors(self):
        """Comando: detener motores."""
        self.motor_ctrl.stop()
        self.lbl_command.setText("Stop motors")


    def update_frame(self):
        """Captura frame, procesa VO y actualiza la UI."""
        # Skip frames según configuración
        self._skip_counter += 1
        if self._skip_counter < self.skip_rate:
            return
        self._skip_counter = 0

        # Capturar y convertir imagen
        rgb = self.camera.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Procesar Visual Odometry
        img_disp, valid = self.process_vo(bgr)

        # Mostrar video en escala de grises
        gray = cv2.cvtColor(img_disp, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        qimg = QtGui.QImage(gray.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

        # Actualizar contador de frames
        self.frame_count += 1
        self.lbl_frame.setText(str(self.frame_count))
        
        # Actualizar plot de trayectoria
        self.curve.setData(self.trajectory_data['x'], self.trajectory_data['z']) #trajectory_x.append(T[0]); trajectory_z.append(T[2])

    # ===== Eventos de teclado =====
    
    def keyPressEvent(self, e):
        """Maneja eventos de teclado y actualiza label de comando."""
        key = e.key()
        cmds = {
            Qt.Key_W: "Forward",
            Qt.Key_S: "Backward",
            Qt.Key_A: "Turn Left",
            Qt.Key_D: "Turn Right",
            Qt.Key_E: "Stop",
            Qt.Key_Q: "Quit"
        }
        if key in cmds:
            self.lbl_command.setText(cmds[key])
        super().keyPressEvent(e)

    # ===== Limpieza =====
    
    def closeEvent(self, event):
        """Maneja el evento de cierre de ventana."""
        print("Cleanup before exit...")
        self.cleanup()
        event.accept()

    def cleanup(self):
        """Limpia recursos antes de salir."""
        self.motor_ctrl.cleanup()
        self.camera.close()