# main.py
import sys, atexit, signal
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg

from extractor import Frame, match_frames, add_ones
from pointmap import Map

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QGridLayout,
    QGroupBox, QFormLayout, QSlider, QShortcut, QAction, QSplitter
)
from PyQt5.QtGui import QKeySequence
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from picamera2 import Picamera2
from time import sleep
from rpi_hardware_pwm import HardwarePWM
import gpiod

# — motor & PWM setup —
IN1, IN2, IN3, IN4 = 5, 6, 23, 24
PWM_CHIP, PWM_CH0, PWM_CH1 = 2, 0, 1
FREQ, INIT_DUTY = 1000, 50
pwmENA = HardwarePWM(pwm_channel=PWM_CH0, hz=FREQ, chip=PWM_CHIP)
pwmENB = HardwarePWM(pwm_channel=PWM_CH1, hz=FREQ, chip=PWM_CHIP)
pwmENA.start(INIT_DUTY)
pwmENB.start(INIT_DUTY)

chip = gpiod.Chip('gpiochip4')
lines = {}
for pin in (IN1, IN2, IN3, IN4):
    l = chip.get_line(pin)
    l.request(consumer="motor", type=gpiod.LINE_REQ_DIR_OUT)
    lines[pin] = l

signal.signal(signal.SIGINT, lambda *args: sys.exit(0))
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))

# — VO parameters —
W, H = 320, 240
F = 450
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
MIN_TRANSLATION = 0.15
TURN_REDUCTION = 20   # % que reduces al girar
mapp = Map()
trajectory_x, trajectory_z = [], []

def triangulate(pose1, pose2, pts1, pts2):
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
    inv = np.linalg.inv(pose)
    good = [p for p in pts if inv.dot(p)[2] > z_thr]
    return np.array(good) if good else np.empty((0,4))

# — MainWindow —
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.skip_rate    = 6    # procesar 1 de cada 5
        self._skip_counter = 0

        #self.timer = QtCore.QTimer(self)
        #self.timer.timeout.connect(self.update_frame)
        #self.timer.start(200)  # → cada 100 ms → ~10 FPS

        self.setWindowTitle("VO + Control")
        self.frame_count = 0
        # — Central widget & layout —
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        grid.setContentsMargins(5,5,5,5)
        grid.setSpacing(10)

        # — Splitter left: video & plot —
        splitter = QSplitter(Qt.Vertical)
        grid.addWidget(splitter, 0, 0, 2, 1)

        # video label
        self.video_label = QLabel()
        self.video_label.setFixedSize(W,H)
        splitter.addWidget(self.video_label)

        # matplotlib canvas
        #self.fig, self.ax = plt.subplots()
        #self.canvas = FigureCanvas(self.fig)
        #splitter.addWidget(self.canvas)

        # PyQtGraph plot (en lugar de Matplotlib)
        self.pg_plot = pg.PlotWidget(title="Trayectoria (X,Z)")
        self.pg_plot.setLabel('bottom', 'X [m]')
        self.pg_plot.setLabel('left',   'Z [m]')
        self.pg_plot.showGrid(x=True, y=True)

        # DESACTIVAR AUTORANGE
        self.pg_plot.enableAutoRange(False, False)      # (x, y)

        # FIJAR LÍMITES
        self.pg_plot.setXRange(-50, 50, padding=0)      # xmin, xmax
        self.pg_plot.setYRange(  0,100, padding=0)      # ymin, ymax

        # esta será tu "línea de trayectoria"
        self.curve = self.pg_plot.plot(pen='y', symbol='o')
        splitter.addWidget(self.pg_plot)

        # — Controls right —
        control = QGroupBox("Control & Status")
        form = QFormLayout()
        control.setLayout(form)
        grid.addWidget(control, 0, 1)

        self.lbl_frame   = QLabel("0")
        self.lbl_command = QLabel("-")
        self.lbl_dutyA   = QLabel(f"{INIT_DUTY}%")
        self.lbl_dutyB   = QLabel(f"{INIT_DUTY}%")
        form.addRow("Frames:",    self.lbl_frame)
        form.addRow("Command:",   self.lbl_command)
        form.addRow("Duty ENA:",  self.lbl_dutyA)
        form.addRow("Duty ENB:",  self.lbl_dutyB)

        # slider for ENA
        self.sliderA = QSlider(Qt.Horizontal)
        self.sliderA.setRange(0,100)
        self.sliderA.setValue(INIT_DUTY)
        self.sliderA.valueChanged.connect(self.on_sliderA)
        form.addRow("Adjust ENA:", self.sliderA)

        # slider for ENB
        self.sliderB = QSlider(Qt.Horizontal)
        self.sliderB.setRange(0,100)
        self.sliderB.setValue(INIT_DUTY)
        self.sliderB.valueChanged.connect(self.on_sliderB)
        form.addRow("Adjust ENB:", self.sliderB)

        grid.setColumnStretch(0,3)
        grid.setColumnStretch(1,1)

        # — menus/toolbars —
        exitAct = QAction("Exit", self, shortcut="Q", triggered=self.close)
        men = self.menuBar().addMenu("File")
        men.addAction(exitAct)
        tb = self.addToolBar("Main")
        tb.addAction(exitAct)

        # — initialize plot —
        #self.traj_line, = self.ax.plot([],[], 'bo-', label='Tray')
        #self.scatter   = self.ax.scatter([],[], c='r', s=5, label='Pts')
        #self.ax.set_xlim(-50,50); self.ax.set_ylim(0,100)
        #self.ax.set_xlabel("X"); self.ax.set_ylabel("Z")
        #self.ax.legend(); self.ax.grid(True)

        # — camera —
        self.picam2 = Picamera2()
        cfg = self.picam2.create_preview_configuration(main={"format":"RGB888","size":(W,H)})
        self.picam2.configure(cfg); self.picam2.start()
        sleep(1)

        # — timer for updates —
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # — shortcuts —
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
            Qt.Key_BracketLeft:  lambda: self.adjustDutyB(-5),
        }
        for k,fn in keys.items():
            sc = QShortcut(QKeySequence(k), self, activated=fn)
            sc.setContext(Qt.ApplicationShortcut)

    def on_sliderA(self, v):
        pwmENA.change_duty_cycle(v)
        self.lbl_dutyA.setText(f"{v}%")
    def on_sliderB(self, v):
        pwmENB.change_duty_cycle(v)
        self.lbl_dutyB.setText(f"{v}%")
    def adjustDutyA(self, d):  
        self.sliderA.setValue(self.sliderA.value()+d)
    def adjustDutyB(self, d):  
        self.sliderB.setValue(self.sliderB.value()+d)

    def forward(self):
        print("forward")
        lines[IN1].set_value(1); lines[IN2].set_value(0)
        lines[IN3].set_value(1); lines[IN4].set_value(0)
        pwmENA.change_duty_cycle(50)
        pwmENB.change_duty_cycle(45)
        self.lbl_command.setText("Forward")
    def backward(self):
        print("backward")
        lines[IN1].set_value(0); lines[IN2].set_value(1)
        lines[IN3].set_value(0); lines[IN4].set_value(1)
        pwmENA.change_duty_cycle(50)
        pwmENB.change_duty_cycle(45)
        self.lbl_command.setText("Backward")
    def turn_right(self):
        print("turn right")
        lines[IN1].set_value(1); lines[IN2].set_value(0)
        lines[IN3].set_value(0); lines[IN4].set_value(1)
        #newB = max(0, self.sliderB.value() - TURN_REDUCTION)
        #pwmENA.change_duty_cycle(self.sliderA.value())
        pwmENA.change_duty_cycle(40)
        pwmENB.change_duty_cycle(40)
        #self.lbl_command.setText("Turn Right")
        #self.lbl_dutyA.setText(f"{self.sliderA.value()}%")
        #self.lbl_dutyB.setText(f"{newB}%")
        self.lbl_command.setText("Turn right")
    def turn_left(self):
        print("turn left")
        lines[IN1].set_value(0); lines[IN2].set_value(1)
        lines[IN3].set_value(1); lines[IN4].set_value(0)
        #newA = max(0, self.sliderA.value() - TURN_REDUCTION)
        pwmENA.change_duty_cycle(40)
        pwmENB.change_duty_cycle(40)
        #pwmENB.change_duty_cycle(self.sliderB.value())
        #self.lbl_command.setText("Turn Left")
        #self.lbl_dutyA.setText(f"{newA}%")  # actualizamos indicador
        #self.lbl_dutyB.setText(f"{self.sliderB.value()}%")
        self.lbl_command.setText("Turn left")
    def stop_motors(self):
        print("stop motors")
        for p in (IN1,IN2,IN3,IN4): lines[p].set_value(0)
        self.lbl_command.setText("Stop motors")

    def update_frame(self):
        self._skip_counter += 1
        if self._skip_counter < self.skip_rate:
            return
        self._skip_counter = 0

        rgb = self.picam2.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # — VO processing —
        img_disp, valid = self.process_vo(bgr)

        # — display video (grayscale) —
        gray = cv2.cvtColor(img_disp, cv2.COLOR_BGR2GRAY)
        h,w = gray.shape
        qimg = QtGui.QImage(gray.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

        # — update frame count indicator —
        self.frame_count += 1
        self.lbl_frame.setText(str(self.frame_count))
        # Actualizar curva de trayectoria en PyQtGraph
        self.curve.setData(trajectory_x, trajectory_z)

    def process_vo(self, img):
        img = cv2.resize(img,(W,H))
        frame = Frame(mapp, img, K)
        if frame.id < 1:
            return img, None

        f1,f2 = mapp.frames[-1], mapp.frames[-2]
        idx1,idx2,Rt = match_frames(f1,f2)
        if idx1.size == 0:
            return img, None

        if np.linalg.norm(Rt[:3,3])<MIN_TRANSLATION:
            Rt[:3,3] = 0
        f1.pose = f2.pose.dot(Rt)

        pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
        pts4d /= pts4d[:,3,None]
        valid = filter_points_behind_camera(pts4d, f1.pose)

        # — update trajectory plot —
        T = f1.pose[:3,3]
        trajectory_x.append(T[0]); trajectory_z.append(T[2])
        #.set_data(trajectory_x, trajectory_z)
        #if valid.size>0:
        #    self.scatter.set_offsets(valid[:,[0,2]])
        #self.ax.relim(); self.ax.autoscale_view()
        #self.canvas.draw_idle()

        # — draw keypoints —
        img_k = img.copy()
        for x,y in frame.kps.astype(int):
            cv2.circle(img_k,(x,y),2,(0,255,0),-1)
        return img_k, valid

    def keyPressEvent(self, e):
        # override to update “Command” label
        key = e.key()
        cmds = {
            Qt.Key_W: "Forward", Qt.Key_S: "Backward",
            Qt.Key_A: "Turn Left", Qt.Key_D: "Turn Right",
            Qt.Key_E: "Stop",     Qt.Key_Q: "Quit"
        }
        if key in cmds:
            self.lbl_command.setText(cmds[key])
        super().keyPressEvent(e)

    def closeEvent(self, event):
        print("Cleanup before exit...")
        self.cleanup()
        event.accept()  # permite que se cierre la ventana

    def cleanup(self):
        self.stop_motors()
        pwmENA.stop()
        pwmENB.stop()
        chip.close()



def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    atexit.register(w.cleanup)  # registra cleanup de la instancia
    w.show()
    app.exec_()
    w.picam2.close()

if __name__ == "__main__":
    main()
