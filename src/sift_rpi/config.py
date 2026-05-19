# config.py - OPTIMIZADO PARA RASPBERRY PI CON TRACKING SUAVE

# ORB (mucho más rápido que SIFT en RPi)
SIFT_N_FEATURES = 250  # Más features para mejor tracking en giros

# Path
CALIB_PATH = "/home/visualslam/PF/Github/CEIA-PF/src/sift_rpi/calibration"

# Camera - RESOLUCION MEDIA (balance entre velocidad y precisión)
WIDTH = 320   # Balance: no es ultra pequeño pero tampoco grande
HEIGHT = 240
F=600

# Motion
MIN_TRANSLATION = 0.02
TURN_REDUCTION = 20   # % que reduce al girar

# Extractor - SELECTIVO PARA EVITAR SATURACIÓN
MIN_PIXEL_DISP = 1.0
MIN_MATCHES = 6  # Mínimo decente para SIFT - no bajar más
MAX_POINTS_IN_MAP = 200  # Límite muy agresivo para no saturar RPi

# Raspberry Pi I/O
IN1, IN2, IN3, IN4 = 5, 6, 23, 24
PWM_CHIP = 2
PWM_CH0 = 0
PWM_CH1 = 1
FREQ =1000
INIT_DUTY = 65
PWM_FORWARD_DUTY_A = 55
PWM_FORWARD_DUTY_B = 55
PWM_BACKWARD_DUTY_A = 55
PWM_BACKWARD_DUTY_B = 50
PWM_TURN_DUTY = 40


# Display - OPTIMIZADO PARA TRACKING SUAVE
SKIP_RATE = 2  # Procesa TODOS los frames (antes: 1 de cada 3)
GUI_UPDATE_MS = 200  # Actualiza UI cada 200ms
TIMER_INTERVAL_MS = 50  # Procesa frame cada 40ms (~25 FPS) - NO usar 0

SLIDER_MIN = 0
SLIDER_MAX = 100

# Plot limits - SIMÉTRICOS Y CENTRADOS EN EL ROBOT (ZOOM)
PLOT_SIZE = 50
PLOT_X_MIN = -PLOT_SIZE
PLOT_X_MAX = PLOT_SIZE
PLOT_Z_MIN = -PLOT_SIZE
PLOT_Z_MAX = PLOT_SIZE
