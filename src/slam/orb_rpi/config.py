# config.py - OPTIMIZADO PARA RASPBERRY PI CON TRACKING SUAVE

# ORB (mucho más rápido que SIFT en RPi)
ORB_N_FEATURES = 700  # Aumentado: más features = más puntos capturados en mapa

# Path
CALIB_PATH = "/home/visualslam/PF/Github/CEIA-PF/src/sift_rpi/calibration"

# Camera - RESOLUCION MEDIA (balance entre velocidad y precisión)
WIDTH = 320   # Balance: no es ultra pequeño pero tampoco grande
HEIGHT = 240
F=600

# Motion
MIN_TRANSLATION = 0.02
TURN_REDUCTION = 20   # % que reduce al girar

# Scale & Constraint (Option A: Fixed camera height)
CAMERA_HEIGHT = 0.089  # metros (89mm above ground plane)
POINT_MERGE_THRESHOLD = 4.0  # REDUCIDO: 4.0 era excesivo, fusionaba puntos lejanos en uno
POINT_Z_MIN = 0.05  # minimum depth (m)
POINT_Z_MAX = 1000.0  # maximum depth (m) - sin límite práctico, fallback si robot_position=None
CHESSBOARD_ROWS = 7
CHESSBOARD_COLS = 7
SCALE_CLAMP_MIN = 0.01  # m (minimum realistic scale)
SCALE_CLAMP_MAX = 2.0  # m (maximum realistic scale for RPi robot)
SOFT_CONSTRAINT_ALPHA = 0.95  # blending factor (1.0 = hard constraint, 0.5 = 50% correction)

# Extractor - SELECTIVO PARA EVITAR SATURACIÓN
MIN_PIXEL_DISP = 1.0
MIN_MATCHES = 6  # Mínimo decente para SIFT - no bajar más
MAX_POINTS_IN_MAP = 750  # Límite muy agresivo para no saturar RPi

# Raspberry Pi I/O
IN1, IN2, IN3, IN4 = 5, 6, 23, 24
PWM_CHIP = 2
PWM_CH0 = 0
PWM_CH1 = 1
FREQ =1000
INIT_DUTY = 65
PWM_FORWARD_DUTY_A = 67
PWM_FORWARD_DUTY_B = 64
PWM_BACKWARD_DUTY_A = 65
PWM_BACKWARD_DUTY_B = 60
PWM_TURN_DUTY = 50


# Display - OPTIMIZADO PARA TRACKING SUAVE
SKIP_RATE = 1  # Procesa TODOS los frames (antes: 1 de cada 3)
GUI_UPDATE_MS = 200  # Actualiza UI cada 200ms
TIMER_INTERVAL_MS = 50  # Procesa frame cada 40ms (~25 FPS) - NO usar 0

SLIDER_MIN = 0
SLIDER_MAX = 100

# Plot limits - SIMÉTRICOS Y CENTRADOS EN EL ROBOT (ZOOM)
PLOT_SIZE = 100
PLOT_X_MIN = -PLOT_SIZE
PLOT_X_MAX = PLOT_SIZE
PLOT_Z_MIN = -PLOT_SIZE
PLOT_Z_MAX = PLOT_SIZE


# Filtros de rotación
MAX_ROTATION_ABSOLUTE_DEG = 45.0   # Límite físico absoluto por frame procesado
MAX_ROTATION_FEW_MATCHES_DEG = 20.0  # Límite cuando hay pocos matches
MIN_MATCHES_FOR_ROTATION_TRUST = 15.0  # Por debajo de esto la rotación es poco fiable