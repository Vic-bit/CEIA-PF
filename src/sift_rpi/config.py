# config.py

# SIFT
SIFT_N_FEATURES = 350

# Path
CALIB_PATH = "/home/visualslam/PF/Github/CEIA-PF/src/sift_rpi/config"

# Camera
WIDTH = 320
HEIGHT = 240
F = 450

MIN_TRANSLATION = 0.15
TURN_REDUCTION = 20   # % que reduce al girar

# Extractor
MIN_PIXEL_DISP = 1.0
MIN_MATCHES = 8

# Raspberry Pi I/O
IN1, IN2, IN3, IN4 = 5, 6, 23, 24
PWM_CHIP, PWM_CH0, PWM_CH1 = 2, 0, 1
FREQ, INIT_DUTY = 1000, 50
