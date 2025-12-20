# motor_controller.py - Control de motores

import gpiod
from rpi_hardware_pwm import HardwarePWM
from config import (
    IN1, IN2, IN3, IN4, PWM_CHIP, PWM_CH0, PWM_CH1, 
    FREQ, INIT_DUTY, PWM_FORWARD_DUTY_A, PWM_FORWARD_DUTY_B,
    PWM_BACKWARD_DUTY_A, PWM_BACKWARD_DUTY_B, PWM_TURN_DUTY
)

    
class MotorController:
    """Controla los motores DC del robot mediante PWM y GPIO."""
    
    def __init__(self):
        # Inicializar PWM
        self.pwm_ena = HardwarePWM(pwm_channel=PWM_CH0, hz=FREQ, chip=PWM_CHIP)
        self.pwm_enb = HardwarePWM(pwm_channel=PWM_CH1, hz=FREQ, chip=PWM_CHIP)
        self.pwm_ena.start(INIT_DUTY)
        self.pwm_enb.start(INIT_DUTY)
        
        # Inicializar GPIO
        self.chip = gpiod.Chip('gpiochip4')
        self.lines = {}
        for pin in (IN1, IN2, IN3, IN4):
            line = self.chip.get_line(pin)
            line.request(consumer="motor", type=gpiod.LINE_REQ_DIR_OUT)
            self.lines[pin] = line
    
    def set_duty_ena(self, duty):
        """Establece el duty cycle del motor A (ENA)."""
        self.pwm_ena.change_duty_cycle(duty)
    
    def set_duty_enb(self, duty):
        """Establece el duty cycle del motor B (ENB)."""
        self.pwm_enb.change_duty_cycle(duty)
    
    def forward(self):
        """Mueve el robot hacia adelante."""
        print("forward")
        self.lines[IN1].set_value(1)
        self.lines[IN2].set_value(0)
        self.lines[IN3].set_value(1)
        self.lines[IN4].set_value(0)
        self.pwm_ena.change_duty_cycle(PWM_FORWARD_DUTY_A)
        self.pwm_enb.change_duty_cycle(PWM_FORWARD_DUTY_B)
    
    def backward(self):
        """Mueve el robot hacia atrás."""
        print("backward")
        self.lines[IN1].set_value(0)
        self.lines[IN2].set_value(1)
        self.lines[IN3].set_value(0)
        self.lines[IN4].set_value(1)
        self.pwm_ena.change_duty_cycle(PWM_BACKWARD_DUTY_A)
        self.pwm_enb.change_duty_cycle(PWM_BACKWARD_DUTY_B)

    def turn_right(self):
        """Gira el robot a la derecha."""
        print("turn right")
        self.lines[IN1].set_value(1)
        self.lines[IN2].set_value(0)
        self.lines[IN3].set_value(0)
        self.lines[IN4].set_value(1)
        self.pwm_ena.change_duty_cycle(PWM_TURN_DUTY)
        self.pwm_enb.change_duty_cycle(PWM_TURN_DUTY)
    
    def turn_left(self):
        """Gira el robot a la izquierda."""
        print("turn left")
        self.lines[IN1].set_value(0)
        self.lines[IN2].set_value(1)
        self.lines[IN3].set_value(1)
        self.lines[IN4].set_value(0)
        self.pwm_ena.change_duty_cycle(PWM_TURN_DUTY)
        self.pwm_enb.change_duty_cycle(PWM_TURN_DUTY)
    
    def stop(self):
        """Detiene todos los motores."""
        print("stop motors")
        for pin in (IN1, IN2, IN3, IN4):
            self.lines[pin].set_value(0)
    
    def cleanup(self):
        """Limpieza de recursos."""
        self.stop()
        self.pwm_ena.stop()
        self.pwm_enb.stop()
        self.chip.close()