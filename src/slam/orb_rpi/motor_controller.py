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
        
        # Almacenar duty cycle actual (se actualiza solo mediante sliders)
        self.current_duty_ena = INIT_DUTY
        self.current_duty_enb = INIT_DUTY
        
        # Inicializar GPIO
        self.chip = gpiod.Chip('gpiochip4')
        self.lines = {}
        for pin in (IN1, IN2, IN3, IN4):
            line = self.chip.get_line(pin)
            line.request(consumer="motor", type=gpiod.LINE_REQ_DIR_OUT)
            self.lines[pin] = line
    
    def set_duty_ena(self, duty):
        """Establece el duty cycle del motor A (ENA)."""
        self.current_duty_ena = duty
        self.pwm_ena.change_duty_cycle(duty)
    
    def set_duty_enb(self, duty):
        """Establece el duty cycle del motor B (ENB)."""
        self.current_duty_enb = duty
        self.pwm_enb.change_duty_cycle(duty)
    
    def forward(self, duty_a=None, duty_b=None):
        """Mueve el robot hacia adelante.
        
        Args:
            duty_a: Duty cycle Motor A (ENA). Si es None, usa current_duty_ena
            duty_b: Duty cycle Motor B (ENB). Si es None, usa current_duty_enb
        """
        print("forward")
        if duty_a is not None:
            self.current_duty_ena = duty_a
        if duty_b is not None:
            self.current_duty_enb = duty_b
        
        self.lines[IN1].set_value(1)
        self.lines[IN2].set_value(0)
        self.lines[IN3].set_value(1)
        self.lines[IN4].set_value(0)
        self.pwm_ena.change_duty_cycle(self.current_duty_ena)
        self.pwm_enb.change_duty_cycle(self.current_duty_enb)
    
    def backward(self, duty_a=None, duty_b=None):
        """Mueve el robot hacia atrás.
        
        Args:
            duty_a: Duty cycle Motor A (ENA). Si es None, usa current_duty_ena
            duty_b: Duty cycle Motor B (ENB). Si es None, usa current_duty_enb
        """
        print("backward")
        if duty_a is not None:
            self.current_duty_ena = duty_a
        if duty_b is not None:
            self.current_duty_enb = duty_b
        
        self.lines[IN1].set_value(0)
        self.lines[IN2].set_value(1)
        self.lines[IN3].set_value(0)
        self.lines[IN4].set_value(1)
        self.pwm_ena.change_duty_cycle(self.current_duty_ena)
        self.pwm_enb.change_duty_cycle(self.current_duty_enb)

    def turn_right(self, duty=None):
        """Gira el robot a la derecha.
        
        Args:
            duty: Duty cycle para ambos motores. Si es None, usa current_duty_ena
        """
        print("turn right")
        if duty is not None:
            self.current_duty_ena = duty
            self.current_duty_enb = duty
        
        self.lines[IN1].set_value(1)
        self.lines[IN2].set_value(0)
        self.lines[IN3].set_value(0)
        self.lines[IN4].set_value(1)
        self.pwm_ena.change_duty_cycle(self.current_duty_ena)
        self.pwm_enb.change_duty_cycle(self.current_duty_enb)
    
    def turn_left(self, duty=None):
        """Gira el robot a la izquierda.
        
        Args:
            duty: Duty cycle para ambos motores. Si es None, usa current_duty_ena
        """
        print("turn left")
        if duty is not None:
            self.current_duty_ena = duty
            self.current_duty_enb = duty
        
        self.lines[IN1].set_value(0)
        self.lines[IN2].set_value(1)
        self.lines[IN3].set_value(1)
        self.lines[IN4].set_value(0)
        self.pwm_ena.change_duty_cycle(self.current_duty_ena)
        self.pwm_enb.change_duty_cycle(self.current_duty_enb)
    
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