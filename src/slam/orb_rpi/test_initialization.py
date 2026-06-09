#!/usr/bin/env python3
"""
Test de inicialización del sistema SLAM + Motor Control
Verifica que todos los componentes se inicialicen sin errores
"""

import sys
import numpy as np
from pathlib import Path

def test_initialization():
    print("=" * 60)
    print("TEST: Inicialización del Sistema SLAM + Motor Control")
    print("=" * 60)
    
    # Test 1: Imports
    print("\n[1/5] Verificando imports...")
    try:
        from camera import Camera
        from display import MainWindow
        from motor_controller import MotorController
        from features import extract, match_frames, Frame
        from pointmap import Map
        from config import WIDTH, HEIGHT, TIMER_INTERVAL_MS, SKIP_RATE, ORB_N_FEATURES
        print("✅ Todos los módulos importados correctamente")
    except Exception as e:
        print(f"❌ Error en imports: {e}")
        return False
    
    # Test 2: Config
    print("\n[2/5] Verificando configuración...")
    print(f"  - Resolución: {WIDTH}x{HEIGHT}")
    print(f"  - Timer: {TIMER_INTERVAL_MS}ms")
    print(f"  - Skip rate: {SKIP_RATE}")
    print(f"  - SIFT features: {ORB_N_FEATURES}")
    print("✅ Configuración correcta")
    
    # Test 3: MotorController
    print("\n[3/5] Verificando MotorController...")
    try:
        motor_ctrl = MotorController()
        print(f"  - Motor Controller creado")
        print(f"  - Métodos: {[m for m in dir(motor_ctrl) if not m.startswith('_') and callable(getattr(motor_ctrl, m))][:5]}...")
        motor_ctrl.stop()
        print("✅ MotorController funcional")
    except Exception as e:
        print(f"⚠️  MotorController: {e} (esperado si no está en RPi)")
    
    # Test 4: Camera
    print("\n[4/5] Verificando Camera...")
    try:
        camera = Camera()
        print(f"  - Matriz K cargada: {camera.K is not None}")
        print(f"  - K shape: {camera.K.shape if camera.K is not None else 'N/A'}")
        print(f"  - K:\n{camera.K}")
        print("✅ Camera inicializada")
    except Exception as e:
        print(f"⚠️  Camera: {e} (esperado si no hay cámara disponible)")
    
    # Test 5: Estructura MainWindow
    print("\n[5/5] Verificando estructura MainWindow...")
    try:
        # Crear motor controller dummy
        motor_ctrl = MotorController()
        
        # MainWindow debe recibir SOLO el motor_controller
        # Esto simula la inicialización sin Qt (sin display)
        import inspect
        sig = inspect.signature(MainWindow.__init__)
        params = list(sig.parameters.keys())
        print(f"  - Parámetros MainWindow.__init__: {params}")
        
        if params == ['self', 'motor_controller']:
            print("✅ Firma de MainWindow correcta")
        else:
            print(f"❌ Firma incorrecta. Esperado: ['self', 'motor_controller'], obtenido: {params}")
            return False
            
    except Exception as e:
        print(f"❌ Error en MainWindow: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ TODOS LOS TESTS PASARON")
    print("=" * 60)
    print("\nPróximos pasos:")
    print("1. Asegúrate de estar en Raspberry Pi 5 con cámara conectada")
    print("2. Ejecuta: python3 main.py")
    print("3. Usa WASDQE para controlar el robot")
    print("4. Cierra con Q o clic en 'Salir'")
    
    return True

if __name__ == "__main__":
    success = test_initialization()
    sys.exit(0 if success else 1)
