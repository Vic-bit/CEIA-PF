#!/usr/bin/env python3
"""
Script de prueba para verificar importaciones y sintaxis
"""
import sys
import os

# Agregar path
sys.path.insert(0, '/home/visualslam/PF/Github/CEIA-PF/src/sift_rpi')

print("=" * 60)
print("VERIFICACIÓN DE IMPORTS - sift_rpi")
print("=" * 60)

try:
    print("\n[1/7] Importando config...", end=" ")
    from config import WIDTH, HEIGHT, TIMER_INTERVAL_MS, SKIP_RATE
    print("✓")
    print(f"      WIDTH={WIDTH}, HEIGHT={HEIGHT}, TIMER={TIMER_INTERVAL_MS}ms, SKIP={SKIP_RATE}")
    
    print("[2/7] Importando numpy...", end=" ")
    import numpy as np
    print("✓")
    
    print("[3/7] Importando cv2...", end=" ")
    import cv2
    print("✓")
    
    print("[4/7] Importando pointmap...", end=" ")
    from pointmap import Map, Point
    print("✓")
    
    print("[5/7] Importando features...", end=" ")
    from features import Frame, match_frames, add_ones, normalize
    print("✓")
    
    print("[6/7] Importando display...", end=" ")
    from display import MainWindow
    print("✓")
    
    print("[7/7] Importando camera...", end=" ")
    from camera import Camera
    print("✓")
    
    print("\n" + "=" * 60)
    print("✅ TODOS LOS IMPORTS CORRECTOS")
    print("=" * 60)
    
    # Verificar Matrix K
    print("\n[Test] Cargando matriz de intrínsecos...")
    from utils import get_intrinsic_matrix_from_npz
    K = get_intrinsic_matrix_from_npz()
    print(f"✓ K shape: {K.shape}")
    print(f"  K =\n{K}")
    
    # Verificar Map
    print("\n[Test] Creando Map vacío...")
    map_obj = Map()
    print(f"✓ Map creado: frames={len(map_obj.frames)}, points={len(map_obj.points)}")
    
    print("\n" + "=" * 60)
    print("✅ SISTEMA LISTO PARA USAR")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
