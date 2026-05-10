#!/usr/bin/env python3
"""
Main Orchestrator - Visual SLAM Comparación SIFT Classic vs Kornia
Punto de entrada para ejecutar el pipeline completo

Uso:
    python main.py [opción]

Opciones:
    ground_truth    - Generar ground truth desde KITTI
    evaluate        - Evaluar trayectorias SLAM
    full            - Pipeline completo (GT + evaluación)
"""

import sys
import subprocess
from pathlib import Path


def run_ground_truth():
    """Ejecuta generación de ground truth"""
    print("\n🚀 Ejecutando: Generación de Ground Truth")
    print("-" * 80)
    result = subprocess.run([sys.executable, "src/pipeline/preprocessing/generate_ground_truth.py"], cwd=Path(__file__).parent)
    return result.returncode == 0


def run_evaluate():
    """Ejecuta evaluación de odometría"""
    print("\n🚀 Ejecutando: Evaluación de Odometría")
    print("-" * 80)
    result = subprocess.run([sys.executable, "src/pipeline/evaluation/evaluate_slam.py"], cwd=Path(__file__).parent)
    return result.returncode == 0


def run_slam_classic():
    """Ejecuta SIFT Classic"""
    print("\n🚀 Ejecutando: SIFT Classic (GUI)")
    print("-" * 80)
    result = subprocess.run([sys.executable, "src/slam/sift_classic/main.py"], cwd=Path(__file__).parent)
    return result.returncode == 0


def run_slam_kornia():
    """Ejecuta SIFT Kornia"""
    print("\n🚀 Ejecutando: SIFT Kornia (GUI)")
    print("-" * 80)
    result = subprocess.run([sys.executable, "src/slam/sift_kornia/main.py"], cwd=Path(__file__).parent)
    return result.returncode == 0


def show_help():
    """Muestra ayuda"""
    print(__doc__)


def main():
    """Función principal"""
    
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nEjecución rápida:")
        print("  python main.py ground_truth  # Generar GT")
        print("  python main.py evaluate       # Evaluar SLAM")
        print("")
        return
    
    command = sys.argv[1].lower()
    
    if command == "ground_truth":
        run_ground_truth()
    
    elif command == "evaluate":
        run_evaluate()
    
    elif command == "classic":
        run_slam_classic()
    
    elif command == "kornia":
        run_slam_kornia()
    
    elif command == "full":
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETO - VISUAL SLAM")
        print("=" * 80)
        
        success = True
        
        # 1. Ground truth
        if not run_ground_truth():
            print("❌ Error en generación de ground truth")
            return
        
        # 2. SIFT Classic
        print("\n⏭️  Nota: Se abrirá interfaz GUI de SIFT Classic")
        if not run_slam_classic():
            print("⚠️  SIFT Classic finalizado")
        
        # 3. SIFT Kornia  
        print("\n⏭️  Nota: Se abrirá interfaz GUI de SIFT Kornia")
        if not run_slam_kornia():
            print("⚠️  SIFT Kornia finalizado")
        
        # 4. Evaluación
        if not run_evaluate():
            print("❌ Error en evaluación")
            return
        
        print("\n" + "=" * 80)
        print("✓ PIPELINE COMPLETADO")
        print("=" * 80)
        print("\nResultados en: outputs/benchmarks/")
    
    elif command == "-h" or command == "--help":
        show_help()
    
    else:
        print(f"❌ Comando desconocido: {command}")
        show_help()


if __name__ == "__main__":
    main()
