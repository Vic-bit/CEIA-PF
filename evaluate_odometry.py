"""
Evalúa el error de odometría visual comparando contra ground truth del dataset KITTI
Utiliza alineación Sim(3) (7-DoF) para SLAM monocular sin escala absoluta

Calcula: ATE (Absolute Trajectory Error), RPE (Relative Pose Error)
Reporta: Factor de escala, matrices de transformación
"""
import json
import numpy as np
from pathlib import Path
from trajectory_comparison import TrajectoryComparison, load_slam_trajectory
import sys

# Importar configuración
sys.path.insert(0, 'src')
from sift_classic.config import MAX_FRAMES


def main():
    """Evaluación completa con alineación Sim(3)"""
    
    print("\n" + "="*80)
    print("EVALUACIÓN DE ODOMETRÍA VISUAL - SLAM MONOCULAR vs GROUND TRUTH")
    print("Alineación: Sim(3) - 7 Grados de Libertad (R, t, s)")
    print("="*80)
    
    # Verificar archivos necesarios
    gt_file = Path('outputs/benchmarks/ground_truth_trajectory.json')
    classic_file = Path('outputs/benchmarks/sift_classic_trajectory.json')
    kornia_file = Path('outputs/benchmarks/sift_kornia_trajectory.json')
    
    if not gt_file.exists():
        print(f"❌ Error: {gt_file} no existe")
        print("   Ejecuta primero: python ground_truth_analysis.py")
        return
    
    # Crear comparador
    print(f"\n📂 Inicializando comparador...")
    comparator = TrajectoryComparison(str(gt_file))
    
    # Cargar e agregar trayectorias
    loaded_methods = []
    
    # SIFT Classic
    if classic_file.exists():
        print(f"\n📥 Cargando SIFT Classic...")
        try:
            classic_traj = load_slam_trajectory(str(classic_file))
            comparator.add_estimated_trajectory("SIFT Classic", classic_traj)
            loaded_methods.append("SIFT Classic")
            print(f"   ✓ {len(classic_traj)} frames cargados")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"⚠️  {classic_file} no existe")
        print(f"   Ejecuta: python src/sift_classic/main.py")
    
    # SIFT Kornia
    if kornia_file.exists():
        print(f"\n📥 Cargando SIFT Kornia...")
        try:
            kornia_traj = load_slam_trajectory(str(kornia_file))
            comparator.add_estimated_trajectory("SIFT Kornia", kornia_traj)
            loaded_methods.append("SIFT Kornia")
            print(f"   ✓ {len(kornia_traj)} frames cargados")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"⚠️  {kornia_file} no existe")
        print(f"   Ejecuta: python src/sift_kornia/main.py")
    
    if not loaded_methods:
        print(f"\n❌ No se cargaron trayectorias estimadas")
        return
    
    # Visualizar y exportar
    print(f"\n📊 Calculando métricas de error...")
    comparator.plot_comparison('outputs/benchmarks/trajectory_comparison_aligned.png')
    comparator.print_summary()
    comparator.export_results('outputs/benchmarks/trajectory_evaluation_sim3.json')
    
    print(f"\n✓ Evaluación completada")
    print(f"  - Gráfica:     outputs/benchmarks/trajectory_comparison_aligned.png")
    print(f"  - Resultados:  outputs/benchmarks/trajectory_evaluation_sim3.json")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
