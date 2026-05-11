#!/usr/bin/env python3
"""
Script ejecutable: Generar Ground Truth desde dataset KITTI
Archivo: src/pipeline/preprocessing/generate_ground_truth.py

Uso:
    python src/pipeline/preprocessing/generate_ground_truth.py
"""

from pathlib import Path

from src.ground_truth.ground_truth import GroundTruthAnalyzer
from src.slam.sift_classic.config import MAX_FRAMES

def main():
    """Genera ground truth desde KITTI dataset 00"""
    
    print("="*70)
    print("GROUND TRUTH ANALYSIS - KITTI Dataset 00")
    print("="*70)
    
    # Rutas del dataset
    poses_file = "dataset/00/poses.txt"
    times_file = "dataset/00/times.txt"
    
    # Verificar que existan
    if not Path(poses_file).exists() or not Path(times_file).exists():
        print(f"❌ Error: Dataset no encontrado en {poses_file} o {times_file}")
        return
    
    # Crear analizador
    analyzer = GroundTruthAnalyzer(poses_file, times_file, max_frames=MAX_FRAMES)
    
    # Estadísticas
    print("\n📊 ESTADÍSTICAS DE TRAYECTORIA GROUND TRUTH")
    print("="*70)
    stats = analyzer.get_statistics()
    
    print(f"\nDataset:")
    print(f"  • Frames: {stats['num_frames']}")
    print(f"  • Duración: {stats['duration']:.2f} segundos")
    print(f"  • FPS: {stats['avg_fps']:.2f}")
    
    print(f"\nTrayectoria:")
    print(f"  • Distancia total: {stats['total_distance']:.2f} metros")
    print(f"  • Velocidad promedio: {stats['avg_velocity']:.2f} m/s")
    
    print(f"\nBoundary Box:")
    bounds = stats['trajectory_bounds']
    print(f"  • X: [{bounds['x_min']:.2f}, {bounds['x_max']:.2f}]")
    print(f"  • Y: [{bounds['y_min']:.2f}, {bounds['y_max']:.2f}]")
    print(f"  • Z: [{bounds['z_min']:.2f}, {bounds['z_max']:.2f}]")
    
    # Generar visualización
    print("\n📈 GENERANDO VISUALIZACIONES")
    print("="*70)
    analyzer.plot_topdown_view("outputs/benchmarks/ground_truth_topdown.png")
    
    # Exportar como JSON
    analyzer.export_trajectory_json("outputs/benchmarks/ground_truth_trajectory.json")
    
    print("\n" + "="*70)
    print("✓ ANÁLISIS COMPLETADO")
    print("="*70 + "\n")
    
    return analyzer


if __name__ == "__main__":
    main()
