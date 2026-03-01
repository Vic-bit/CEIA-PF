#!/usr/bin/env python3
"""
Evalúa el error de odometría visual comparando contra ground truth del dataset KITTI
Calcula ATE (Absolute Trajectory Error) y RPE (Relative Pose Error)
"""
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import sys

# Importar configuración
sys.path.insert(0, 'src')
from sift_classic.config import MAX_FRAMES

def load_ground_truth_json():
    """Carga ground truth desde JSON generado por ground_truth_analysis.py"""
    gt_file = Path('outputs/benchmarks/ground_truth_trajectory.json')
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth JSON no encontrado: {gt_file}")
    
    with open(gt_file, 'r') as f:
        data = json.load(f)
    
    gt_positions = np.array([
        [data['trajectory']['x'][i], data['trajectory']['y'][i], data['trajectory']['z'][i]]
        for i in range(len(data['trajectory']['x']))
    ])
    
    return gt_positions, data

def load_estimated_trajectory(implementation_name):
    """Carga trayectoria estimada desde JSON de benchmarks"""
    json_file = Path(f'outputs/benchmarks/{implementation_name}.json')
    if not json_file.exists():
        raise FileNotFoundError(f"Archivo de benchmarks no encontrado: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Acceder a las trayectorias si están disponibles
    if 'trajectory' in data:
        est_positions = np.array([
            [data['trajectory']['x'][i], data['trajectory']['y'][i], data['trajectory']['z'][i]]
            for i in range(len(data['trajectory']['x']))
        ])
        return est_positions, data
    else:
        print(f"⚠️  No hay trayectoria en {json_file}. Revisa que main.py exporte las poses.")
        return None, data

def compute_ate(gt_positions, est_positions):
    """
    Calcula ATE (Absolute Trajectory Error)
    Métrica de error absoluto acumulativo
    
    Args:
        gt_positions: array (N, 3) con posiciones ground truth
        est_positions: array (N, 3) con posiciones estimadas
    
    Returns:
        ate: error promedio en metros
        ate_list: errores por frame
    """
    if len(gt_positions) != len(est_positions):
        # Alinear longitudes
        min_len = min(len(gt_positions), len(est_positions))
        gt_positions = gt_positions[:min_len]
        est_positions = est_positions[:min_len]
    
    # Calcular diferencias
    differences = gt_positions - est_positions
    distances = np.linalg.norm(differences, axis=1)
    ate = np.mean(distances)
    
    return ate, distances

def compute_rpe(gt_positions, est_positions, delta=1):
    """
    Calcula RPE (Relative Pose Error)
    Métrica de error relativo entre frames consecutivos
    
    Args:
        gt_positions: array (N, 3) con posiciones ground truth
        est_positions: array (N, 3) con posiciones estimadas
        delta: número de frames para calcular error relativo (default=1)
    
    Returns:
        rpe: error relativo promedio
        rpe_list: errores relativos por frame
    """
    if len(gt_positions) != len(est_positions):
        min_len = min(len(gt_positions), len(est_positions))
        gt_positions = gt_positions[:min_len]
        est_positions = est_positions[:min_len]
    
    # Calcular desplazamientos relativos
    gt_relative = np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)
    est_relative = np.linalg.norm(np.diff(est_positions, axis=0), axis=1)
    
    # Error relativo
    rpe_errors = np.abs(gt_relative - est_relative)
    rpe = np.mean(rpe_errors)
    
    return rpe, rpe_errors

def evaluate_implementation(implementation_name, gt_positions, gt_data):
    """
    Evalúa una implementación contra ground truth
    
    Args:
        implementation_name: 'sift_classic' o 'sift_kornia'
        gt_positions: array con posiciones ground truth
        gt_data: dict con datos de ground truth
    
    Returns:
        dict con métricas de evaluación
    """
    print(f"\n{'='*70}")
    print(f"EVALUACIÓN: {implementation_name.upper()}")
    print(f"{'='*70}")
    
    try:
        est_positions, est_data = load_estimated_trajectory(implementation_name)
        if est_positions is None:
            return None
        
        # Alinear longitudes
        min_len = min(len(gt_positions), len(est_positions))
        gt_pos = gt_positions[:min_len]
        est_pos = est_positions[:min_len]
        
        # Calcular ATE
        ate, ate_list = compute_ate(gt_pos, est_pos)
        
        # Calcular RPE
        rpe, rpe_list = compute_rpe(gt_pos, est_pos)
        
        # Estadísticas
        ate_std = np.std(ate_list)
        ate_max = np.max(ate_list)
        ate_min = np.min(ate_list)
        
        rpe_std = np.std(rpe_list)
        rpe_max = np.max(rpe_list)
        rpe_min = np.min(rpe_list)
        
        # Mostrar resultados
        print(f"\n📊 MÉTRICAS DE ERROR:")
        print(f"\n🎯 ATE (Absolute Trajectory Error):")
        print(f"   Promedio:      {ate:.4f} m")
        print(f"   Std Dev:       {ate_std:.4f} m")
        print(f"   Min/Max:       {ate_min:.4f} / {ate_max:.4f} m")
        
        print(f"\n🔄 RPE (Relative Pose Error):")
        print(f"   Promedio:      {rpe:.4f} m")
        print(f"   Std Dev:       {rpe_std:.4f} m")
        print(f"   Min/Max:       {rpe_min:.4f} / {rpe_max:.4f} m")
        
        # Trayectoria
        total_distance_gt = np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))
        total_distance_est = np.sum(np.linalg.norm(np.diff(est_pos, axis=0), axis=1))
        distance_error = abs(total_distance_gt - total_distance_est)
        distance_error_percent = (distance_error / total_distance_gt) * 100
        
        print(f"\n📏 DISTANCIA RECORRIDA:")
        print(f"   Ground Truth:  {total_distance_gt:.2f} m")
        print(f"   Estimada:      {total_distance_est:.2f} m")
        print(f"   Error:         {distance_error:.2f} m ({distance_error_percent:.1f}%)")
        
        print(f"\n📈 FRAMES ANALIZADOS: {min_len}")
        
        return {
            'implementation': implementation_name,
            'num_frames': min_len,
            'ate': float(ate),
            'ate_std': float(ate_std),
            'ate_min': float(ate_min),
            'ate_max': float(ate_max),
            'rpe': float(rpe),
            'rpe_std': float(rpe_std),
            'rpe_min': float(rpe_min),
            'rpe_max': float(rpe_max),
            'total_distance_gt': float(total_distance_gt),
            'total_distance_est': float(total_distance_est),
            'distance_error': float(distance_error),
            'distance_error_percent': float(distance_error_percent),
            'ate_list': ate_list.tolist(),
            'rpe_list': rpe_list.tolist()
        }
    
    except Exception as e:
        print(f"\n❌ Error evaluando {implementation_name}: {e}")
        return None

def main():
    """Función principal de evaluación"""
    
    print("\n" + "="*70)
    print("EVALUACIÓN DE ODOMETRÍA VISUAL CONTRA GROUND TRUTH (KITTI)")
    print("="*70)
    
    # Cargar ground truth
    print(f"\n📂 Cargando Ground Truth...")
    try:
        gt_positions, gt_data = load_ground_truth_json()
        print(f"✓ Ground Truth cargado: {len(gt_positions)} frames")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\nEjecuta primero: python ground_truth_analysis.py")
        return
    
    # Evaluar implementaciones
    results = {}
    for impl in ['sift_classic', 'sift_kornia']:
        result = evaluate_implementation(impl, gt_positions, gt_data)
        if result:
            results[impl] = result
    
    # Comparación entre implementaciones
    if len(results) == 2:
        print(f"\n{'='*70}")
        print("COMPARACIÓN: SIFT Classic vs SIFT Kornia")
        print(f"{'='*70}")
        
        classic_ate = results['sift_classic']['ate']
        kornia_ate = results['sift_kornia']['ate']
        ate_diff = kornia_ate - classic_ate
        ate_ratio = kornia_ate / classic_ate if classic_ate != 0 else 1
        
        classic_rpe = results['sift_classic']['rpe']
        kornia_rpe = results['sift_kornia']['rpe']
        rpe_diff = kornia_rpe - classic_rpe
        rpe_ratio = kornia_rpe / classic_rpe if classic_rpe != 0 else 1
        
        print(f"\n📊 ATE (Absolute Trajectory Error):")
        print(f"   SIFT Classic:  {classic_ate:.4f} m")
        print(f"   SIFT Kornia:   {kornia_ate:.4f} m")
        print(f"   Diferencia:    {ate_diff:+.4f} m ({ate_ratio:.2f}x)")
        
        better_worse = "mejor" if kornia_ate < classic_ate else "peor"
        print(f"   → Kornia es {better_worse} en ATE" + (" ✓" if kornia_ate < classic_ate else " ✗"))
        
        print(f"\n🔄 RPE (Relative Pose Error):")
        print(f"   SIFT Classic:  {classic_rpe:.4f} m")
        print(f"   SIFT Kornia:   {kornia_rpe:.4f} m")
        print(f"   Diferencia:    {rpe_diff:+.4f} m ({rpe_ratio:.2f}x)")
        
        better_worse = "mejor" if kornia_rpe < classic_rpe else "peor"
        print(f"   → Kornia es {better_worse} en RPE" + (" ✓" if kornia_rpe < classic_rpe else " ✗"))
        
        print(f"\n📏 Distancia Recorrida:")
        classic_dist_err = results['sift_classic']['distance_error_percent']
        kornia_dist_err = results['sift_kornia']['distance_error_percent']
        print(f"   SIFT Classic:  {classic_dist_err:.1f}% error")
        print(f"   SIFT Kornia:   {kornia_dist_err:.1f}% error")
    
    # Guardar resultados
    output_file = Path('outputs/benchmarks/odometry_evaluation.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Resultados guardados en: {output_file}")
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
