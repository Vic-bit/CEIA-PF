"""
Comparación de trayectorias estimadas vs Ground Truth
Módulo reutilizable para análisis de odometría visual

Calcula ATE (Absolute Trajectory Error) y RPE (Relative Pose Error)
con alineación Sim(3) - Similitud 3D con factor de escala uniforme
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Importar funciones de alineación
from src.alignment.alignment import (
    align_sim3_umeyama,
    apply_sim3_transform,
)


class TrajectoryComparison:
    """Compara trayectorias estimadas contra ground truth"""
    
    def __init__(self, ground_truth_json: str):
        """
        Inicializa con ground truth
        
        Args:
            ground_truth_json: Ruta a ground_truth_trajectory.json
        """
        self.gt_file = Path(ground_truth_json)
        self.gt_data = self._load_ground_truth()
        self.gt_trajectory = np.array([
            [x, y, z] for x, y, z in zip(
                self.gt_data['trajectory']['x'],
                self.gt_data['trajectory']['y'],
                self.gt_data['trajectory']['z']
            )
        ])
        self.results = {}
        self.alignment_params = {}
    
    def _load_ground_truth(self):
        """Carga ground truth desde JSON"""
        with open(self.gt_file, 'r') as f:
            data = json.load(f)
        print(f"✓ Ground truth cargado: {self.gt_file}")
        return data
    
    def add_estimated_trajectory(self, name: str, estimated_trajectory: np.ndarray) -> dict:
        """Agrega trayectoria estimada y calcula métricas"""
        
        # Ajustar longitudes
        min_len = min(len(self.gt_trajectory), len(estimated_trajectory))
        gt = self.gt_trajectory[:min_len]
        est = estimated_trajectory[:min_len]
        
        # Alineación Sim(3)
        R, t, s, alignment_error = align_sim3_umeyama(est, gt)
        
        # Almacenar parámetros de alineación
        self.alignment_params[name] = {
            'R': R,
            't': t,
            's': s,
            'alignment_error': alignment_error
        }
        
        # Aplicar transformación a trayectoria estimada
        est_aligned = apply_sim3_transform(est, R, t, s)
        
        # ATE: Error absoluto de posición
        ate_errors = np.linalg.norm(est_aligned - gt, axis=1)
        ate_stats = {
            'ate_mean': float(np.mean(ate_errors)),
            'ate_std': float(np.std(ate_errors)),
            'ate_rmse': float(np.sqrt(np.mean(ate_errors**2))),
            'ate_min': float(np.min(ate_errors)),
            'ate_max': float(np.max(ate_errors)),
            'ate_median': float(np.median(ate_errors))
        }
        
        # RPE: Error relativo de movimiento (5 frames)
        delta_frames = 5
        if min_len > delta_frames:
            gt_delta = gt[delta_frames:] - gt[:-delta_frames]
            est_delta = est_aligned[delta_frames:] - est_aligned[:-delta_frames]
            rpe_errors = np.linalg.norm(est_delta - gt_delta, axis=1)
            rpe_stats = {
                'rpe_mean': float(np.mean(rpe_errors)),
                'rpe_std': float(np.std(rpe_errors)),
                'rpe_rmse': float(np.sqrt(np.mean(rpe_errors**2))),
                'rpe_min': float(np.min(rpe_errors)),
                'rpe_max': float(np.max(rpe_errors)),
                'rpe_median': float(np.median(rpe_errors))
            }
        else:
            rpe_stats = {key: 0.0 for key in ['rpe_mean', 'rpe_std', 'rpe_rmse', 'rpe_min', 'rpe_max', 'rpe_median']}
        
        # Almacenar resultados
        self.results[name] = {
            'ate': ate_stats,
            'rpe': rpe_stats,
            'num_frames_compared': min_len,
            'trajectory_aligned': est_aligned,
            'trajectory_original': est,
            'scale_factor': float(s),
            'alignment_rmse': float(alignment_error)
        }
        
        return {**ate_stats, **rpe_stats, 'scale_factor': float(s)}
    
    def plot_comparison(self, output_file: str = None) -> str:
        """
        Crea visualización comparativa de trayectorias ALINEADAS
        
        Args:
            output_file: Ruta para guardar imagen
            
        Returns:
            Ruta del archivo
        """
        if not self.results:
            print("⚠️  Sin trayectorias estimadas para comparar")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Trajectory Comparison: Estimated (Scaled) vs Ground Truth\n' +
                     '(Estimated trajectories aligned via Sim(3) transformation)', 
                     fontsize=14, fontweight='bold')
        
        # 1. Vista superior (X-Z)
        ax = axes[0, 0]
        x_gt = self.gt_data['trajectory']['x']
        z_gt = self.gt_data['trajectory']['z']
        ax.plot(x_gt, z_gt, 'b-', linewidth=2.5, label='Ground Truth (KITTI)', alpha=0.8)
        
        colors = {'SIFT Classic': 'r', 'SIFT Kornia': 'g'}
        for name, data in self.results.items():
            traj = data['trajectory_aligned']
            ax.plot(traj[:, 0], traj[:, 2], '--', linewidth=2, 
                   label=f"{name} (aligned, s={data['scale_factor']:.3f})", 
                   alpha=0.7, color=colors.get(name, 'orange'))
        
        ax.set_xlabel('X (metros)', fontsize=11)
        ax.set_ylabel('Z (metros)', fontsize=11)
        ax.set_title('Top-Down View (X-Z) - AFTER Sim(3) Alignment', fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 2. Vista lateral (X-Y)
        ax = axes[0, 1]
        y_gt = self.gt_data['trajectory']['y']
        ax.plot(x_gt, y_gt, 'b-', linewidth=2.5, label='Ground Truth (KITTI)', alpha=0.8)
        
        for name, data in self.results.items():
            traj = data['trajectory_aligned']
            ax.plot(traj[:, 0], traj[:, 1], '--', linewidth=2,
                   label=f"{name} (s={data['scale_factor']:.3f})", 
                   alpha=0.7, color=colors.get(name, 'orange'))
        
        ax.set_xlabel('X (metros)', fontsize=11)
        ax.set_ylabel('Y (metros)', fontsize=11)
        ax.set_title('Side View (X-Y) - AFTER Sim(3) Alignment', fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 3. ATE por método
        ax = axes[1, 0]
        methods = list(self.results.keys())
        ate_rmse = [self.results[m]['ate']['ate_rmse'] for m in methods]
        colors_list = [colors.get(m, 'orange') for m in methods]
        
        bars = ax.bar(methods, ate_rmse, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, ate_rmse):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}m', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel('ATE RMSE (metros)', fontsize=11)
        ax.set_title('Absolute Trajectory Error (After Alignment)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. RPE por método
        ax = axes[1, 1]
        rpe_rmse = [self.results[m]['rpe']['rpe_rmse'] for m in methods]
        
        bars = ax.bar(methods, rpe_rmse, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, rpe_rmse):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}m', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel('RPE RMSE (metros)', fontsize=11)
        ax.set_title('Relative Pose Error (After Alignment)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_file is None:
            output_file = 'outputs/benchmarks/trajectory_comparison.png'
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparación guardada: {output_file}")
        return output_file
    
    def print_summary(self):
        """Imprime resumen de resultados con parámetros Sim(3)"""
        print("\n" + "="*80)
        print("RESUMEN DE MÉTRICAS DE ERROR - SLAM MONOCULAR VS GROUND TRUTH")
        print("="*80)
        print("\n⚠️  IMPORTANTE: Las trayectorias estimadas están alineadas")
        print("   (Rotación + Traslación + Escala Uniforme - Protocolo KITTI)")
        
        for name, data in self.results.items():
            params = self.alignment_params.get(name, {})
            s = params.get('s', 1.0)
            
            print(f"\n{name}:")
            print(f"  ╔═ PARÁMETROS DE ALINEACIÓN SIM(3) - UNIFORME ═════╗")
            print(f"  ║ Factor de Escala (s): {s:.6f}")
            print(f"  ║   → SLAM × {s:.4f} ≈ metros reales")
            print(f"  ║ Alignment RMSE:  {data['alignment_rmse']:.6f} m")
            print(f"  ║ Frames comparados: {data['num_frames_compared']}")
            print(f"  ║ Método: Alineación Sim(3) (Umeyama 1991)")
            print(f"  ╚════════════════════════════════════════════════════╝")
            
            print(f"\n  ATE (Absolute Trajectory Error):")
            print(f"    • RMSE:   {data['ate']['ate_rmse']:.6f} m  ← Métrica principal")
            print(f"    • Mean:   {data['ate']['ate_mean']:.6f} m")
            print(f"    • Median: {data['ate']['ate_median']:.6f} m")
            print(f"    • Std:    {data['ate']['ate_std']:.6f} m")
            print(f"    • Range:  [{data['ate']['ate_min']:.6f}, {data['ate']['ate_max']:.6f}] m")
            
            print(f"\n  RPE (Relative Pose Error - Δ5 frames):")
            print(f"    • RMSE:   {data['rpe']['rpe_rmse']:.6f} m")
            print(f"    • Mean:   {data['rpe']['rpe_mean']:.6f} m")
            print(f"    • Median: {data['rpe']['rpe_median']:.6f} m")
            print(f"    • Std:    {data['rpe']['rpe_std']:.6f} m")
            print(f"    • Range:  [{data['rpe']['rpe_min']:.6f}, {data['rpe']['rpe_max']:.6f}] m")
        
        # Comparación entre métodos
        if len(self.results) > 1:
            print(f"\n" + "="*80)
            print("COMPARACIÓN ENTRE MÉTODOS")
            print("="*80)
            
            methods = list(self.results.keys())
            ate_values = [self.results[m]['ate']['ate_rmse'] for m in methods]
            
            best_ate = np.min(ate_values)
            best_idx = np.argmin(ate_values)
            
            for i, method in enumerate(methods):
                params = self.alignment_params.get(method, {})
                s = params.get('s', 1.0)
                diff = ate_values[i] - best_ate
                ratio = ate_values[i] / best_ate if best_ate > 0 else 1.0
                symbol = "✓ MEJOR" if i == best_idx else f"({diff:+.6f}m, {ratio:.2f}x)"
                print(f"\n  {method}: ATE RMSE = {ate_values[i]:.6f} m {symbol}")
                print(f"    Factor de Escala: s = {s:.6f}")
        
        print("\n" + "="*80 + "\n")
    
    def export_results(self, output_file: str):
        """Exporta resultados como JSON (Sim(3) uniforme - Protocolo KITTI)"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'ground_truth_source': str(self.gt_file),
            'method': 'Sim(3) Uniform Scale Alignment (Rotation + Translation + Scale)',
            'description': 'Para SLAM monocular sin escala absoluta - Protocolo oficial KITTI',
            'alignment_params': {},
            'results': {}
        }
        
        # Guardar parámetros de alineación
        for name, params in self.alignment_params.items():
            s = params.get('s', 1.0)
            data['alignment_params'][name] = {
                'scale_factor': float(s),
                'rotation_matrix': params['R'].tolist(),
                'translation_vector': params['t'].tolist(),
                'alignment_rmse': float(params['alignment_error'])
            }
        
        # Guardar resultados de métricas
        for name, results in self.results.items():
            data['results'][name] = {
                'ate': results['ate'],
                'rpe': results['rpe'],
                'scale_factor': results['scale_factor'],
                'alignment_rmse': results['alignment_rmse'],
                'num_frames_compared': results['num_frames_compared']
            }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Resultados exportados: {output_file}")


def load_slam_trajectory(slam_json_file: str) -> np.ndarray:
    """
    Carga trayectoria del archivo JSON del SLAM
    
    Args:
        slam_json_file: Ruta al JSON de SLAM (contiene trayectoria)
        
    Returns:
        Array (N, 3) con poses [x, y, z]
    """
    with open(slam_json_file, 'r') as f:
        data = json.load(f)
    
    # Intentar extraer trayectoria de diferentes formatos posibles
    if 'trajectory' in data:
        traj = data['trajectory']
        n_frames = len(traj['x'])
        
        # Verificar si hay 'y' (3D) o solo 'x' y 'z' (2D monocular)
        if 'y' in traj:
            # 3D: usar x, y, z
            return np.array([[traj['x'][i], traj['y'][i], traj['z'][i]] 
                            for i in range(n_frames)])
        else:
            # 2D monocular: usar x, z, 0 para mantener dimensión 3
            return np.array([[traj['x'][i], 0.0, traj['z'][i]] 
                            for i in range(n_frames)])
    elif 'poses' in data:
        return np.array(data['poses'])
    else:
        raise ValueError(f"No se encontró trayectoria en {slam_json_file}")
