"""
Comparación de trayectorias estimadas vs Ground Truth
Calcula ATE (Absolute Trajectory Error) y RPE (Relative Pose Error)
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


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
    
    def _load_ground_truth(self):
        """Carga ground truth desde JSON"""
        with open(self.gt_file, 'r') as f:
            data = json.load(f)
        print(f"✓ Ground truth cargado: {self.gt_file}")
        return data
    
    def add_estimated_trajectory(self, name: str, 
                                 estimated_trajectory: np.ndarray) -> dict:
        """
        Agrega trayectoria estimada y calcula métricas
        
        Args:
            name: Nombre del método (ej: 'SIFT Classic')
            estimated_trajectory: Array (N, 3) con poses [x, y, z]
            
        Returns:
            Dict con métricas de error
        """
        # Ajustar longitudes
        min_len = min(len(self.gt_trajectory), len(estimated_trajectory))
        gt = self.gt_trajectory[:min_len]
        est = estimated_trajectory[:min_len]
        
        # ATE
        ate_errors = np.linalg.norm(est - gt, axis=1)
        ate_stats = {
            'ate_mean': float(np.mean(ate_errors)),
            'ate_std': float(np.std(ate_errors)),
            'ate_rmse': float(np.sqrt(np.mean(ate_errors**2))),
            'ate_min': float(np.min(ate_errors)),
            'ate_max': float(np.max(ate_errors))
        }
        
        # RPE (5 frames)
        delta_frames = 5
        gt_delta = gt[delta_frames:] - gt[:-delta_frames]
        est_delta = est[delta_frames:] - est[:-delta_frames]
        rpe_errors = np.linalg.norm(est_delta - gt_delta, axis=1)
        rpe_stats = {
            'rpe_mean': float(np.mean(rpe_errors)),
            'rpe_std': float(np.std(rpe_errors)),
            'rpe_rmse': float(np.sqrt(np.mean(rpe_errors**2))),
            'rpe_min': float(np.min(rpe_errors)),
            'rpe_max': float(np.max(rpe_errors))
        }
        
        # Almacenar
        self.results[name] = {
            'ate': ate_stats,
            'rpe': rpe_stats,
            'num_frames_compared': min_len,
            'trajectory': est
        }
        
        return {**ate_stats, **rpe_stats}
    
    def plot_comparison(self, output_file: str = None) -> str:
        """
        Crea visualización comparativa de trayectorias
        
        Args:
            output_file: Ruta para guardar imagen
            
        Returns:
            Ruta del archivo
        """
        if not self.results:
            print("⚠️  Sin trayectorias estimadas para comparar")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Trajectory Comparison: Estimated vs Ground Truth', 
                     fontsize=14, fontweight='bold')
        
        # 1. Vista superior (X-Z)
        ax = axes[0, 0]
        x_gt = self.gt_data['trajectory']['x']
        z_gt = self.gt_data['trajectory']['z']
        ax.plot(x_gt, z_gt, 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
        
        colors = {'SIFT Classic': 'r', 'SIFT Kornia': 'g'}
        for name, data in self.results.items():
            traj = data['trajectory']
            ax.plot(traj[:, 0], traj[:, 2], '--', linewidth=1.5, 
                   label=name, alpha=0.7, color=colors.get(name, 'orange'))
        
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Z (m)', fontsize=11)
        ax.set_title('Top-Down View (X-Z)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 2. Vista lateral (X-Y)
        ax = axes[0, 1]
        y_gt = self.gt_data['trajectory']['y']
        ax.plot(x_gt, y_gt, 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
        
        for name, data in self.results.items():
            traj = data['trajectory']
            ax.plot(traj[:, 0], traj[:, 1], '--', linewidth=1.5,
                   label=name, alpha=0.7, color=colors.get(name, 'orange'))
        
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title('Side View (X-Y)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. ATE por método
        ax = axes[1, 0]
        methods = list(self.results.keys())
        ate_rmse = [self.results[m]['ate']['ate_rmse'] for m in methods]
        colors_list = [colors.get(m, 'orange') for m in methods]
        
        bars = ax.bar(methods, ate_rmse, color=colors_list, alpha=0.7, edgecolor='black')
        for bar, val in zip(bars, ate_rmse):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}m', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('ATE RMSE (metros)', fontsize=11)
        ax.set_title('Absolute Trajectory Error (ATE)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. RPE por método
        ax = axes[1, 1]
        rpe_rmse = [self.results[m]['rpe']['rpe_rmse'] for m in methods]
        
        bars = ax.bar(methods, rpe_rmse, color=colors_list, alpha=0.7, edgecolor='black')
        for bar, val in zip(bars, rpe_rmse):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}m', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('RPE RMSE (metros)', fontsize=11)
        ax.set_title('Relative Pose Error (RPE)', fontweight='bold')
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
        """Imprime resumen de resultados"""
        print("\n" + "="*70)
        print("RESUMEN DE MÉTRICAS DE ERROR")
        print("="*70)
        
        for name, data in self.results.items():
            print(f"\n{name}:")
            print(f"  ATE (Absolute Trajectory Error):")
            print(f"    • RMSE: {data['ate']['ate_rmse']:.4f} m")
            print(f"    • Mean: {data['ate']['ate_mean']:.4f} m")
            print(f"    • Std:  {data['ate']['ate_std']:.4f} m")
            print(f"    • [Min-Max]: [{data['ate']['ate_min']:.4f}, {data['ate']['ate_max']:.4f}] m")
            
            print(f"\n  RPE (Relative Pose Error):")
            print(f"    • RMSE: {data['rpe']['rpe_rmse']:.4f} m")
            print(f"    • Mean: {data['rpe']['rpe_mean']:.4f} m")
            print(f"    • Std:  {data['rpe']['rpe_std']:.4f} m")
            print(f"    • [Min-Max]: [{data['rpe']['rpe_min']:.4f}, {data['rpe']['rpe_max']:.4f}] m")
        
        print("\n" + "="*70)
    
    def export_results(self, output_file: str):
        """Exporta resultados como JSON"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'ground_truth_source': str(self.gt_file),
            'results': self.results
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Resultados exportados: {output_file}")


# Funciones auxiliares para cargar trayectorias desde SLAM
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
        return np.array([[traj['x'][i], traj['y'][i], traj['z'][i]] 
                        for i in range(len(traj['x']))])
    elif 'poses' in data:
        return np.array(data['poses'])
    else:
        raise ValueError(f"No se encontró trayectoria en {slam_json_file}")


def main():
    """Ejemplo de uso"""
    
    print("\n" + "="*70)
    print("TRAJECTORY COMPARISON - SLAM vs Ground Truth")
    print("="*70)
    
    # Crear comparador
    gt_file = "outputs/benchmarks/ground_truth_trajectory.json"
    
    if not Path(gt_file).exists():
        print(f"⚠️  {gt_file} no existe")
        print("Ejecuta: python ground_truth_analysis.py")
        return
    
    comparator = TrajectoryComparison(gt_file)
    
    # Aquí se agregarían las trayectorias estimadas por los métodos SLAM
    # Ejemplo (datos simulados para demostración):
    
    # Simular trayectorias con pequeños errores
    print("\n📊 Cargando trayectorias estimadas...")
    
    gt_traj = comparator.gt_trajectory
    
    # SIFT Classic: pequeños errores aleatorios
    noise_classic = np.random.normal(0, 0.1, gt_traj.shape)
    slam_classic = gt_traj + noise_classic
    
    # SIFT Kornia: errores ligeramente mayores
    noise_kornia = np.random.normal(0, 0.15, gt_traj.shape)
    slam_kornia = gt_traj + noise_kornia
    
    # Agregar trayectorias
    print("\n📈 Calculando métricas de error...")
    comparator.add_estimated_trajectory("SIFT Classic", slam_classic)
    comparator.add_estimated_trajectory("SIFT Kornia", slam_kornia)
    
    # Visualizar
    comparator.plot_comparison("outputs/benchmarks/trajectory_comparison.png")
    
    # Resumen
    comparator.print_summary()
    
    # Exportar
    comparator.export_results("outputs/benchmarks/trajectory_errors.json")


if __name__ == "__main__":
    main()
