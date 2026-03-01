#!/usr/bin/env python3
"""
Ground Truth Analysis - Lectura de poses.txt y times.txt del dataset KITTI
para crear ground truth y calcular métricas de error (ATE, RPE)

La variable MAX_FRAMES debe coincidir con la de config.py para análisis consistente
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import json
from datetime import datetime

# CONFIGURACIÓN: Cambiar MAX_FRAMES para que coincida con config.py
# None = usar todos los frames, o especificar número máximo (ej: 200)
MAX_FRAMES = 250
class GroundTruthAnalyzer:
    """Análisis de poses ground truth del dataset KITTI"""
    
    def __init__(self, poses_file: str, times_file: str):
        """
        Inicializa con rutas a poses.txt y times.txt
        
        Args:
            poses_file: Ruta a dataset/00/poses.txt
            times_file: Ruta a dataset/00/times.txt
        """
        self.poses_file = Path(poses_file)
        self.times_file = Path(times_file)
        
        self.poses = None  # Lista de matrices 4x4
        self.times = None  # Timestamps
        self.trajectory = None  # Trayectoria 3D (x, y, z)
        
        self._load_data()
    
    def _load_data(self):
        """Carga poses.txt y times.txt"""
        print(f"\n📂 Cargando ground truth desde:")
        print(f"   - {self.poses_file}")
        print(f"   - {self.times_file}")
        
        # Leer poses.txt
        poses_data = np.loadtxt(self.poses_file)
        num_frames = poses_data.shape[0]
        
        # Limitar a MAX_FRAMES si está configurado
        if MAX_FRAMES is not None:
            num_frames = min(num_frames, MAX_FRAMES)
            poses_data = poses_data[:num_frames]
        
        # Convertir de 3x4 (vectorizado) a 4x4
        self.poses = []
        for i in range(num_frames):
            # Cada fila tiene 12 elementos: [R11 R12 R13 t1 R21 R22 R23 t2 R31 R32 R33 t3]
            pose_3x4 = poses_data[i].reshape(3, 4)
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :4] = pose_3x4
            self.poses.append(pose_4x4)
        
        # Leer times.txt
        times_data = np.loadtxt(self.times_file)
        # Limitar también los tiempos
        if MAX_FRAMES is not None:
            times_data = times_data[:num_frames]
        self.times = times_data
        
        # Extraer trayectoria (posiciones x, y, z)
        self.trajectory = np.array([pose[:3, 3] for pose in self.poses])
        
        print(f"   ✓ Cargados {num_frames} frames")
        if MAX_FRAMES is not None:
            print(f"   ℹ Limitado a MAX_FRAMES={MAX_FRAMES}")
        print(f"   ✓ Trayectoria: {self.trajectory.shape}")
    
    def get_trajectory_topdown(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtiene la proyección de la trayectoria vista superior (X-Z)
        
        Returns:
            Tupla (x_coords, z_coords)
        """
        x = self.trajectory[:, 0]
        z = self.trajectory[:, 2]
        return x, z
    
    def plot_topdown_view(self, output_file: str = None) -> str:
        """
        Crea visualización superior de la trayectoria ground truth
        
        Args:
            output_file: Ruta para guardar la imagen (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        x, z = self.get_trajectory_topdown()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plotear trayectoria
        ax.plot(x, z, 'b-', linewidth=2, label='Ground Truth Trajectory')
        
        # Marcar inicio y fin
        ax.plot(x[0], z[0], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(x[-1], z[-1], 'ro', markersize=12, label='End', zorder=5)
        
        # Plotear cada N-ésimo frame
        step = max(1, len(x) // 50)  # Aproximadamente 50 puntos
        for i in range(0, len(x), step):
            ax.plot(x[i], z[i], 'b.', markersize=4, alpha=0.5)
        
        ax.set_xlabel('X (metros)', fontsize=12)
        ax.set_ylabel('Z (metros)', fontsize=12)
        ax.set_title('Ground Truth Trajectory - Top-Down View (X-Z)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.axis('equal')
        
        # Guardar
        if output_file is None:
            output_file = 'outputs/benchmarks/ground_truth_topdown.png'
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Gráfico superior guardado: {output_file}")
        return str(output_path)
    
    def calculate_ate(self, estimated_trajectory: np.ndarray) -> dict:
        """
        Calcula Absolute Trajectory Error (ATE)
        
        Args:
            estimated_trajectory: Array de forma (N, 3) con poses estimadas [x, y, z]
            
        Returns:
            Dict con estadísticas de ATE
        """
        if len(estimated_trajectory) != len(self.trajectory):
            print(f"⚠️  Warning: Longitudes diferentes")
            print(f"   Ground truth: {len(self.trajectory)}")
            print(f"   Estimada: {len(estimated_trajectory)}")
            # Ajustar a la longitud más corta
            min_len = min(len(self.trajectory), len(estimated_trajectory))
            gt = self.trajectory[:min_len]
            est = estimated_trajectory[:min_len]
        else:
            gt = self.trajectory
            est = estimated_trajectory
        
        # Calcular diferencias
        differences = est - gt
        distances = np.linalg.norm(differences, axis=1)
        
        results = {
            'ate_mean': float(np.mean(distances)),
            'ate_std': float(np.std(distances)),
            'ate_min': float(np.min(distances)),
            'ate_max': float(np.max(distances)),
            'ate_rmse': float(np.sqrt(np.mean(distances**2)))
        }
        
        return results
    
    def calculate_rpe(self, estimated_trajectory: np.ndarray, 
                     delta_frames: int = 5) -> dict:
        """
        Calcula Relative Pose Error (RPE)
        
        Args:
            estimated_trajectory: Array de forma (N, 3)
            delta_frames: Número de frames para calcular error relativo
            
        Returns:
            Dict con estadísticas de RPE
        """
        min_len = min(len(self.trajectory), len(estimated_trajectory))
        gt = self.trajectory[:min_len]
        est = estimated_trajectory[:min_len]
        
        # Diferencias entre frames consecutivos
        gt_delta = gt[delta_frames:] - gt[:-delta_frames]
        est_delta = est[delta_frames:] - est[:-delta_frames]
        
        # Error en desplazamientos
        delta_diff = est_delta - gt_delta
        delta_distances = np.linalg.norm(delta_diff, axis=1)
        
        results = {
            'rpe_mean': float(np.mean(delta_distances)),
            'rpe_std': float(np.std(delta_distances)),
            'rpe_min': float(np.min(delta_distances)),
            'rpe_max': float(np.max(delta_distances)),
            'rpe_rmse': float(np.sqrt(np.mean(delta_distances**2)))
        }
        
        return results
    
    def get_statistics(self) -> dict:
        """Obtiene estadísticas de la trayectoria ground truth"""
        x, z = self.get_trajectory_topdown()
        
        # Distancia total
        distances = np.linalg.norm(
            np.diff(self.trajectory, axis=0), 
            axis=1
        )
        total_distance = np.sum(distances)
        
        stats = {
            'num_frames': len(self.trajectory),
            'duration': float(self.times[-1] - self.times[0]),
            'avg_fps': len(self.trajectory) / (self.times[-1] - self.times[0]),
            'total_distance': float(total_distance),
            'avg_velocity': float(total_distance / (self.times[-1] - self.times[0])),
            'trajectory_bounds': {
                'x_min': float(x.min()),
                'x_max': float(x.max()),
                'z_min': float(z.min()),
                'z_max': float(z.max()),
                'y_min': float(self.trajectory[:, 1].min()),
                'y_max': float(self.trajectory[:, 1].max())
            }
        }
        
        return stats
    
    def export_trajectory_json(self, output_file: str) -> str:
        """Exporta trayectoria como JSON para comparación"""
        x, z = self.get_trajectory_topdown()
        
        data = {
            'metadata': {
                'source': 'KITTI Dataset 00',
                'type': 'ground_truth',
                'num_frames': len(self.trajectory)
            },
            'trajectory': {
                'x': x.tolist(),
                'y': self.trajectory[:, 1].tolist(),
                'z': z.tolist(),
                'times': self.times.tolist()
            },
            'statistics': self.get_statistics()
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Trayectoria ground truth exportada: {output_file}")
        return str(output_path)


def main():
    """Demostración de uso"""
    
    print("="*70)
    print("GROUND TRUTH ANALYSIS - KITTI Dataset 00")
    print("="*70)
    
    # Rutas relativas a este script
    poses_file = "dataset/00/poses.txt"
    times_file = "dataset/00/times.txt"
    
    # Crear analizador
    analyzer = GroundTruthAnalyzer(poses_file, times_file)
    
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
    print("="*70)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
