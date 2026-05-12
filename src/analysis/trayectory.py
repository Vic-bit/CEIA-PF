"""
Comparación de trayectorias estimadas vs Ground Truth
Módulo de análisis de odometría visual

Calcula ATE (Absolute Trajectory Error) y RPE (Relative Pose Error)
tras alineación Sim(3) según Umeyama (1991).

Referencia de alineación:
    S. Umeyama, "Least-squares estimation of transformation parameters
    between two point patterns", IEEE TPAMI, 13(4):376-380, 1991.

Métricas implementadas:
    ATE — Absolute Trajectory Error: error euclidiano pose a pose entre
          la trayectoria estimada (alineada) y el ground truth.
    RPE — Relative Pose Error: error de movimiento relativo en ventanas
          de Δ frames consecutivos; captura la consistencia local.

Protocolo de evaluación: el estándar KITTI para SLAM monocular exige
alineación Sim(3) (7-DoF) antes de calcular cualquier métrica, porque
la escala absoluta es no-observable en monocular puro.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from src.analysis.alignment import align_sim3_umeyama, apply_sim3_transform


class TrajectoryComparison:
    """
    Compara trayectorias SLAM estimadas contra ground truth KITTI.

    Flujo de uso:
        comp = TrajectoryComparison("outputs/benchmarks/ground_truth_trajectory.json")
        comp.add_estimated_trajectory("SIFT Classic", traj_array)
        comp.add_estimated_trajectory("SIFT Kornia",  traj_array)
        comp.plot_comparison("outputs/benchmarks/trajectory_comparison_aligned.png")
        comp.print_summary()
        comp.export_results("outputs/benchmarks/trajectory_evaluation_sim3.json")
    """

    def __init__(self, ground_truth_json: str):
        """
        Args:
            ground_truth_json: ruta a ground_truth_trajectory.json
        """
        self.gt_file = Path(ground_truth_json)
        self.gt_data = self._load_ground_truth()
        self.gt_trajectory = np.array([
            [x, y, z] for x, y, z in zip(
                self.gt_data['trajectory']['x'],
                self.gt_data['trajectory']['y'],
                self.gt_data['trajectory']['z'],
            )
        ])
        self.results = {}
        self.alignment_params = {}

    # ── Carga ────────────────────────────────────────────────────────────────

    def _load_ground_truth(self) -> dict:
        with open(self.gt_file, 'r') as f:
            data = json.load(f)
        print(f"Ground truth cargado: {self.gt_file}")
        return data

    # ── Evaluación ───────────────────────────────────────────────────────────

    def add_estimated_trajectory(self,
                                 name: str,
                                 estimated_trajectory: np.ndarray) -> dict:
        """
        Registra una trayectoria estimada y calcula sus métricas.

        Pasos internos:
            1. Recorta ambas trayectorias a la longitud mínima común.
            2. Alinea con Sim(3) via Umeyama (1991).
            3. Calcula ATE sobre la trayectoria alineada.
            4. Calcula RPE con ventana de Δ=5 frames.

        Args:
            name                 : identificador del método (p.ej. "SIFT Classic")
            estimated_trajectory : array (N, 3) con poses [x, y, z]

        Returns:
            dict con todas las métricas ATE + RPE + scale_factor
        """
        # 1. Alinear longitudes
        min_len = min(len(self.gt_trajectory), len(estimated_trajectory))
        gt  = self.gt_trajectory[:min_len]
        est = estimated_trajectory[:min_len]

        # 2. Alineación Sim(3) — Umeyama 1991
        #    Devuelve (R, t, s, rmse_alineacion)
        #    La transformación óptima minimiza (1/n)·Σ‖gt_i − (s·R·est_i + t)‖²
        R, t, s, alignment_rmse = align_sim3_umeyama(est, gt)

        self.alignment_params[name] = {
            'R': R,
            't': t,
            's': s,
            'alignment_rmse': alignment_rmse,
        }

        est_aligned = apply_sim3_transform(est, R, t, s)

        # 3. ATE — Absolute Trajectory Error
        #    ate_i = ‖ gt_i − est_aligned_i ‖₂
        ate_errors = np.linalg.norm(est_aligned - gt, axis=1)
        ate_stats = {
            'ate_mean'  : float(np.mean(ate_errors)),
            'ate_std'   : float(np.std(ate_errors)),
            'ate_rmse'  : float(np.sqrt(np.mean(ate_errors ** 2))),
            'ate_min'   : float(np.min(ate_errors)),
            'ate_max'   : float(np.max(ate_errors)),
            'ate_median': float(np.median(ate_errors)),
        }

        # 4. RPE — Relative Pose Error  (ventana Δ = 5 frames)
        #    rpe_i = ‖ (gt_{i+Δ} − gt_i) − (est_{i+Δ} − est_i) ‖₂
        delta = 5
        if min_len > delta:
            gt_delta  = gt[delta:]  - gt[:-delta]
            est_delta = est_aligned[delta:] - est_aligned[:-delta]
            rpe_errors = np.linalg.norm(est_delta - gt_delta, axis=1)
            rpe_stats = {
                'rpe_mean'  : float(np.mean(rpe_errors)),
                'rpe_std'   : float(np.std(rpe_errors)),
                'rpe_rmse'  : float(np.sqrt(np.mean(rpe_errors ** 2))),
                'rpe_min'   : float(np.min(rpe_errors)),
                'rpe_max'   : float(np.max(rpe_errors)),
                'rpe_median': float(np.median(rpe_errors)),
            }
        else:
            rpe_stats = {k: 0.0 for k in [
                'rpe_mean', 'rpe_std', 'rpe_rmse',
                'rpe_min', 'rpe_max', 'rpe_median',
            ]}

        self.results[name] = {
            'ate'                : ate_stats,
            'rpe'                : rpe_stats,
            'num_frames_compared': min_len,
            'trajectory_aligned' : est_aligned,
            'trajectory_original': est,
            'scale_factor'       : float(s),
            'alignment_rmse'     : float(alignment_rmse),
        }

        return {**ate_stats, **rpe_stats, 'scale_factor': float(s)}

    # ── Visualización ─────────────────────────────────────────────────────────

    def plot_comparison(self, output_file: str = None) -> str:
        """
        Genera figura 2×2 con trayectorias alineadas y métricas ATE/RPE.

        Paneles:
            [0,0] Vista superior X-Z    [0,1] Vista lateral X-Y
            [1,0] ATE RMSE por método   [1,1] RPE RMSE por método

        Args:
            output_file: ruta de salida; por defecto
                         'outputs/benchmarks/trajectory_comparison.png'

        Returns:
            Ruta del archivo guardado.
        """
        if not self.results:
            print("Sin trayectorias estimadas para comparar.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(
            'Trajectory comparison: estimated (Sim(3) aligned) vs ground truth\n'
            'Alignment: Umeyama (1991) — 7-DoF Sim(3)',
            fontsize=14, fontweight='bold',
        )

        colors = {'SIFT Classic': 'r', 'SIFT Kornia': 'g'}

        x_gt = self.gt_data['trajectory']['x']
        y_gt = self.gt_data['trajectory']['y']
        z_gt = self.gt_data['trajectory']['z']

        # Panel [0,0]: vista superior X-Z
        ax = axes[0, 0]
        ax.plot(x_gt, z_gt, 'b-', linewidth=2.5,
                label='Ground truth (KITTI)', alpha=0.8)
        for name, data in self.results.items():
            traj = data['trajectory_aligned']
            ax.plot(traj[:, 0], traj[:, 2], '--', linewidth=2,
                    label=f"{name}  (s={data['scale_factor']:.3f})",
                    alpha=0.7, color=colors.get(name, 'orange'))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title('Top-down view (X-Z) — after Sim(3) alignment',
                     fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Panel [0,1]: vista lateral X-Y
        ax = axes[0, 1]
        ax.plot(x_gt, y_gt, 'b-', linewidth=2.5,
                label='Ground truth (KITTI)', alpha=0.8)
        for name, data in self.results.items():
            traj = data['trajectory_aligned']
            ax.plot(traj[:, 0], traj[:, 1], '--', linewidth=2,
                    label=f"{name}  (s={data['scale_factor']:.3f})",
                    alpha=0.7, color=colors.get(name, 'orange'))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Side view (X-Y) — after Sim(3) alignment',
                     fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel [1,0]: ATE RMSE
        ax = axes[1, 0]
        methods     = list(self.results.keys())
        ate_rmse    = [self.results[m]['ate']['ate_rmse'] for m in methods]
        color_list  = [colors.get(m, 'orange') for m in methods]
        bars = ax.bar(methods, ate_rmse, color=color_list,
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, ate_rmse):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f} m', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
        ax.set_ylabel('ATE RMSE (m)')
        ax.set_title('Absolute Trajectory Error — after alignment',
                     fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Panel [1,1]: RPE RMSE
        ax = axes[1, 1]
        rpe_rmse = [self.results[m]['rpe']['rpe_rmse'] for m in methods]
        bars = ax.bar(methods, rpe_rmse, color=color_list,
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, rpe_rmse):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f} m', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
        ax.set_ylabel('RPE RMSE (m)')
        ax.set_title('Relative Pose Error (Δ=5 frames) — after alignment',
                     fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_file is None:
            output_file = 'outputs/benchmarks/trajectory_comparison.png'

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparación guardada: {output_file}")
        return output_file

    # ── Reporte por consola ───────────────────────────────────────────────────

    def print_summary(self):
        """Imprime resumen de métricas y parámetros Sim(3) por método."""
        sep = "=" * 78
        print(f"\n{sep}")
        print("RESUMEN DE MÉTRICAS — SLAM MONOCULAR VS GROUND TRUTH KITTI")
        print(sep)
        print("Alineación: Sim(3) Umeyama (1991) — protocolo oficial KITTI")
        print("Nota: todas las métricas se calculan DESPUÉS de la alineación.\n")

        for name, data in self.results.items():
            params = self.alignment_params[name]
            s = params['s']

            print(f"{'─'*78}")
            print(f"Método: {name}")
            print(f"{'─'*78}")

            print(f"\n  Parámetros Sim(3) — Umeyama (1991)")
            print(f"    Factor de escala s : {s:.6f}  "
                  f"(SLAM × {s:.4f} ≈ metros reales)")
            print(f"    RMSE de alineación : {data['alignment_rmse']:.6f} m")
            print(f"    Frames comparados  : {data['num_frames_compared']}")

            print(f"\n  ATE — Absolute Trajectory Error")
            print(f"    RMSE   : {data['ate']['ate_rmse']:.6f} m  ← métrica principal")
            print(f"    Media  : {data['ate']['ate_mean']:.6f} m")
            print(f"    Mediana: {data['ate']['ate_median']:.6f} m")
            print(f"    Std    : {data['ate']['ate_std']:.6f} m")
            print(f"    Rango  : [{data['ate']['ate_min']:.6f}, "
                  f"{data['ate']['ate_max']:.6f}] m")

            print(f"\n  RPE — Relative Pose Error  (Δ = 5 frames)")
            print(f"    RMSE   : {data['rpe']['rpe_rmse']:.6f} m")
            print(f"    Media  : {data['rpe']['rpe_mean']:.6f} m")
            print(f"    Mediana: {data['rpe']['rpe_median']:.6f} m")
            print(f"    Std    : {data['rpe']['rpe_std']:.6f} m")
            print(f"    Rango  : [{data['rpe']['rpe_min']:.6f}, "
                  f"{data['rpe']['rpe_max']:.6f}] m")

        if len(self.results) > 1:
            print(f"\n{'─'*78}")
            print("Comparación entre métodos")
            print(f"{'─'*78}")
            methods    = list(self.results.keys())
            ate_values = [self.results[m]['ate']['ate_rmse'] for m in methods]
            best_idx   = int(np.argmin(ate_values))

            for i, method in enumerate(methods):
                s    = self.alignment_params[method]['s']
                diff = ate_values[i] - ate_values[best_idx]
                tag  = "MEJOR" if i == best_idx else f"+{diff:.6f} m"
                print(f"\n  {method}")
                print(f"    ATE RMSE       : {ate_values[i]:.6f} m  [{tag}]")
                print(f"    Factor escala s: {s:.6f}")

        print(f"\n{sep}\n")

    # ── Exportación ───────────────────────────────────────────────────────────

    def export_results(self, output_file: str):
        """
        Exporta todos los resultados a JSON.

        Campos exportados por método:
            alignment_params : R, t, s, rmse de alineación
            results          : métricas ATE y RPE completas
        """
        data = {
            'timestamp'           : datetime.now().isoformat(),
            'ground_truth_source' : str(self.gt_file),
            'alignment_method'    : 'Sim(3) Umeyama (1991) — 7-DoF',
            'alignment_reference' : (
                'S. Umeyama, Least-squares estimation of transformation '
                'parameters between two point patterns, '
                'IEEE TPAMI 13(4):376-380, 1991.'
            ),
            'rpe_delta_frames'    : 5,
            'alignment_params'    : {},
            'results'             : {},
        }

        for name, params in self.alignment_params.items():
            data['alignment_params'][name] = {
                'scale_factor'     : float(params['s']),
                'rotation_matrix'  : params['R'].tolist(),
                'translation_vector': params['t'].tolist(),
                'alignment_rmse'   : float(params['alignment_rmse']),
            }

        for name, res in self.results.items():
            data['results'][name] = {
                'ate'                : res['ate'],
                'rpe'                : res['rpe'],
                'scale_factor'       : res['scale_factor'],
                'alignment_rmse'     : res['alignment_rmse'],
                'num_frames_compared': res['num_frames_compared'],
            }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Resultados exportados: {output_file}")


# ── Helper standalone ─────────────────────────────────────────────────────────

def load_slam_trajectory(slam_json_file: str) -> np.ndarray:
    """
    Carga una trayectoria SLAM desde JSON.

    Soporta dos formatos:
        - {'trajectory': {'x': [...], 'y': [...], 'z': [...]}}
        - {'trajectory': {'x': [...], 'z': [...]}}   (monocular 2D → y=0)
        - {'poses': [[x, y, z], ...]}

    Args:
        slam_json_file: ruta al JSON de salida del SLAM

    Returns:
        Array (N, 3) con poses [x, y, z]
    """
    with open(slam_json_file, 'r') as f:
        data = json.load(f)

    if 'trajectory' in data:
        traj    = data['trajectory']
        n       = len(traj['x'])
        if 'y' in traj:
            return np.array([[traj['x'][i], traj['y'][i], traj['z'][i]]
                             for i in range(n)])
        else:
            # SLAM monocular 2D: y se fija a 0 para mantener dimensión 3D
            return np.array([[traj['x'][i], 0.0, traj['z'][i]]
                             for i in range(n)])

    if 'poses' in data:
        return np.array(data['poses'])

    raise ValueError(
        f"Formato no reconocido en {slam_json_file}. "
        "Se esperaba clave 'trajectory' o 'poses'."
    )