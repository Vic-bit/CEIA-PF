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

from src.evaluation.alignment import align_sim3_umeyama, apply_sim3_transform


class TrajectoryComparison:
    """
    Compara trayectorias SLAM estimadas contra ground truth KITTI.

    Flujo de uso:
        comp = TrajectoryComparison("outputs/benchmarks/ground_truth_trajectory.json")
        comp.add_estimated_trajectory("SIFT Classic", traj_array)
        comp.add_estimated_trajectory("SIFT Kornia",  traj_array)
        comp.plot_comparison("outputs/benchmarks/slam_analysis_complete.png")
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
        Genera figura 4×2 con el análisis completo de trayectorias.

        Layout (Classic izquierda, Kornia derecha):
            Fila 0 — sin alineación : raw vs GT        | raw vs GT
            Fila 1 — Sim(3) global  : alineado vs GT   | alineado vs GT
            Fila 2 — desde origen   : visual (offset)  | visual (offset)
            Fila 3 — métricas       : ATE por frame     | barras ATE + RPE

        Notas:
            - Filas 0, 1 y 2 comparten los mismos límites de ejes,
              calculados sobre raw + alineadas, para que nada se corte
              y la corrección entre filas sea visualmente comparable.
            - Fila 2 es solo visual: el offset al origen no cambia las
              métricas, que siguen siendo las del Sim(3) global.

        Args:
            output_file: ruta de salida. Por defecto
                         'outputs/benchmarks/slam_analysis_complete.png'

        Returns:
            Ruta del archivo guardado.
        """
        if not self.results:
            print("Sin trayectorias estimadas para comparar.")
            return None

        methods = list(self.results.keys())
        if len(methods) < 2:
            print("plot_comparison requiere al menos dos métodos registrados.")
            return None

        # ── Extraer datos ─────────────────────────────────────────────────────
        name_c,   name_k   = methods[0],   methods[1]
        data_c,   data_k   = self.results[name_c],        self.results[name_k]
        params_c, params_k = self.alignment_params[name_c], self.alignment_params[name_k]

        gt        = self.gt_trajectory
        raw_c     = data_c['trajectory_original']
        raw_k     = data_k['trajectory_original']
        aligned_c = data_c['trajectory_aligned']
        aligned_k = data_k['trajectory_aligned']
        ate_c     = data_c['ate']
        ate_k     = data_k['ate']
        rpe_c     = data_c['rpe']
        rpe_k     = data_k['rpe']
        s_c       = params_c['s']
        s_k       = params_k['s']
        n         = data_c['num_frames_compared']

        # Offset visual fila 2: frame 0 coincide con GT[0]
        origin_c = aligned_c + (gt[0] - aligned_c[0])
        origin_k = aligned_k + (gt[0] - aligned_k[0])

        # ── Límites globales compartidos (filas 0-2) ──────────────────────────
        all_x = np.concatenate([gt[:, 0],
                                 raw_c[:, 0],     raw_k[:, 0],
                                 aligned_c[:, 0], aligned_k[:, 0]])
        all_z = np.concatenate([gt[:, 2],
                                 raw_c[:, 2],     raw_k[:, 2],
                                 aligned_c[:, 2], aligned_k[:, 2]])
        mx    = (all_x.max() - all_x.min()) * 0.1
        mz    = (all_z.max() - all_z.min()) * 0.1
        x_lim = [all_x.min() - mx, all_x.max() + mx]
        z_lim = [all_z.min() - mz, all_z.max() + mz]

        # ── Figura ────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(4, 2, figsize=(14, 24))
        fig.suptitle(
            f'Evaluación SLAM Monocular — KITTI  ({n} frames)\n'
            'Alineación Sim(3) — Umeyama (1991)',
            fontsize=14, fontweight='bold',
        )

        # ── Fila 0: sin alineación ────────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(gt[:, 0], gt[:, 2], 'b-', linewidth=2.5,
                label='Ground Truth', alpha=0.8)
        ax.plot(raw_c[:, 0], raw_c[:, 2], 'r--', linewidth=1.5,
                label=f'{name_c} (raw)', alpha=0.7)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'{name_c} — sin alineación', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim)

        ax = axes[0, 1]
        ax.plot(gt[:, 0], gt[:, 2], 'b-', linewidth=2.5,
                label='Ground Truth', alpha=0.8)
        ax.plot(raw_k[:, 0], raw_k[:, 2], color='orange', linestyle='--',
                linewidth=1.5, label=f'{name_k} (raw)', alpha=0.7)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'{name_k} — sin alineación', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim)

        # ── Fila 1: Sim(3) global ─────────────────────────────────────────────
        ax = axes[1, 0]
        ax.plot(gt[:, 0], gt[:, 2], 'b-', linewidth=2.5,
                label='Ground Truth', alpha=0.8)
        ax.plot(aligned_c[:, 0], aligned_c[:, 2], 'g-', linewidth=2,
                label=f'{name_c}  (s={s_c:.3f})', alpha=0.9)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'{name_c} — Sim(3) global\n'
                     f'ATE RMSE = {ate_c["ate_rmse"]:.3f} m',
                     fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim)

        ax = axes[1, 1]
        ax.plot(gt[:, 0], gt[:, 2], 'b-', linewidth=2.5,
                label='Ground Truth', alpha=0.8)
        ax.plot(aligned_k[:, 0], aligned_k[:, 2], 'purple', linewidth=2,
                label=f'{name_k}  (s={s_k:.3f})', alpha=0.9)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'{name_k} — Sim(3) global\n'
                     f'ATE RMSE = {ate_k["ate_rmse"]:.3f} m',
                     fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim)

        # ── Fila 2: desde origen (solo visual) ────────────────────────────────
        ax = axes[2, 0]
        ax.plot(gt[:, 0], gt[:, 2], 'b-', linewidth=2.5,
                label='Ground Truth', alpha=0.8)
        ax.plot(origin_c[:, 0], origin_c[:, 2], 'g-', linewidth=2,
                label=f'{name_c}  (s={s_c:.3f})', alpha=0.9)
        ax.plot(gt[0, 0], gt[0, 2], 'ko', markersize=8,
                label='Inicio común', zorder=5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'{name_c} — desde origen  (solo visual)\n'
                     f'ATE RMSE = {ate_c["ate_rmse"]:.3f} m  '
                     r'$\leftarrow$ calculado sobre Sim(3) global',
                     fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim)

        ax = axes[2, 1]
        ax.plot(gt[:, 0], gt[:, 2], 'b-', linewidth=2.5,
                label='Ground Truth', alpha=0.8)
        ax.plot(origin_k[:, 0], origin_k[:, 2], 'purple', linewidth=2,
                label=f'{name_k}  (s={s_k:.3f})', alpha=0.9)
        ax.plot(gt[0, 0], gt[0, 2], 'ko', markersize=8,
                label='Inicio común', zorder=5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'{name_k} — desde origen  (solo visual)\n'
                     f'ATE RMSE = {ate_k["ate_rmse"]:.3f} m  '
                     r'$\leftarrow$ calculado sobre Sim(3) global',
                     fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim)

        # ── Fila 3: métricas ──────────────────────────────────────────────────

        # [3,0] ATE por frame
        ax     = axes[3, 0]
        frames = np.arange(n)
        err_c  = np.linalg.norm(aligned_c - gt[:n], axis=1)
        err_k  = np.linalg.norm(aligned_k - gt[:n], axis=1)

        ax.plot(frames, err_c, 'g-',          linewidth=1.5, alpha=0.8,
                label=f'{name_c}  (RMSE={ate_c["ate_rmse"]:.3f} m)')
        ax.plot(frames, err_k, color='purple', linewidth=1.5, alpha=0.8,
                label=f'{name_k}  (RMSE={ate_k["ate_rmse"]:.3f} m)')
        ax.axhline(ate_c['ate_rmse'], color='g',      linestyle=':',
                   linewidth=1.5, alpha=0.5)
        ax.axhline(ate_k['ate_rmse'], color='purple',  linestyle=':',
                   linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Error (m)')
        ax.set_title('ATE por frame — después de Sim(3)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [3,1] Barras ATE + RPE
        ax       = axes[3, 1]
        labels   = [name_c.replace(' ', '\n'), name_k.replace(' ', '\n')]
        ate_vals = [ate_c['ate_rmse'], ate_k['ate_rmse']]
        rpe_vals = [rpe_c['rpe_rmse'], rpe_k['rpe_rmse']]
        x_pos    = np.arange(len(labels))
        width    = 0.35

        bars_ate = ax.bar(x_pos - width / 2, ate_vals, width,
                          label='ATE RMSE',
                          color=['green', 'purple'], alpha=0.7,
                          edgecolor='black', linewidth=1.5)
        bars_rpe = ax.bar(x_pos + width / 2, rpe_vals, width,
                          label='RPE RMSE (Δ=5)',
                          color=['lightgreen', 'plum'], alpha=0.7,
                          edgecolor='black', linewidth=1.5)

        for bar in list(bars_ate) + list(bars_rpe):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{bar.get_height():.3f} m',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel('Error (m)')
        ax.set_title('ATE y RPE por método', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # ── Guardado ──────────────────────────────────────────────────────────
        plt.tight_layout()

        if output_file is None:
            output_file = 'outputs/benchmarks/slam_analysis_complete.png'

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Figura guardada: {output_file}")
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
                'scale_factor'      : float(params['s']),
                'rotation_matrix'   : params['R'].tolist(),
                'translation_vector': params['t'].tolist(),
                'alignment_rmse'    : float(params['alignment_rmse']),
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
        traj = data['trajectory']
        n    = len(traj['x'])
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