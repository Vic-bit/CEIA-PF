"""
Alineación de Trayectorias SLAM vs Ground Truth
Algoritmos Sim(3) y escala no-uniforme para odometría monocular

Para SLAM monocular sin escala absoluta, se requiere alineación porque:
- SLAM monocular produce trayectorias en unidades arbitrarias
- Ground truth KITTI está en metros reales
- Usamos Sim(3) uniforme siguiendo protocolo oficial KITTI
- Transformación: P' = s * R @ P + t (un único factor s)
"""

import numpy as np


def align_sim3_umeyama(source: np.ndarray, target: np.ndarray) -> tuple:
    """
    Alineación Sim(3) usando método de Umeyama (1991)
    
    Transformación: target ≈ s * R @ source + t
    
    Args:
        source: trayectoria estimada (N, 3)
        target: ground truth (N, 3)
    
    Returns:
        R: rotación (3, 3)
        t: traslación (3,)
        s: escala (float)
        error: RMSE de alineación
    """
    # 1. Centrar trayectorias
    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    
    source_centered = source - mu_source
    target_centered = target - mu_target
    
    # 2. Matriz de covarianza
    H = source_centered.T @ target_centered  # (3, 3)
    
    # 3. SVD para calcular rotación
    U, D, Vt = np.linalg.svd(H)
    
    # Manejar reflexión
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    
    R = U @ S @ Vt
    
    # 4. ESCALA (FÓRMULA CORRECTA DE UMEYAMA)
    # Rotar source primero
    source_rotated = (R @ source_centered.T).T
    
    # Escala = ratio de normas
    numerator = np.sum(target_centered * source_rotated)
    denominator = np.sum(source_rotated ** 2)
    
    if denominator > 1e-10:
        s = numerator / denominator
    else:
        s = 1.0
    
    # Asegurar escala positiva
    s = abs(s)
    
    # 5. Traslación
    t = mu_target - s * R @ mu_source
    
    # 6. Verificar alineación
    source_aligned = s * (R @ source.T).T + t
    errors = np.linalg.norm(source_aligned - target, axis=1)
    error = np.sqrt(np.mean(errors ** 2))
    
    return R, t, s, error


def apply_sim3_transform(trajectory: np.ndarray, R: np.ndarray, 
                         t: np.ndarray, s: float) -> np.ndarray:
    """
    Aplica transformación Sim(3) a trayectoria: P' = s * R @ P + t
    
    Args:
        trajectory: array (N, 3)
        R: matriz de rotación (3, 3)
        t: vector de traslación (3,)
        s: factor de escala
    
    Returns:
        Trayectoria transformada
    """
    return s * (R @ trajectory.T).T + t

