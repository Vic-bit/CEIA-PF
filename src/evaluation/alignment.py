"""
Alineación de Trayectorias SLAM vs Ground Truth
Implementación Sim(3) — Umeyama (1991)

Referencia:
    S. Umeyama, "Least-squares estimation of transformation parameters
    between two point patterns", IEEE TPAMI, 13(4):376-380, 1991.
    https://ieeexplore.ieee.org/document/70844

Contexto de uso:
    SLAM monocular produce trayectorias en unidades arbitrarias (sin escala
    absoluta). Ground truth KITTI está en metros. La transformación Sim(3)
    recupera la escala, rotación y traslación óptimas en sentido de mínimos
    cuadrados.

Transformación buscada (ec. 1 del paper):
    target ≈ s · R · source + t

    donde:
        s  ∈ ℝ⁺          — escala uniforme (un único escalar)
        R  ∈ SO(3)        — rotación propia (det R = +1)
        t  ∈ ℝ³           — traslación
"""

import numpy as np


def align_sim3_umeyama(source: np.ndarray, target: np.ndarray) -> tuple:
    """
    Alineación Sim(3) óptima en mínimos cuadrados según Umeyama (1991).

    Encuentra (s, R, t) que minimizan:

        (1/n) · Σᵢ ‖ target_i − (s · R · source_i + t) ‖²        (ec. 4)

    Derivación paso a paso siguiendo las ecuaciones del paper:

        μ_src  = (1/n) Σ source_i                                  (ec. 34)
        μ_tgt  = (1/n) Σ target_i                                  (ec. 35)
        σ²_src = (1/n) Σ ‖ source_i − μ_src ‖²                    (ec. 36)
        Σ_st   = (1/n) Σ (target_i − μ_tgt)(source_i − μ_src)ᵀ   (ec. 38)
        U D Vᵀ = SVD(Σ_st)                                         (ec. 39-40)
        S      = diag(1,...,1, det(U)·det(V))                      (ec. 43)
        R      = U · S · Vᵀ                                        (ec. 40)
        s      = (1/σ²_src) · tr(D · S)                            (ec. 42)
        t      = μ_tgt − s · R · μ_src                             (ec. 41)

    Args:
        source: trayectoria estimada por SLAM,  shape (N, 3)
        target: ground truth KITTI,             shape (N, 3)

    Returns:
        R     : matriz de rotación,   shape (3, 3),  det(R) = +1
        t     : vector de traslación, shape (3,)
        s     : escala uniforme,      float > 0
        error : RMSE de alineación,   float  [metros]
    """
    n = len(source)

    # ── Medias (ec. 34-35) ──────────────────────────────────────────────────
    mu_src = source.mean(axis=0)   # (3,)
    mu_tgt = target.mean(axis=0)   # (3,)

    src_c = source - mu_src        # centradas, shape (N, 3)
    tgt_c = target - mu_tgt

    # ── Varianza de source (ec. 36) ─────────────────────────────────────────
    # σ²_src = (1/n) Σ ‖source_i − μ_src‖²
    sigma2_src = np.sum(src_c ** 2) / n

    # ── Covarianza cruzada Σ_st (ec. 38) ────────────────────────────────────
    # Σ_st = (1/n) · tgt_cᵀ · src_c    →   shape (3, 3)
    # Nota: en el paper Σ_st[i,j] = cov(target_i, source_j), por eso
    #       el orden es tgtᵀ · src (target en filas, source en columnas).
    Sigma_st = (tgt_c.T @ src_c) / n   # (3, 3)

    # ── SVD de Σ_st (ec. 39) ────────────────────────────────────────────────
    # Σ_st = U · D · Vᵀ,  D = diag(d₁ ≥ d₂ ≥ d₃ ≥ 0)
    U, d, Vt = np.linalg.svd(Sigma_st)

    # ── Corrección de reflexión — S (ec. 43) ────────────────────────────────
    # Garantiza det(R) = +1 (rotación propia, sin reflexión).
    # Si det(U)·det(V) = −1, el SVD produjo una reflexión: se niega
    # la última componente de S para corregirlo.
    sign = np.linalg.det(U) * np.linalg.det(Vt.T)   # det(V) = det(Vᵀ)ᵀ
    S = np.diag([1.0, 1.0, sign])                    # diag(1, 1, ±1)

    # ── Rotación R (ec. 40) ─────────────────────────────────────────────────
    # R = U · S · Vᵀ
    R = U @ S @ Vt

    # ── Escala s (ec. 42) ───────────────────────────────────────────────────
    # s = (1/σ²_src) · tr(D · S)
    # tr(D · S) = Σ dₖ · Sₖₖ  =  d₁ + d₂ + sign·d₃
    s = np.trace(np.diag(d) @ S) / sigma2_src

    # Salvaguarda numérica: si σ²_src es degenerado (trayectoria puntual)
    if sigma2_src < 1e-10:
        s = 1.0

    # ── Traslación t (ec. 41) ───────────────────────────────────────────────
    # t = μ_tgt − s · R · μ_src
    t = mu_tgt - s * (R @ mu_src)

    # ── RMSE de verificación ────────────────────────────────────────────────
    src_aligned = s * (R @ source.T).T + t   # (N, 3)
    residuals = np.linalg.norm(src_aligned - target, axis=1)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return R, t, s, rmse


def apply_sim3_transform(trajectory: np.ndarray,
                         R: np.ndarray,
                         t: np.ndarray,
                         s: float) -> np.ndarray:
    """
    Aplica la transformación Sim(3) a una trayectoria.

    Implementa directamente ec. 1 del paper:
        P'ᵢ = s · R · Pᵢ + t

    Args:
        trajectory : puntos a transformar,  shape (N, 3)
        R          : matriz de rotación,    shape (3, 3)
        t          : traslación,            shape (3,)
        s          : escala uniforme,       float

    Returns:
        Trayectoria transformada, shape (N, 3)
    """
    # s · R · trajectoryᵀ  →  (3, N), luego .T → (N, 3)
    return s * (R @ trajectory.T).T + t