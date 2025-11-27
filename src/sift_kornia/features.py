# features.py
import torch
import cv2
import numpy as np
import kornia as K
import kornia.feature as KF
import torch.nn.functional as F
from typing import Tuple, Optional

from kornia_moons.feature import OpenCVFeatureKornia
from config import SIFT_N_FEATURES

_lg_matcher = None  # Variable global para LightGlue
_opencv_sift = None  # Variable global para OpenCVFeatureKornia


def get_lightglue(device):
    global _lg_matcher
    if _lg_matcher is None:
        print("Inicializando LightGlue (una sola vez)...")
        _lg_matcher = KF.LightGlueMatcher("sift").eval().to(device)
    return _lg_matcher

def get_opencv_sift(nfeatures=SIFT_N_FEATURES):
    global _opencv_sift
    if _opencv_sift is None:
        print("Inicializando OpenCVFeatureKornia (SIFT) una sola vez...")
        _opencv_sift = OpenCVFeatureKornia(cv2.SIFT_create(nfeatures), mrSize=1.0)
    return _opencv_sift

class Frame(object):
    """Frame con keypoints, descriptores y pose en tensores
    
    Args:
        map (Map): Mapa al que pertenece
        img_tensor (torch.Tensor): Imagen en formato (1, C, H, W) o (C, H, W)
        K (torch.Tensor): Matriz intrínseca (3, 3)
        device (torch.device): Device para cálculos
    """
    def __init__(self, map, img_tensor: torch.Tensor, K: torch.Tensor, device: torch.device):
        self.K = K.to(device)
        self.Kinv = torch.inverse(self.K)
        self.device = device
        
        # Guardar dimensiones de la imagen para LightGlue
        if img_tensor.dim() == 4:
            _, _, H, W = img_tensor.shape
        elif img_tensor.dim() == 3:
            C, H, W = img_tensor.shape
        else:
            raise ValueError(f"img_tensor debe tener 3 o 4 dimensiones, tiene {img_tensor.dim()}")
        
        self.img_shape = (H, W)
        
        # Pose inicial (identidad 4x4)
        self.pose = torch.eye(4, device=device, dtype=torch.float32)
        self.id = len(map.frames)
        map.frames.append(self)
        
        # Extraer features
        self.lafs, self.des, self.kps = extract(img_tensor, device)
        
        # Puntos normalizados para triangulación
        if self.des is not None and self.kps.shape[0] > 0:
            self.pts = normalize(self.Kinv, self.kps)
        else:
            self.pts = torch.empty((0, 2), device=device)


def extract(img_tensor: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extrae features SIFT usando OpenCVFeatureKornia (wrapper de cv2.SIFT)
    
    Args:
        img_tensor (torch.Tensor): Imagen en formato (1, C, H, W) o (C, H, W)
        device (torch.device): Device donde realizar los cálculos

    Returns:
        lafs (torch.Tensor): Local Affine Frames (1, N, 2, 3)
        descriptors (torch.Tensor): Descriptores RootSIFT normalizados (1, N, 128)
        keypoints (torch.Tensor): Coordenadas (x,y) de keypoints (N, 2)
    """
    # Asegurar formato correcto
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)  # (C, H, W)
    
    img_tensor = img_tensor.to(device).contiguous()
    
    # SIFT con OpenCV en Kornia
    opencv_sift = get_opencv_sift(SIFT_N_FEATURES)
    
    # Extraer features
    lafs, resps, des = opencv_sift(img_tensor)
    
    # Normalizar descriptores
    des = F.normalize(des, dim=-1, p=1).sqrt()
    
    # Extraer centros de LAFs
    kps = KF.get_laf_center(lafs)[0]  # (N, 2)
    
    return lafs, des, kps


def add_ones(x: torch.Tensor) -> torch.Tensor:
    """
    Añade columna de 1s para coordenadas homogéneas [x, y, 1]
    
    Args:
        x (torch.Tensor): Puntos (N, 2)
    
    Returns:
        torch.Tensor: Puntos homogéneos (N, 3)
    """
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    return torch.cat([x, ones], dim=1)


def normalize(Kinv: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """
    Normaliza puntos píxel a coordenadas de cámara usando Kinv
    
    Args:
        Kinv (torch.Tensor): Inversa de matriz intrínseca (3, 3)
        pts (torch.Tensor): Puntos en píxeles (N, 2)
    
    Returns:
        torch.Tensor: Puntos normalizados (N, 2)
    """
    pts_homo = add_ones(pts)  # (N, 3)
    pts_normalized = torch.mm(pts_homo, Kinv.T)  # (N, 3)
    return pts_normalized[:, :2]

def match_frames(f1: Frame,
                 f2: Frame,
                 min_good_matches: int = 8,
                 ransac_thresh: float = 1.0,
                 min_inliers: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Empareja features entre f1 y f2 usando LightGlue, estima F con RANSAC
    y recupera pose con cv2.recoverPose
    
    Args:
        f1, f2 (Frame): Frames a emparejar
        ratio_thresh (float): No usado con LightGlue (mantener por compatibilidad)
        min_good_matches (int): Mínimo de matches tras LightGlue
        ransac_thresh (float): Umbral RANSAC para findFundamentalMat
        min_inliers (int): Mínimo de inliers válidos tras recoverPose
    
    Returns:
        idx1_inliers (torch.Tensor): Índices de inliers en f1 (M,)
        idx2_inliers (torch.Tensor): Índices de inliers en f2 (M,)
        Rt (torch.Tensor): Matriz de transformación 4x4
    """
    device = f1.kps.device
    identity = torch.eye(4, device=device, dtype=torch.float32)
    empty_idx = torch.empty(0, dtype=torch.long, device=device)
    
    # Verificar descriptores
    if f1.des is None or f2.des is None or f1.des.shape[1] == 0 or f2.des.shape[1] == 0:
        return empty_idx, empty_idx, identity
    
    # Matching con LightGlue
    H, W = f1.img_shape  # Altura y Ancho de la imagen
    hw = torch.as_tensor([H, W], device=device)
    
    lg_matcher = get_lightglue(device)
    
    try:
        dists, idxs = lg_matcher(
            f1.des[0], f2.des[0],
            f1.lafs, f2.lafs,
            hw1=hw, hw2=hw
        )
    except Exception as e:
        print(f"[WARN] LightGlue matching falló: {e}")
        return empty_idx, empty_idx, identity
    
    if idxs.shape[0] < min_good_matches:
        return empty_idx, empty_idx, identity
    
    # Puntos matched
    mkpts1 = f1.kps[idxs[:, 0]]  # (M, 2)
    mkpts2 = f2.kps[idxs[:, 1]]  # (M, 2)
    
    # Convertir a numpy para OpenCV
    mkpts1_np = mkpts1.detach().cpu().numpy()
    mkpts2_np = mkpts2.detach().cpu().numpy()
    K_np = f1.K.cpu().numpy()
    
    # Estimar matriz fundamental con RANSAC
    Fm, inliers_mask = cv2.findFundamentalMat(
        mkpts1_np, mkpts2_np,
        cv2.USAC_MAGSAC,
        ransac_thresh,
        0.999,
        100000
    )
    
    if Fm is None or inliers_mask is None:
        return empty_idx, empty_idx, identity
    
    inliers_mask = inliers_mask.ravel().astype(bool)
    
    if inliers_mask.sum() < min_inliers:
        return empty_idx, empty_idx, identity
    
    # Convertir F -> E
    E = K_np.T @ Fm @ K_np
    
    # Recuperar pose con recoverPose
    retval, R, t, pose_mask = cv2.recoverPose(
        E,
        mkpts1_np[inliers_mask],
        mkpts2_np[inliers_mask],
        K_np
    )
    
    pose_mask = pose_mask.ravel().astype(bool)
    
    if pose_mask.sum() < min_inliers:
        return empty_idx, empty_idx, identity
    
    # Combinar máscaras de inliers
    final_inliers = np.zeros(len(inliers_mask), dtype=bool)
    final_inliers[inliers_mask] = pose_mask
    
    # Índices finales de inliers
    idx1_inliers = idxs[final_inliers, 0]
    idx2_inliers = idxs[final_inliers, 1]
    
    # Construir Rt 4x4
    Rt = torch.eye(4, device=device, dtype=torch.float32)
    Rt[:3, :3] = torch.from_numpy(R).float().to(device)
    Rt[:3, 3] = torch.from_numpy(t.ravel()).float().to(device)
    
    return idx1_inliers, idx2_inliers, Rt
