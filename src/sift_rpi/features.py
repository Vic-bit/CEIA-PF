# features.py
import cv2
import numpy as np

from config import SIFT_N_FEATURES
ORB_N_FEATURES = SIFT_N_FEATURES  # Usar el mismo número de features para ORB en RPi

# Global cache para ORB detector y BFMatcher (evita re-instanciación en cada frame)
_orb_detector = None
_bf_matcher = None


def get_orb_detector():
    """Obtiene el detector ORB global (singleton pattern)."""
    global _orb_detector
    if _orb_detector is None:
        _orb_detector = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
    return _orb_detector


def get_bf_matcher():
    """Obtiene el matcher BF global (singleton pattern)."""
    global _bf_matcher
    if _bf_matcher is None:
        _bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return _bf_matcher

"""
def extract(img: np.ndarray):
    Obtiene puntos de interés y sus descriptores de una imagen img.
    En Raspberry Pi usa ORB por rendimiento.
    
    Args:
        img (np.ndarray): Imagen de entrada en BGR

    Returns:
        keypoints_array (np.ndarray): Array de coordenadas (x,y) de los keypoints
        des (np.ndarray): Descriptores asociados a los keypoints
    
    # Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Usar SIFT (mucho más robusto que ORB para SLAM)
    sift = cv2.ORB(nfeatures=SIFT_N_FEATURES)
    kps = sift.detect(gray_img, None)
    kps, des = sift.compute(gray_img, kps)

    # Si no detectó nada, devuelve un array vacío
    if not kps:
        return np.array([]), None
    
    # Extrae las coordenadas para visualización
    keypoints_array = np.array([kp.pt for kp in kps])

    return keypoints_array, des
"""

def extract(img: np.ndarray):
    """
    Obtiene puntos de interés y sus descriptores de una imagen img.
    En Raspberry Pi usa ORB por rendimiento.
    
    Args:
        img (np.ndarray): Imagen de entrada en escala de grises

    Returns:
        keypoints_array (np.ndarray): Array de coordenadas (x,y) de los keypoints
        des (np.ndarray): Descriptores asociados a los keypoints
    """
    # Imagen ya en escala de grises desde captura
    gray_img = img

    # Usar ORB detector cacheado (evita re-instanciación)
    orb = get_orb_detector()
    
    # Detectar keypoints
    kps = orb.detect(gray_img, None)
    
    # Si no detectó nada, devuelve arrays vacíos
    if not kps:
        return np.array([]), None
    
    # Calcular descriptores
    kps, des = orb.compute(gray_img, kps)
    
    # Si compute falló, devuelve arrays vacíos
    if des is None:
        return np.array([]), None
    
    # Extrae las coordenadas para visualización
    keypoints_array = np.array([kp.pt for kp in kps])

    return keypoints_array, des

def add_ones(x: np.ndarray) -> np.ndarray:
    """
    A x le añade una columna de 1s para convertirlo en coordenadas homogéneas
    representando cada punto como [x,y,1]

    Args:
        x (np.ndarray): Array de puntos Nx2

    Returns:
        np.ndarray: Array de puntos Nx3 con coordenadas homogéneas
    """
    return np.concatenate([x, np.ones((x.shape[0],1))], axis = 1)


def normalize(Kinv, pts):
    """
    Normaliza puntos píxel pts (Nx2) a coordenadas de cámara usando Kinv
    
    Args:
        Kinv (np.ndarray): Inversa de la matriz intrínseca K (3x3)
        pts (np.ndarray): Puntos en píxeles (Nx2)   
    Returns:
        np.ndarray: Puntos normalizados (Nx2)
    """
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]


def denormalize(K, pt):
    """
    Convierte un punto pt (x,y) en coordenadas de cámara a píxeles usando K
    
    Args:
        K (np.ndarray): Matriz intrínseca (3x3)
        pt (tuple): Punto en coordenadas de cámara (x,y)
    
    Returns:
        tuple: Punto en píxeles (x_pix, y_pix)
    """
    ret = np.dot(K, [pt[0],pt[1],1.0])
    ret/= ret[2]
    return int(round(ret[0])), int(round(ret[1]))


def match_frames(f1, f2,
                 ratio_thresh=0.75,      # CAMBIAR de 0.80 a 0.75
                 min_good_matches=6,     # CAMBIAR de 8 a 6
                 ransac_thresh=0.8,      # CAMBIAR de 1.0 a 0.8
                 min_inliers=6):  
    """
    Empareja f1.des con f2.des usando ORB (descriptores binarios).
    Aplica ratio test, estima F con findFundamentalMat (píxeles),
    convierte a E = K^T F K y usa recoverPose para obtener R,t.
    
    Args:
        f1, f2 (Frame): Frames a emparejar (deben tener f1.K, f2.K)
        ratio_thresh (float): Umbral para ratio test (0.80 para ORB)
        min_good_matches (int): Mínimo de buenos matches tras ratio test
        ransac_thresh (float): Umbral RANSAC para findFundamentalMat (píxeles)
        min_inliers (int): Mínimo de inliers para considerar válido
        
    Returns:
        idx1_inliers (np.ndarray): Índices de inliers en f1.kps
        idx2_inliers (np.ndarray): Índices de inliers en f2.kps
        Rt4x4 (np.ndarray): Matriz de transformación 4x4 de f2 a f1
    """
    # Verificar que existan descriptores
    if f1.des is None or f2.des is None:
        return np.array([], dtype=int), np.array([], dtype=int), np.eye(4)
    
    # BFMatcher cacheado con HAMMING para ORB (descriptores binarios)
    bf = get_bf_matcher()
    
    # KNN matching (k=2 para ratio test)
    knn = bf.knnMatch(f1.des, f2.des, k=2)
    
    # Ratio test de Lowe
    good_qidx, good_tidx = [], []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_qidx.append(m.queryIdx)
            good_tidx.append(m.trainIdx)
    
    # Verificar mínimo de matches
    if len(good_qidx) < min_good_matches:
        return np.array([], dtype=int), np.array([], dtype=int), np.eye(4)
    
    # Construir arrays de puntos en píxeles para findFundamentalMat
    pts1_pix = np.float32([f1.kps[i] for i in good_qidx])
    pts2_pix = np.float32([f2.kps[j] for j in good_tidx])
    
    # Estimar matriz fundamental F con RANSAC
    F, mask = cv2.findFundamentalMat(
        pts1_pix, pts2_pix, 
        cv2.FM_RANSAC,
        ransac_thresh, 
        0.99,    # Confianza
        2000     # Máximo de iteraciones
    )
    
    # Verificar que F sea válida
    if F is None or mask is None:
        return np.array([], dtype=int), np.array([], dtype=int), np.eye(4)
    
    mask = mask.ravel().astype(bool)
    
    # Verificar mínimo de inliers
    if mask.sum() < min_inliers:
        return np.array([], dtype=int), np.array([], dtype=int), np.eye(4)
    
    # Índices de inliers relativos a f1/f2
    inlier_idx1 = np.array(good_qidx)[mask]
    inlier_idx2 = np.array(good_tidx)[mask]
    
    # ===== AQUÍ ESTÁ K: viene de f1.K =====
    # K se asignó cuando creaste Frame(mapp, img, K)
    K = f1.K
    
    # Convertir F → E (matriz esencial)
    E = K.T @ F @ K
    
    # Preparar puntos inliers en píxel para recoverPose
    pts1_in = pts1_pix[mask]
    pts2_in = pts2_pix[mask]
    
    # recoverPose: extrae R, t de E usando test de cheirality
    retval, R, t, pose_mask = cv2.recoverPose(E, pts1_in, pts2_in, K)
    
    # Verificar que recoverPose haya recuperado suficientes puntos
    if retval < min_inliers:
        return np.array([], dtype=int), np.array([], dtype=int), np.eye(4)
    
    # Construir matriz de transformación 4x4
    Rt = np.eye(4, dtype=float)
    Rt[:3, :3] = R
    Rt[:3, 3] = t.ravel()
    
    return inlier_idx1, inlier_idx2, Rt

class Frame(object):
    """Clase Frame que contiene keypoints, descriptores y pose
    
    Args:
        map (Map): Mapa al que pertenece el frame
        img (np.ndarray): Imagen del frame en escala de grises
        K (np.ndarray): Matriz intrínseca de la cámara (3x3)
    """
    def __init__(self, map, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

        # Inicializar matriz identidad de 4x4
        IRt = np.eye(4)
        self.pose = IRt
        self.id = len(map.frames)
        map.frames.append(self)

        kps, des = extract(img)
        # Guarda los keypoints originales en píxeles para visualización
        self.des = des
        self.kps = kps  # Coordenadas en píxeles (N, 2)
        
        # Si se extrajeron descriptores, normaliza los puntos para triangulación
        if self.des is not None and kps.size > 0:
            self.pts = normalize(self.Kinv, self.kps)
        # TODO Ver si esto es una mejora o un problema
        #else:
            # Si no hay keypoints, asignar array vacío
        #    self.pts = np.array([])
