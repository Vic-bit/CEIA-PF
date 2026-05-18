# features.py
import cv2
import numpy as np

from config import SIFT_N_FEATURES


def extract(img: np.ndarray):
    """
    Obtiene puntos de interes y sus descriptores de una imagen img
    
    Args:
        img (np.ndarray): Imagen de entrada en BGR

    Returns:
        keypoints_array (np.ndarray): Array de coordenadas (x,y) de los key
        des (np.ndarray): Descriptores asociados a los keypoints
    """
    # Crea un instancia de sift
    sift = cv2.SIFT_create(nfeatures=SIFT_N_FEATURES)

    # Convertir a escala de grises la imagen
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extracción de puntos
    kps = sift.detect(gray_img, None)

    # Si no detectó nada, devuelve un array
    if not kps:
        return np.array([]), None
    
    # Extracción
    kps, des = sift.compute(gray_img, kps)

    # Extrae las coordenadas para visualización
    keypoints_array = np.array([kp.pt for kp in kps])

    return keypoints_array, des

    #return np.array([(kp.pt[0],kp.pt[1]) for kp in kps]), des

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
                 ratio_thresh=0.75,
                 min_good_matches=8,
                 ransac_thresh=1.0,
                 min_inliers=8):
    """
    Empareja f1.des con f2.des, aplica ratio test, estima F con findFundamentalMat (píxeles),
    convierte a E = K^T F K y usa recoverPose para obtener R,t.
    Devuelve (idx1_inliers, idx2_inliers, Rt4x4).
    Si falla algo, devuelve arrays vacíos y Rt = I.

    Args:
        f1, f2 (Frame): Frames a emparejar
        ratio_thresh (float): Umbral para ratio test
        min_good_matches (int): Mínimo de buenos matches tras ratio test
        ransac_thresh (float): Umbral RANSAC para findFundamentalMat
        min_inliers (int): Mínimo de inliers para considerar válido el emparejamiento
    Returns:
        idx1_inliers (np.ndarray): Índices de inliers en f1.kps
        idx2_inliers (np.ndarray): Índices de inliers en f2.kps
        Rt4x4 (np.ndarray): Matriz de transformación 4x4 de f1 a f2
    """

    # chequeos
    if f1.des is None or f2.des is None:
        return np.array([], dtype=int), np.array([], dtype=int), np.eye(4)

    # BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(f1.des, f2.des, k=2)

    good_qidx, good_tidx = [], []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_qidx.append(m.queryIdx)
            good_tidx.append(m.trainIdx)

    if len(good_qidx) < min_good_matches:
        # insuficientes matches
        return np.array([], dtype=int), np.array([], dtype=int), np.eye(4)

    # Construir arrays de puntos en píxeles para findFundamentalMat
    pts1_pix = np.float32([f1.kps[i] for i in good_qidx])
    pts2_pix = np.float32([f2.kps[j] for j in good_tidx])

    # Estimar F con RANSAC (función OpenCV)
    F, mask = cv2.findFundamentalMat(pts1_pix, pts2_pix, cv2.FM_RANSAC,
                                     ransac_thresh, 0.99, 2000)
    if F is None or mask is None:
        return np.array([], dtype=int), np.array([], dtype=int), np.eye(4)

    mask = mask.ravel().astype(bool)
    if mask.sum() < min_inliers:
        return np.array([], dtype=int), np.array([], dtype=int), np.eye(4)

    # índices de inliers relativos a f1/f2 (no a good_idx)
    inlier_idx1 = np.array(good_qidx)[mask]
    inlier_idx2 = np.array(good_tidx)[mask]

    # convertir F -> E con la K del frame (asumimos misma K)
    K = f1.K
    E = K.T @ F @ K

    # preparar puntos inliers en píxel para recoverPose
    pts1_in = pts1_pix[mask]
    pts2_in = pts2_pix[mask]

    # recoverPose devuelve R,t y un nuevo mask de puntos con cheirality
    retval, R, t, pose_mask = cv2.recoverPose(E, pts1_in, pts2_in, K)

    # Si recoverPose recuperó pocos puntos, considerar fallback
    if retval < min_inliers:
        # fallback: no actualizar pose
        return np.array([], dtype=int), np.array([], dtype=int), np.eye(4)

    # construir Rt 4x4
    Rt = np.eye(4, dtype=float)
    Rt[:3,:3] = R
    Rt[:3,3]  = t.ravel()

    return inlier_idx1, inlier_idx2, Rt 


class Frame(object):
    """Clase Frame que contiene keypoints, descriptores y pose
    
    Args:
        map (Map): Mapa al que pertenece el frame
        img (np.ndarray): Imagen del frame en BGR
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
        self.des = des  # Forma: (1, N, D)
        self.kps = kps  # Coordenadas en píxeles (N, 2)
        
        # Si se extrajeron descriptores, normaliza los puntos para triangulación
        if self.des is not None and kps.size > 0:
            self.pts = normalize(self.Kinv, self.kps)
