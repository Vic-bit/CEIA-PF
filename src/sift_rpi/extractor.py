# extractor.py
import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from config import (
    SIFT_N_FEATURES, MIN_PIXEL_DISP, MIN_MATCHES
)

# Inicializar matriz identidad de 4x4
IRt = np.eye(4)


def add_ones(x):
    # A x le añade una columna de 1s para convertirlo en coordenadas homogéneas
    # representando cada punto como [x,y,1]
    return np.concatenate([x, np.ones((x.shape[0],1))], axis = 1)


def extractPose(F):
    '''
    Recibe la matriz fundamental y se extrae de ella la pose (rot. y trasl.)
    '''
    # Definir la matriz para computar la rotación
    W = np.asmatrix([[0,-1,0],[1,0,0],[0,0,1]])

    # Singular Value Decomposition (SVD) de la matriz fundamental
    U,d,Vt = np.linalg.svd(F)

    # Verificar que U tenga determinante positivo para asegurarse de que las
    # rotaciones son válidas y evitar reflexiones (si det(U)<0), en caso de 
    # ocurrir se interrumpe el programa y lanza un error.
    assert np.linalg.det(U) > 0

    # Corrige Vt si su determinante es negativo para asegurar una matriz de 
    # rotación apropiada 
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Calcular la matriz de rotación: R=U⋅W⋅Vt
    R = np.dot(np.dot(U,W),Vt)

    # Si la traza es negativa se usa W.T para que la rotación sea válida
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U,W.T),Vt)

    # Tomar la tercera columna de U como vector de traslación
    t = U[:,2]

    # Inicializar matriz ret como una matriz identidad de 4x4
    ret = np.eye(4)

    # Asignar a la matriz cuadrada de 3x3 la matriz R
    ret[:3,:3] = R

    # Asingar al vector de 3x1 el vector t
    ret[:3,3] = t

    print(d)

    return ret


def extract(img):
    '''
    Obtiene puntos de interes y sus descriptores de una imagen img
    '''
    # Crea un instancia de orb
    sift = cv2.SIFT_create(nfeatures=SIFT_N_FEATURES)

    # Convertir a escala de grises la imagen
    #gray_img = img 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detectar keypoints
    kps = sift.detect(gray_img, None)
    if not kps:
        return np.array([]), None
    
    # Extracción
    kps, des = sift.compute(gray_img, kps)

    # Extrae las coordenadas para visualización
    keypoints_array = np.array([kp.pt for kp in kps])
    return keypoints_array, des

    #return np.array([(kp.pt[0],kp.pt[1]) for kp in kps]), des


def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]


def denormalize(K, pt):
    ret = np.dot(K, [pt[0],pt[1],1.0])
    ret/= ret[2]
    return int(round(ret[0])), int(round(ret[1]))


class Matcher(object):
    def __init__(self):
        self.last = None


def match_frames(f1,f2):
    # Verificar si alguno de los frames no tiene descriptores
    if f1.des is None or f2.des is None:
        print("No se pueden encontrar los descritores de alguno de los frames")
        return np.array([]), np.array([]), np.eye(4)

    # Asegurarse que el tiepo de dato sea float 32
    f1.des = f1.des.astype(np.float32)
    f2.des = f2.des.astype(np.float32) 

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # Lowie's radio test
    ret = []
    idx1, idx2 = [], []
    for match in matches:
        if len(match)<2:
            continue
        m, n = match[0], match[1]
        if m.distance < 0.75 * n.distance:
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]

            if np.linalg.norm((p1-p2)) < 0.1:
                # Mantener idx
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                ret.append((p1,p2))
                pass

##--------- Nuevo para error de no movimiento ---------
    # Umbral de paralaje mínimo (en píxeles) para descartar frames sin movimiento
    idx1   = np.array(idx1, dtype=np.int32)
    idx2   = np.array(idx2, dtype=np.int32)

    # Extraemos las coordenadas en píxel de los keypoints
    pts1_pix = f1.kps[np.array(idx1)]
    pts2_pix = f2.kps[np.array(idx2)]
    # Calcular desplazamientos y su media
    disps = np.linalg.norm(pts1_pix - pts2_pix, axis=1)
    mean_disp = disps.mean() if disps.size>0 else 0.0
    if mean_disp < MIN_PIXEL_DISP:
        #print(f"Poco paralaje ({mean_disp:.2f}px), salto frame")
        return np.array([]), np.array([]), np.eye(4)
##--------- Nuevo para error de no movimiento ---------

    #assert len(ret) > 8
    if len(ret) <= MIN_MATCHES:
        #print("No se encontraron suficientes Matches, saltanto el frame")
        return np.array([]), np.array([]), np.eye(4)

    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # Matiz de fit
    model, inliers = ransac((ret[:,0], ret[:,1]), 
                            FundamentalMatrixTransform,
                            min_samples=8, 
                            residual_threshold = 0.005,
                            max_trials = 200)
    
    # Ignore outliers
    ret = ret[inliers]
    Rt = extractPose(model.params)

    return idx1[inliers], idx2[inliers], Rt
    

class Frame(object):
    def __init__(self, mapp, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.pose = IRt
        self.id = len(mapp.frames)
        mapp.frames.append(self)

        pts, self.des = extract(img)
        # Guarda los keypoints originales en píxeles para visualización
        self.kps = pts  
        
        # Si se extrajeron descriptores, normaliza los puntos para triangulación
        if self.des is not None and pts.size > 0:
            self.pts = normalize(self.Kinv, pts)
