# pointmap.py
import numpy as np

class Map(object):
    def __init__(self):
        self.frames = []  # Frames de la cámara
        self.points = []  # Puntos 3D del mapa

    def add_or_update_point(self, pt, frame, idx, threshold=0.1):
        """
        Si existe un punto en el mapa cercano a 'pt' (dentro de 'threshold'),
        se añade la observación; si no, se crea un nuevo punto.
        """
        pt_np = pt[:3]  # Solo X, Y, Z
        for p in self.points:
            p_np = p.pt[:3]
            if np.linalg.norm(pt_np - p_np) < threshold:
                p.add_observation(frame, idx)
                return p
        # Si no se encuentra, se crea un nuevo punto
        new_point = Point(self, pt)
        new_point.add_observation(frame, idx)
        return new_point

# Clase que representa un punto 3D
class Point(object):
    def __init__(self, mapp, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []
        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)
