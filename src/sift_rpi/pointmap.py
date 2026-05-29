# pointmap.py
import numpy as np

from features import Frame
from config import POINT_MERGE_THRESHOLD, POINT_Z_MIN, POINT_Z_MAX

class Map(object):
    """Clase que representa el mapa 3D construido por el SLAM"""
    def __init__(self):
        self.frames = []  # Frames de la cámara
        self.points = []  # Puntos 3D del mapa
    
    def add_or_update_point(self, pt: np.array, frame: Frame, idx: int, threshold=None) -> np.array:
        """
        Si existe un punto en el mapa cercano a 'pt' (dentro de 'threshold'),
        se añade la observación; si no, se crea un nuevo punto.

        Args:
            pt (np.ndarray): Coordenadas 3D del punto a añadir/actualizar
            frame (Frame): Frame donde se observa el punto
            idx (int): Índice del keypoint en el frame
            threshold (float): Distancia máxima para considerar el punto como existente
                             Si es None, usa POINT_MERGE_THRESHOLD del config

        Returns:
            Point: El punto 3D añadido o actualizado
        """
        if threshold is None:
            threshold = POINT_MERGE_THRESHOLD
        pt_xyz = pt[:3]  # Solo X, Y, Z
        
        # Buscar punto existente cercano
        for p in self.points:
            p_xyz = p.pt[:3]
            if np.linalg.norm(pt_xyz - p_xyz) < threshold:
                p.add_observation(frame, idx)  # ✓ Agregar observación al punto existente
                return p
        
        # Aplicar límite de puntos en el mapa (para no saturar RPi)
        from config import MAX_POINTS_IN_MAP
        if len(self.points) >= MAX_POINTS_IN_MAP:
            # Eliminar el punto con menos observaciones (menos importante)
            self.points.sort(key=lambda p: len(p.frames))
            removed_point = self.points.pop(0)
            print(f"🗑️  Evicted point with {len(removed_point.frames)} observations (map full at {MAX_POINTS_IN_MAP})")
        
        # Si no se encuentra, se crea un nuevo punto
        new_point = Point(self, pt)
        new_point.add_observation(frame, idx)
        return new_point


class Point(object):
    """Clase que representa un punto 3D en el mapa"""
    def __init__(self, map, loc):
        self.frames = []
        self.pt = loc[:3] if len(loc) > 3 else loc  # Solo guardar X,Y,Z
        self.idxs = []
        self.id = len(map.points)
        map.points.append(self)

    def add_observation(self, frame, idx):
        """Añade una observación del punto en un frame específico, 
        evitando duplicados.
        
        Args:
            frame (Frame): Frame donde se observa el punto
            idx (int): Índice del keypoint en el frame
        """
        if frame not in self.frames:
            self.frames.append(frame)
            self.idxs.append(idx)
    
    def update_position(self, new_pt):
        """Actualiza la posición del punto (promedio de observaciones)
        
        Args:
            new_pt (np.ndarray): Nueva posición 3D propuesta
        """
        if len(self.frames) > 1:
            # Por ahora solo actualizamos si es significativamente diferente
            if np.linalg.norm(self.pt - new_pt[:3]) > 0.1:
                self.pt = (self.pt + new_pt[:3]) / 2.0
    
    def is_good(self):
        """Determina si el punto es bueno basado en número de observaciones y consistencia"""
        # Al menos 2 observaciones de frames diferentes
        if len(self.frames) < 2:
            return False
        
        # Verificar que las observaciones son de frames suficientemente separados
        frame_ids = [f.id for f in self.frames]
        if len(set(frame_ids)) < 2:  # Observaciones del mismo frame
            return False
        
        # El punto debe tener una profundidad razonable (usando config)
        if abs(self.pt[2]) < POINT_Z_MIN or abs(self.pt[2]) > POINT_Z_MAX:
            return False
            
        return True
