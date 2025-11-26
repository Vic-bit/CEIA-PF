# pointmap.py
import torch

from features import Frame

class Map(object):
    """Clase que representa el mapa 3D construido por el SLAM"""
    def __init__(self, device: torch.device):
        self.frames = []  # Frames de la cámara
        self.points = []  # Puntos 3D del mapa
        self.device = device

    def add_or_update_point(self, pt: torch.Tensor, frame: Frame, idx: int, threshold=5.0):
        """
        Si existe un punto en el mapa cercano a 'pt' (dentro de 'threshold'),
        se añade la observación; si no, se crea un nuevo punto.

        Args:
            pt (torch.Tensor): Coordenadas 3D del punto a añadir/actualizar
            frame (Frame): Frame donde se observa el punto
            idx (int): Índice del keypoint en el frame
            threshold (float): Distancia máxima para considerar el punto como existente

        Returns:
            Point: El punto 3D añadido o actualizado
        """
        # No hay puntos existentes, crear nuevo directamente
        if len(self.points) == 0:
            new_point = Point(self, pt)
            new_point.add_observation(frame, idx)
            return new_point
        
        # Intentar operación batch en tensores
        pt_xyz = pt[:3]
        
        all_positions = torch.stack([p.pt for p in self.points])
        
        distances = torch.norm(all_positions - pt_xyz.unsqueeze(0), dim=1)
            
        # Encontrar punto más cercano
        min_dist, min_idx = torch.min(distances, dim=0)
            
        if min_dist.item() < threshold:
            return self.points[min_idx.item()]
    
        # Si no se encuentra, crear nuevo punto
        new_point = Point(self, pt)
        new_point.add_observation(frame, idx)
        return new_point
    
            
class Point(object):
    def __init__(self, map: Map, loc: torch.Tensor):
        self.frames = []
        self.idxs = []
        self.pt = loc[:3].clone()
        self.id = len(map.points)
        map.points.append(self)

    def add_observation(self, frame: Frame, idx: int):
        """Añade una observación del punto en un frame específico"""
        # Evitar observaciones duplicadas del mismo frame
        if frame not in self.frames:
            self.frames.append(frame)
            self.idxs.append(idx)

    def is_good(self) -> bool:
        """
        Determina si el punto es bueno basado en número de observaciones y consistencia
        """
        # Al menos 2 observaciones de frames diferentes
        if len(self.frames) < 2:
            return False
        
        # Verificar que las observaciones son de frames suficientemente separados
        frame_ids = [f.id for f in self.frames]
        unique_frames = len(set(frame_ids))
        if unique_frames < 2:
            return False
        
        # Profundidad razonable
        z = self.pt[2].item()
        if abs(z) < 0.1 or abs(z) > 1000:
            return False
        
        return True
