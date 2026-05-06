import json
import psutil
import time
from pathlib import Path
from datetime import datetime


class BenchmarkLogger:
    """
    Logger para capturar métricas de performance en Visual SLAM.
    
    Métricas capturadas por frame:
    - frame_id: ID del frame
    - time_ms: Tiempo de procesamiento en milisegundos
    - num_matches: Número de correspondencias encontradas
    
    Resumen al final:
    - total_time: Tiempo total de ejecución (segundos)
    - avg_fps: FPS promedio durante toda la secuencia
    - avg_matches: Promedio de matches por frame
    - total_ram_mb: Consumo máximo de memoria en MB
    """
    
    def __init__(self, implementation_name: str):
        """
        Args:
            implementation_name: 'sift_classic' o 'sift_kornia'
        """
        self.implementation_name = implementation_name
        self.frames_data = []  # Lista de dict con métricas por frame
        self.start_time = time.perf_counter()
        self.process = psutil.Process()
        self.max_ram_mb = 0.0
        
    def log_frame(self, frame_id: int, num_matches: int, elapsed_ms: float):
        """
        Registra métrica de un frame.
        
        Args:
            frame_id: ID del frame procesado
            num_matches: Número de correspondencias encontradas
            elapsed_ms: Tiempo de procesamiento del frame en milisegundos
        """
        # Actualizar máximo de memoria
        current_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        self.max_ram_mb = max(self.max_ram_mb, current_ram_mb)
        
        # Registrar datos del frame
        self.frames_data.append({
            "frame_id": frame_id,
            "time_ms": round(elapsed_ms, 2),
            "num_matches": num_matches
        })
    
    def export_summary(self, output_path: str) -> dict:
        """
        Calcula y exporta el resumen de benchmarking en JSON.
        
        Args:
            output_path: Ruta donde guardar el archivo JSON
            
        Returns:
            dict: El resumen exportado
        """
        if not self.frames_data:
            print("[Warning] No hay datos de frames registrados")
            return {}
        
        total_time = time.perf_counter() - self.start_time
        num_frames = len(self.frames_data)
        avg_fps = num_frames / total_time if total_time > 0 else 0
        
        # Calcular promedio de matches
        matches_list = [f["num_matches"] for f in self.frames_data]
        avg_matches = sum(matches_list) / len(matches_list) if matches_list else 0
        
        # Construir resumen
        summary = {
            "metadata": {
                "implementation": self.implementation_name,
                "timestamp": datetime.now().isoformat(),
                "total_frames": num_frames
            },
            "summary": {
                "total_time": round(total_time, 2),
                "avg_fps": round(avg_fps, 2),
                "avg_matches": round(avg_matches, 1),
                "total_ram_mb": round(self.max_ram_mb, 1)
            }
        }
        
        # Crear directorio si no existe
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar JSON
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[Logger] Benchmark guardado en: {output_path}")
        
        return summary
