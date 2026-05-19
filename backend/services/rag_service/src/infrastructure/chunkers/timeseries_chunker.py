from typing import Any, Dict, List

import numpy as np

from backend.services.rag_service.src.infrastructure.chunkers.base_chunker import BaseChunker, Chunk


class TimeseriesChunker(BaseChunker):
    """
    Non-text chunker designed for raw telemetry float arrays.
    
    Implements overlapping rolling windows (e.g., 60s window, 30s step).
    Extracts statistical features useful for RAG or anomaly detection downstream:
    mean, std, min, max, trend_slope, and zero_crossing_rate.
    """

    def __init__(self, window_size: int = 60, step: int = 30) -> None:
        super().__init__()
        self.window_size = window_size
        self.step = step

    def _extract_features(self, window_data: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from a 1D float array."""
        if len(window_data) == 0:
            return {}
            
        mean_val = float(np.mean(window_data))
        
        # Zero crossing rate (how often it crosses the mean)
        centered = window_data - mean_val
        zero_crossings = np.where(np.diff(np.signbit(centered)))[0]
        zcr = float(len(zero_crossings) / len(window_data))
        
        # Trend slope using simple linear regression
        x = np.arange(len(window_data))
        slope, _ = np.polyfit(x, window_data, 1) if len(window_data) > 1 else (0.0, 0.0)

        return {
            "mean": mean_val,
            "std": float(np.std(window_data)),
            "min": float(np.min(window_data)),
            "max": float(np.max(window_data)),
            "trend_slope": float(slope),
            "zero_crossing_rate": zcr
        }

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk timeseries data.
        
        Note: The BaseChunker signature expects `text: str`. For timeseries, we assume 
        `text` is a comma-separated list of floats (e.g., "12.4, 12.5, 12.6") representing
        sequential sensor readings at a fixed sample rate.
        """
        doc_id = metadata.get("doc_id", "unknown_ts")
        machine_id = metadata.get("machine_id", "unknown_machine")
        sensor_id = metadata.get("sensor_id", "unknown_sensor")
        
        chunks: List[Chunk] = []
        
        try:
            # Parse text into float array
            data = np.array([float(x.strip()) for x in text.split(",") if x.strip()])
        except ValueError:
            return chunks

        if len(data) == 0:
            return chunks

        chunk_idx = 0
        for i in range(0, len(data), self.step):
            window = data[i:i + self.window_size]
            
            # Skip if window is too small (e.g., at the very end) unless it's the only data
            if len(window) < self.window_size and chunk_idx > 0:
                break
                
            features = self._extract_features(window)
            
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_strategy": "timeseries",
                "machine_id": machine_id,
                "sensor_id": sensor_id,
                "window_start_idx": i,
                "window_end_idx": i + len(window),
                "features": features
            })
            
            # Represent the chunk textually as a JSON of its features so it can be embedded
            # (Alternatively, one could just embed the raw array if using a specialized TS model)
            text_representation = (
                f"Sensor {sensor_id} on Machine {machine_id}. "
                f"Mean: {features['mean']:.2f}, Std: {features['std']:.2f}, "
                f"Trend: {features['trend_slope']:.4f}, Range: {features['min']:.2f} to {features['max']:.2f}."
            )
            
            chunks.append(
                Chunk(
                    chunk_id=self.generate_chunk_id(doc_id, chunk_idx),
                    text=text_representation,
                    metadata=chunk_meta,
                    start_idx=i,
                    end_idx=i + len(window),
                    token_count=self.count_tokens(text_representation)
                )
            )
            chunk_idx += 1

        return chunks
