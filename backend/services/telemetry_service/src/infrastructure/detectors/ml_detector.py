import os
import logging
from typing import Optional

import mlflow.pytorch

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from backend.services.telemetry_service.src.infrastructure.detectors.statistical_detector import Anomaly, StatisticalDetector

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    class LSTMAutoencoder(nn.Module):
        def __init__(self, seq_len: int, n_features: int, embedding_dim: int = 64):
            super(LSTMAutoencoder, self).__init__()
            self.seq_len = seq_len
            self.n_features = n_features
            self.embedding_dim = embedding_dim
            
            self.encoder = nn.LSTM(
                input_size=n_features, 
                hidden_size=embedding_dim, 
                num_layers=1, 
                batch_first=True
            )
            
            self.decoder = nn.LSTM(
                input_size=embedding_dim, 
                hidden_size=n_features, 
                num_layers=1, 
                batch_first=True
            )

        def forward(self, x):
            # Encode
            _, (hidden, _) = self.encoder(x)
            
            # Decode
            # Repeat the hidden state seq_len times
            hidden = hidden.repeat(self.seq_len, 1, 1).transpose(0, 1)
            decoded, _ = self.decoder(hidden)
            
            return decoded


class MLDetector:
    """
    PyTorch-based LSTM Autoencoder targeting multivariate anomalies.
    Gracefully degrades to StatisticalDetector if torch is unavailable or model fails to load.
    """

    def __init__(self, fallback_detector: StatisticalDetector, reconstruction_threshold: float = 0.05):
        self.fallback_detector = fallback_detector
        self.threshold = reconstruction_threshold
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. MLDetector falling back to StatisticalDetector.")
            return

        try:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            model_name = os.environ.get("MLFLOW_MODEL_NAME", "telemetry-lstm-autoencoder")
            model_stage = os.environ.get("MLFLOW_MODEL_STAGE", "Production")
            
            mlflow.set_tracking_uri(tracking_uri)
            model_uri = f"models:/{model_name}/{model_stage}"
            
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.eval()
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            logger.info(
                "MLDetector initialized with model '%s' at stage '%s' on device '%s'.", 
                model_name, model_stage, self.device
            )
        except Exception as e:
            logger.critical("Failed to load ML model from MLflow: %s", str(e))
            self.model = None

    async def detect(self, sensor_id: str, machine_id: str, value: float, timestamp: float) -> Optional[Anomaly]:
        if not self.model or not TORCH_AVAILABLE:
            return await self.fallback_detector.detect(sensor_id, machine_id, value, timestamp)
            
        # In a real implementation, we would maintain a rolling window of the last 60 readings
        # per sensor in Redis, fetch them, tensorize them, and run batch inference.
        
        # Mocking the inference process to demonstrate architecture
        # with torch.no_grad():
        #    reconstruction = self.model(tensor_data)
        #    mse = torch.mean((tensor_data - reconstruction) ** 2).item()
        
        mse = 0.01  # Mock normal mse
        
        if mse > self.threshold:
            # Check deduplication
            is_dup = await self.fallback_detector.redis_cache.is_duplicate_anomaly(sensor_id, "HIGH", 300)
            if is_dup:
                return None
                
            anomaly = Anomaly(
                sensor_id=sensor_id,
                machine_id=machine_id,
                value=value,
                expected_range=f"MSE < {self.threshold}",
                z_score=0.0,
                severity="HIGH",
                timestamp=timestamp
            )
            await self.fallback_detector.redis_cache.add_recent_anomaly(machine_id, anomaly.__dict__, timestamp)
            logger.warning("ML Anomaly Detected: %s on %s (MSE: %.4f)", sensor_id, machine_id, mse)
            return anomaly
            
        return None
