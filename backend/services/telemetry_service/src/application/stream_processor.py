import asyncio
import json
import logging
import time
from typing import List

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from prometheus_client import Counter, Histogram

from backend.services.telemetry_service.src.infrastructure.time_series_db.influxdb_client import AsyncInfluxDBClient, SensorReading
from backend.services.telemetry_service.src.infrastructure.cache.redis_cache import TelemetryRedisCache
from backend.services.telemetry_service.src.infrastructure.detectors.statistical_detector import StatisticalDetector
from backend.services.telemetry_service.src.infrastructure.detectors.rule_detector import RuleDetector
from backend.services.telemetry_service.src.infrastructure.detectors.ml_detector import MLDetector

logger = logging.getLogger(__name__)

# Prometheus Metrics
MESSAGES_PROCESSED = Counter("messages_processed_total", "Total number of telemetry messages processed")
PROCESSING_LATENCY = Histogram("processing_latency_ms", "Latency of batch processing in milliseconds", buckets=[10, 25, 50, 100, 200, 500])
ANOMALIES_DETECTED = Counter("anomalies_detected_total", "Total number of anomalies detected", ["severity", "detector_type"])
DETECTOR_ERRORS = Counter("detector_errors_total", "Errors encountered during anomaly detection")


class TelemetryStreamProcessor:
    """
    High-throughput asynchronous stream processor for telemetry data.
    Consumes from Kafka, processes batches concurrently across InfluxDB and ML/Statistical detectors.
    """

    def __init__(
        self,
        kafka_bootstrap_servers: str,
        influx_client: AsyncInfluxDBClient,
        redis_cache: TelemetryRedisCache,
        stat_detector: StatisticalDetector,
        rule_detector: RuleDetector,
        ml_detector: MLDetector,
        batch_size: int = 500,
        flush_interval_ms: int = 100
    ):
        self.bootstrap_servers = kafka_bootstrap_servers
        self.influx_client = influx_client
        self.redis_cache = redis_cache
        self.stat_detector = stat_detector
        self.rule_detector = rule_detector
        self.ml_detector = ml_detector
        
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_ms / 1000.0
        
        self.consumer = None
        self.producer = None
        self._running = False

    async def start(self):
        """Start the Kafka consumer loop."""
        self.consumer = AIOKafkaConsumer(
            "ikb.sensors.raw",
            bootstrap_servers=self.bootstrap_servers,
            group_id="telemetry_processor_group",
            auto_offset_reset="latest",
            enable_auto_commit=False # Manual commit after batch processing
        )
        self.producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers)
        
        await self.consumer.start()
        await self.producer.start()
        self._running = True
        logger.info("Telemetry Stream Processor started.")
        
        asyncio.create_task(self._consume_loop())

    async def stop(self):
        """Stop the processor gracefully."""
        self._running = False
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
        logger.info("Telemetry Stream Processor stopped.")

    async def _consume_loop(self):
        """Consume messages using a batching strategy."""
        batch = []
        last_flush = time.time()
        
        try:
            while self._running:
                try:
                    # Wait for message with a timeout to evaluate flush interval
                    msg = await asyncio.wait_for(self.consumer.getone(), timeout=0.01)
                    batch.append(msg)
                except asyncio.TimeoutError:
                    pass
                
                now = time.time()
                if len(batch) >= self.batch_size or (now - last_flush) >= self.flush_interval_seconds:
                    if batch:
                        await self._process_batch(batch)
                        await self.consumer.commit()
                        batch = []
                    last_flush = now
        except Exception as e:
            logger.error("Fatal error in consume loop: %s", e)

    async def _process_batch(self, messages: List[Any]):
        """
        Process a batch concurrently.
        """
        start_time = time.time()
        readings = []
        
        for msg in messages:
            try:
                data = json.loads(msg.value.decode("utf-8"))
                reading = SensorReading(
                    sensor_id=data["sensor_id"],
                    machine_id=data["machine_id"],
                    value=float(data["value"]),
                    timestamp=float(data.get("timestamp", time.time()))
                )
                readings.append(reading)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.debug("Malformed message skipped: %s", e)
                
        if not readings:
            return

        # Parallel Execution using asyncio.gather
        # 1. Write to InfluxDB
        # 2. Update Feature Store (handled within detectors implicitly here, or distinct task)
        # 3. Run Anomaly Detection
        
        try:
            results = await asyncio.gather(
                self._write_to_influx(readings),
                self._run_anomaly_detection(readings),
                return_exceptions=True
            )
            
            # Check for exceptions
            for r in results:
                if isinstance(r, Exception):
                    logger.error("Batch processing sub-task failed: %s", r)
                    
        except Exception as e:
            logger.error("Batch parallel execution failed: %s", e)
            
        MESSAGES_PROCESSED.inc(len(readings))
        
        latency_ms = (time.time() - start_time) * 1000
        PROCESSING_LATENCY.observe(latency_ms)
        
        # Backpressure warning
        if latency_ms > 200:
            logger.warning("Backpressure detected! Batch processing took %.2f ms", latency_ms)

    async def _write_to_influx(self, readings: List[SensorReading]):
        await self.influx_client.write_batch(readings)

    async def _run_anomaly_detection(self, readings: List[SensorReading]):
        """Run multi-tier anomaly detection on the batch."""
        for r in readings:
            try:
                # We can execute these in parallel per reading or sequentially.
                # Sequentially per reading is usually fine if detectors are fast O(1).
                
                # 1. Rule Detector (Hard bounds)
                anomaly = await self.rule_detector.detect(r.sensor_id, r.machine_id, r.value, r.timestamp)
                detector_type = "rule"
                
                # 2. Statistical Detector (if no rule breach)
                if not anomaly:
                    anomaly = await self.stat_detector.detect(r.sensor_id, r.machine_id, r.value, r.timestamp)
                    detector_type = "statistical"
                    
                # 3. ML Detector (Multivariate, if no statistical breach)
                if not anomaly:
                    anomaly = await self.ml_detector.detect(r.sensor_id, r.machine_id, r.value, r.timestamp)
                    detector_type = "ml"
                    
                if anomaly:
                    ANOMALIES_DETECTED.labels(severity=anomaly.severity, detector_type=detector_type).inc()
                    await self._handle_anomaly(anomaly)
                    
            except Exception as e:
                DETECTOR_ERRORS.inc()
                logger.error("Detector error for sensor %s: %s", r.sensor_id, e)

    async def _handle_anomaly(self, anomaly: Any):
        """Publish anomalies to Kafka, routing HIGH/CRITICAL to Agent tasks."""
        anomaly_dict = anomaly.__dict__
        payload = json.dumps(anomaly_dict).encode("utf-8")
        
        # 1. Standard Anomaly Feed
        await self.producer.send_and_wait("ikb.anomalies", value=payload, key=anomaly.machine_id.encode("utf-8"))
        
        # 2. RCA Escalation
        if anomaly.severity in ["HIGH", "CRITICAL"]:
            # Route to Agent orchestrator topic
            agent_task = {
                "task_type": "anomaly_analysis",
                "machine_id": anomaly.machine_id,
                "sensor_id": anomaly.sensor_id,
                "severity": anomaly.severity,
                "trigger_value": anomaly.value
            }
            await self.producer.send_and_wait(
                "ikb.agent.tasks", 
                value=json.dumps(agent_task).encode("utf-8"), 
                key=anomaly.machine_id.encode("utf-8")
            )
            logger.info("CRITICAL anomaly routed to ikb.agent.tasks for RCA on machine %s", anomaly.machine_id)
