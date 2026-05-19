import logging
from typing import Any, Dict, List
from dataclasses import dataclass
from datetime import datetime

from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client.client.write_api_async import WriteApiAsync
from influxdb_client import Point

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    sensor_id: str
    machine_id: str
    value: float
    timestamp: float  # Unix epoch


@dataclass
class DataPoint:
    timestamp: float
    value: float


class AsyncInfluxDBClient:
    """
    High-throughput async wrapper for InfluxDB v2.
    Uses Batched Line Protocol for writes.
    """

    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.client: InfluxDBClientAsync = None
        self.write_api: WriteApiAsync = None

    async def connect(self) -> None:
        self.client = InfluxDBClientAsync(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api()
        logger.info("Connected to InfluxDB at %s (org: %s, bucket: %s)", self.url, self.org, self.bucket)
        await self._ensure_retention_policies()

    async def _ensure_retention_policies(self) -> None:
        """
        Mock implementation for ensuring bucket retention rules exist.
        Actual implementation would use the InfluxDB management API to set:
        - raw: 7d
        - 1min_agg: 90d
        - 1hr_agg: 2y
        """
        logger.info("Ensuring retention policies on bucket %s", self.bucket)
        pass

    async def close(self) -> None:
        if self.client:
            await self.client.close()
            logger.info("Closed InfluxDB connection.")

    async def write_batch(self, readings: List[SensorReading]) -> None:
        """Write a batch of sensor readings using high-performance line protocol."""
        if not self.write_api or not readings:
            return

        points = []
        for r in readings:
            point = Point("sensor_data") \
                .tag("machine_id", r.machine_id) \
                .tag("sensor_id", r.sensor_id) \
                .field("value", float(r.value)) \
                .time(int(r.timestamp * 1e9))  # InfluxDB expects nanoseconds
            points.append(point)

        try:
            await self.write_api.write(bucket=self.bucket, org=self.org, record=points)
        except Exception as e:
            logger.error("Failed to write batch to InfluxDB: %s", e)
            raise

    async def query_range(self, machine_id: str, metric: str, start: str, end: str, aggregation_window: str = "1m") -> List[DataPoint]:
        """Query historical time-series data with aggregation."""
        if not self.client:
            return []

        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start}, stop: {end})
          |> filter(fn: (r) => r["_measurement"] == "sensor_data")
          |> filter(fn: (r) => r["machine_id"] == "{machine_id}")
          |> filter(fn: (r) => r["_field"] == "{metric}")
          |> aggregateWindow(every: {aggregation_window}, fn: mean, createEmpty: false)
          |> yield(name: "mean")
        '''
        
        try:
            query_api = self.client.query_api()
            result = await query_api.query(query, org=self.org)
            
            data_points = []
            for table in result:
                for record in table.records:
                    ts = record.get_time().timestamp() if record.get_time() else 0.0
                    val = record.get_value()
                    data_points.append(DataPoint(timestamp=ts, value=val))
                    
            return data_points
        except Exception as e:
            logger.error("Failed to query InfluxDB: %s", e)
            return []
