import logging
from typing import Any, AsyncGenerator

import aiobotocore.session
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class MinIOClient:
    """
    Async MinIO client using aiobotocore (S3 compatible).
    Auto-creates required buckets.
    """

    REQUIRED_BUCKETS = ["raw-documents", "processed", "models"]

    def __init__(self, endpoint_url: str, access_key: str, secret_key: str, region_name: str = "us-east-1"):
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.region_name = region_name
        self.session = aiobotocore.session.get_session()

    def _create_client(self) -> Any:
        return self.session.create_client(
            "s3",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

    async def setup(self) -> None:
        """Initialize and auto-create required buckets."""
        async with self._create_client() as client:
            for bucket in self.REQUIRED_BUCKETS:
                try:
                    await client.head_bucket(Bucket=bucket)
                except ClientError as e:
                    error_code = int(e.response['Error']['Code'])
                    if error_code == 404:
                        logger.info("Bucket %s not found. Creating...", bucket)
                        await client.create_bucket(Bucket=bucket)
                    else:
                        raise e

    async def upload_file(self, bucket: str, key: str, data: bytes) -> None:
        """Upload a file to MinIO."""
        async with self._create_client() as client:
            await client.put_object(Bucket=bucket, Key=key, Body=data)
            logger.debug("Uploaded %s to bucket %s", key, bucket)

    async def get_file(self, bucket: str, key: str) -> bytes:
        """Download a file from MinIO."""
        async with self._create_client() as client:
            response = await client.get_object(Bucket=bucket, Key=key)
            async with response['Body'] as stream:
                data = await stream.read()
            return data

    async def list_files(self, bucket: str, prefix: str = "") -> AsyncGenerator[str, None]:
        """List files in a bucket with a given prefix."""
        async with self._create_client() as client:
            paginator = client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        yield obj["Key"]
