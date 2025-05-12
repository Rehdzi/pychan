from aiobotocore.session import get_session
from aiobotocore.config import AioConfig
from contextlib import asynccontextmanager
from urllib.parse import quote, unquote
from typing import AsyncGenerator
import uuid
import hashlib
import traceback
import logging

from fastapi import UploadFile

# Configure logging
logger = logging.getLogger(__name__)

class S3Service:
    def __init__(
            self,
            access_key: str,
            secret_key: str,
            bucket_name: str,
            endpoint_url: str,
            region_name: str = 'ru-1'  # Default region for Timeweb
    ):
        """Initialize S3Service with Timeweb-specific configuration"""
        # Clean up endpoint URL if needed (some S3 providers prefer no trailing slash)
        if endpoint_url.endswith('/'):
            endpoint_url = endpoint_url[:-1]
            
        self.config = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "endpoint_url": endpoint_url,
            "region_name": region_name
        }
        self.bucket_name = bucket_name
        self.session = get_session()

        # Configure for Timeweb S3 compatibility
        # Timeweb often requires different settings than standard S3
        self.s3_config = AioConfig(
            signature_version='s3v4',  # Use v4 signature format
            s3={
                'addressing_style': 'path',
                'payload_signing_enabled': False,  # Try disabling payload signing which can cause issues
                'use_accelerate_endpoint': False,
                'use_dualstack_endpoint': False
            },
            retries={
                'max_attempts': 3,  # Retry for better reliability
                'mode': 'standard'
            },
            connect_timeout=30,
            read_timeout=30
        )

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator:
        """Async context manager for S3 client with proper configuration"""
        try:
            async with self.session.create_client(
                    "s3",
                    config=self.s3_config,  # Use the configured AioConfig
                    **self.config
            ) as client:
                # For Timeweb: remove any custom headers that might interfere
                client._request_signer._additional_headers = {}
                
                # Log successful connection
                logger.debug(f"Connected to S3 endpoint: {self.config['endpoint_url']}")
                
                yield client
        except Exception as e:
            logger.error(f"Failed to create S3 client: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def upload_file(self, upload_file: UploadFile) -> str:
        """Upload file to Timeweb S3 storage with special handling for Timeweb compatibility"""
        file_key = f"{uuid.uuid4()}"
        
        try:
            # Create a copy of the file in memory to avoid streaming issues
            contents = await upload_file.read()
            await upload_file.seek(0)  # Reset the file pointer
            
            # Timeweb S3 can be picky about content types
            content_type = upload_file.content_type or 'application/octet-stream'
            
            # Try multiple approaches to handle Timeweb's unique requirements
            for attempt in range(3):
                try:
                    if attempt == 0:
                        # First attempt: Standard approach with minimal headers
                        async with self.session.create_client(
                                "s3",
                                config=AioConfig(
                                    signature_version='s3v4',
                                    s3={'addressing_style': 'path', 'payload_signing_enabled': False}
                                ),
                                **self.config
                        ) as client:
                            await client.put_object(
                                Bucket=self.bucket_name,
                                Key=file_key,
                                Body=contents,
                                ContentType=content_type
                            )
                            break
                            
                    elif attempt == 1:
                        # Second attempt: Try with checksums disabled
                        async with self.session.create_client(
                                "s3",
                                config=AioConfig(
                                    signature_version='s3',
                                    s3={'addressing_style': 'path', 'payload_signing_enabled': False}
                                ),
                                **self.config
                        ) as client:
                            await client.put_object(
                                Bucket=self.bucket_name,
                                Key=file_key,
                                Body=contents,
                                ContentType=content_type
                            )
                            break
                            
                    elif attempt == 2:
                        # Third attempt: Try with basic authentication approach
                        import aiohttp
                        import base64
                        
                        auth = base64.b64encode(
                            f"{self.config['aws_access_key_id']}:{self.config['aws_secret_access_key']}".encode()
                        ).decode()
                        
                        endpoint = f"{self.config['endpoint_url']}/{self.bucket_name}/{file_key}"
                        
                        async with aiohttp.ClientSession() as session:
                            async with session.put(
                                endpoint,
                                data=contents,
                                headers={
                                    'Authorization': f'Basic {auth}',
                                    'Content-Type': content_type
                                }
                            ) as resp:
                                if resp.status < 300:
                                    break
                                else:
                                    logger.error(f"Direct PUT failed: {await resp.text()}")
                                    raise Exception(f"HTTP error: {resp.status}")
                                
                except Exception as e:
                    logger.warning(f"Upload attempt {attempt+1} failed: {str(e)}")
                    if attempt == 2:  # Last attempt
                        raise
                        
            logger.info(f"Successfully uploaded file {upload_file.filename} to S3 with key {file_key}")
            return file_key
            
        except Exception as e:
            logger.error(f"S3 upload error: {str(e)}")
            logger.error(f"File: {upload_file.filename}, Content-Type: {upload_file.content_type}")
            logger.error(traceback.format_exc())
            raise TimewebS3Error(f"Upload failed: {str(e)}")

    async def upload_file_direct_http(self, upload_file: UploadFile) -> str:
        """
        Upload file using direct HTTP PUT, bypassing S3 SDK entirely.
        This is a fallback method when normal S3 upload fails.
        """
        try:
            import aiohttp
            import base64
            
            # Generate a unique file key
            file_key = f"direct_{uuid.uuid4()}"
            
            # Read file contents
            contents = await upload_file.read()
            await upload_file.seek(0)
            
            # Create basic auth for S3
            auth = base64.b64encode(
                f"{self.config['aws_access_key_id']}:{self.config['aws_secret_access_key']}".encode()
            ).decode()
            
            # Create endpoint URL
            endpoint = f"{self.config['endpoint_url']}/{self.bucket_name}/{file_key}"
            
            # Determine content type
            content_type = upload_file.content_type or 'application/octet-stream'
            
            # Upload using direct HTTP PUT
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    endpoint,
                    data=contents,
                    headers={
                        'Authorization': f'Basic {auth}',
                        'Content-Type': content_type,
                        'Content-Length': str(len(contents)),
                        'x-amz-meta-original-filename': quote(upload_file.filename)
                    }
                ) as resp:
                    if resp.status < 300:
                        logger.info(f"Direct HTTP upload successful for file {upload_file.filename}")
                        return file_key
                    else:
                        error_text = await resp.text()
                        logger.error(f"Direct HTTP upload failed with status {resp.status}: {error_text}")
                        raise TimewebS3Error(f"Direct HTTP upload failed with status {resp.status}")
                        
        except Exception as e:
            logger.error(f"Direct HTTP upload error: {str(e)}")
            logger.error(traceback.format_exc())
            raise TimewebS3Error(f"Direct HTTP upload failed: {str(e)}")

    async def generate_presigned_url(self, file_key: str, expires: int = 3600) -> str:
        """Генерация временной ссылки на файл"""
        async with self.get_client() as client:
            try:
                url = await client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': self.bucket_name,
                        'Key': file_key
                    },
                    ExpiresIn=expires
                )
                return url
            except Exception as e:
                raise TimewebS3Error(f"URL generation failed: {str(e)}")

    async def get_file_metadata(self, file_key: str) -> dict:
        """Получение метаданных файла"""
        async with self.get_client() as client:
            try:
                response = await client.head_object(
                    Bucket=self.bucket_name,
                    Key=file_key
                )
                metadata = response["Metadata"]
                return {
                    "filename": unquote(metadata.get("original-filename", "")),
                    "size": int(metadata.get("content-length", 0)),
                    "content_type": response["ContentType"],
                    "last_modified": response["LastModified"]
                }
            except Exception as e:
                raise TimewebS3Error(f"Metadata fetch failed: {str(e)}")

    async def delete_file(self, file_key: str) -> None:
        """Удаление файла из хранилища"""
        async with self.get_client() as client:
            try:
                await client.delete_object(
                    Bucket=self.bucket_name,
                    Key=file_key
                )
            except Exception as e:
                raise TimewebS3Error(f"Deletion failed: {str(e)}")

    async def check_connection(self) -> bool:
        """
        Test the S3 connection and configuration.
        Returns True if connection is successful, False otherwise.
        """
        try:
            # Try to list buckets as a simple test
            async with self.get_client() as client:
                # Just list the buckets to verify credentials
                await client.list_buckets()
            
            # Additional validation specific to the bucket we want to use
            try:
                async with self.get_client() as client:
                    await client.head_bucket(Bucket=self.bucket_name)
            except Exception as e:
                logger.error(f"Bucket access error: {str(e)}")
                return False
                
            logger.info(f"Successfully connected to S3 and verified access to bucket {self.bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"S3 connection test failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False


class TimewebS3Error(Exception):
    """Кастомное исключение для ошибок Timeweb S3"""
    pass