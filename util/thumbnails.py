from PIL import Image
from io import BytesIO
from fastapi import UploadFile
import os
import logging
import uuid
import base64
import aiohttp
from urllib.parse import quote

# Configure logging
logger = logging.getLogger(__name__)

async def generate_thumbnail(file: UploadFile, s3_service, file_id: str, size: tuple = (200, 200)) -> str:
    """
    Generate a thumbnail for an image file and upload it using a direct HTTP approach,
    avoiding S3 signing issues.
    
    Args:
        file: The uploaded file
        s3_service: S3 service instance
        file_id: Unique identifier for the file
        size: Thumbnail dimensions
        
    Returns:
        str: The thumbnail file key in S3, or None if generation fails
    """
    try:
        # Read the file contents
        contents = await file.read()
        
        # Generate thumbnail
        with Image.open(BytesIO(contents)) as img:
            img.thumbnail(size)
            output = BytesIO()
            # Use original format if possible, fallback to JPEG
            format = img.format if img.format else 'JPEG'
            img.save(output, format=format, quality=85)
            output.seek(0)
        
        # Generate a unique thumbnail key
        thumb_key = f"thumb_{uuid.uuid4()}"
        
        # Get content from BytesIO
        thumb_content = output.getvalue()
        
        # Determine content type
        content_type = f"image/{format.lower()}" if format != 'JPEG' else 'image/jpeg'
        
        # Try using direct HTTP PUT instead of S3 client
        # This avoids the content hash validation issues
        try:
            # Create auth string
            auth = base64.b64encode(
                f"{s3_service.config['aws_access_key_id']}:{s3_service.config['aws_secret_access_key']}".encode()
            ).decode()
            
            # Create endpoint URL
            endpoint = f"{s3_service.config['endpoint_url']}/{s3_service.bucket_name}/{thumb_key}"
            
            # Upload using direct HTTP request
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    endpoint,
                    data=thumb_content,
                    headers={
                        'Authorization': f'Basic {auth}',
                        'Content-Type': content_type,
                        'Content-Length': str(len(thumb_content)),
                        'x-amz-meta-original-filename': quote(f"{file.filename}_thumb")
                    }
                ) as resp:
                    if resp.status < 300:
                        logger.info(f"Successfully uploaded thumbnail with key {thumb_key}")
                    else:
                        error_text = await resp.text()
                        logger.error(f"Direct thumbnail upload failed with status {resp.status}: {error_text}")
                        return None
        except Exception as direct_error:
            logger.error(f"Direct HTTP upload for thumbnail failed: {str(direct_error)}")
            return None
            
        # Reset original file's position
        await file.seek(0)
        
        return thumb_key
        
    except Exception as e:
        logger.error(f"Error generating thumbnail: {str(e)}")
        # Return None instead of raising, as thumbnail is optional
        await file.seek(0)
        return None