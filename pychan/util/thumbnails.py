from PIL import Image
from io import BytesIO
from fastapi import UploadFile
import logging
import uuid
import traceback

# Configure logging
logger = logging.getLogger(__name__)

async def generate_thumbnail(file: UploadFile, s3_service, file_id: str, size: tuple = (200, 200)) -> str:
    """
    Generate a thumbnail for an image file and store it directly in Redis.
    
    Args:
        file: The uploaded file
        s3_service: S3 service instance
        file_id: Unique identifier for the file
        size: Thumbnail dimensions
        
    Returns:
        str: The thumbnail ID in Redis, or None if generation fails
    """
    try:
        from pychan.util.redis_config import redis_client
        
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
        
        # Get binary thumbnail data
        thumb_content = output.getvalue()
        
        # Determine content type
        content_type = f"image/{format.lower()}" if format != 'JPEG' else 'image/jpeg'
        
        # Generate a unique ID for the thumbnail
        thumb_id = f"thumb_{uuid.uuid4()}"
        
        # Store thumbnail directly in Redis
        logger.info(f"Storing thumbnail directly in Redis with key: thumb:{thumb_id}")
        
        # Store the actual binary data
        await redis_client.set(f"thumb:{thumb_id}", thumb_content)
        
        # Store metadata separately
        await redis_client.hset(
            f"thumb_meta:{thumb_id}", 
            mapping={
                "content_type": content_type,
                "size": str(len(thumb_content)),
                "original_file": file.filename,
                "width": str(img.width),
                "height": str(img.height)
            }
        )
        
        # Set expiration for both keys (30 days)
        await redis_client.expire(f"thumb:{thumb_id}", 60*60*24*30)
        await redis_client.expire(f"thumb_meta:{thumb_id}", 60*60*24*30)
        
        logger.info(f"Successfully stored thumbnail in Redis with ID: {thumb_id}")
        
        # Reset original file's position
        await file.seek(0)
        
        return thumb_id
        
    except Exception as e:
        logger.error(f"Error generating thumbnail: {str(e)}")
        logger.error(traceback.format_exc())
        # Return None instead of raising, as thumbnail is optional
        try:
            await file.seek(0)
        except:
            pass
        return None