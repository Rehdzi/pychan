import imghdr
import json
import logging
import os
import traceback
import uuid
import sys

import uvicorn
from loguru import logger
from functools import lru_cache
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from redis.asyncio import Redis
from sqlalchemy import select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse

from db.database import get_db
from db.models import *
from util.s3_connect import S3Service
from util.schemas import *
from util.thumbnails import generate_thumbnail

load_dotenv()

# Remove existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller to get correct stack depth
        frame, depth = logging.currentframe(), 2
        while frame.f_back and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Intercept standard logging
logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)

logger.add(
    "logs/app.log",
    rotation="50 MB",
    compression="zip",
    level="INFO",
    backtrace=True,
    diagnose=True,
)

loggers = (
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "fastapi",
    "sqlalchemy",
    "asyncio",
    "starlette",
)

for logger_name in loggers:
    logging_logger = logging.getLogger(logger_name)
    logging_logger.handlers = []
    logging_logger.propagate = True

# Environment variables with default values
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_BUCKET = os.getenv("S3_BUCKET", "images")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Validate required environment variables
if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_ENDPOINT]):
    logger.error("Missing required environment variables for S3 connection")


# Error handling middleware
class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            error_id = str(uuid.uuid4())
            logger.error(
                f"Error ID: {error_id} - Request: {request.method} {request.url}\n"
                f"Error: {str(e)}\n{traceback.format_exc()}"
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "error_id": error_id,
                    "message": str(e) if os.getenv("DEBUG") == "true" else "An unexpected error occurred"
                }
            )


app = FastAPI(title="Image Board API", version="1.0.0")

# Add middleware
app.add_middleware(ErrorLoggingMiddleware)

origins = [
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Startup event to test connections
@app.on_event("startup")
async def startup_event():
    # Test Redis connection
    from util.redis_config import test_redis_connection
    await test_redis_connection()

    # Test S3 connection
    s3_service = get_s3_service()
    s3_connected = await s3_service.check_connection()
    if not s3_connected:
        logger.warning("S3 connection test failed - file uploads may not work")
    else:
        logger.info("S3 connection verified successfully")

    # Log application startup
    logger.info("Application started successfully")


# Shutdown event handler to clean up resources
@app.on_event("shutdown")
async def shutdown_event():
    global _redis_pool
    # Close Redis connection pool
    if _redis_pool:
        await _redis_pool.close()
        _redis_pool = None
    logger.info("Application shutdown, resources cleaned up")


# Create a cached S3 service to reduce connection overhead
@lru_cache()
def get_s3_service():
    return S3Service(
        access_key=AWS_ACCESS_KEY,
        secret_key=AWS_SECRET_KEY,
        bucket_name=S3_BUCKET,
        endpoint_url=S3_ENDPOINT,
        region_name='ru-1'
    )


async def get_s3() -> S3Service:
    return get_s3_service()


# Redis connection pool to avoid creating new connections
_redis_pool = None


async def get_redis() -> Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_pool


# Cache for frequently accessed data
CACHE_TTL = 60 * 5  # 5 minutes


@app.get("/categories/")
async def get_categories(db: AsyncSession = Depends(get_db), redis: Redis = Depends(get_redis)):
    try:
        # Try to get from cache first
        cached = await redis.get("cache:categories")
        if cached:
            return json.loads(cached)

        # If not in cache, query database
        result = await db.execute(select(Category))
        categories = result.scalars().all()

        # Cache the result
        categories_dict = [cat.to_dict() for cat in categories]
        await redis.set("cache:categories", json.dumps(categories_dict), ex=CACHE_TTL)

        return categories
    except Exception as e:
        logger.error(f"Error in get_categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/boardlist")
async def get_categories_with_boards(
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis)
):
    try:
        # Try to get from cache first
        cached = await redis.get("cache:boardlist")
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                logger.warning("Failed to decode cached boardlist data")

        # If not in cache, query database with explicit eager loading
        # This avoids lazy loading issues by loading everything at once
        query = select(Category).options(
            selectinload(Category.boards)
        ).order_by(Category.id)

        result = await db.execute(query)
        categories = result.scalars().all()

        # Manually construct the response without relying on lazy loading
        response = []
        for cat in categories:
            # Skip hidden categories
            if not cat.is_visible:
                continue

            # Only include visible boards
            visible_boards = []
            for board in cat.boards:
                if board.is_visible:
                    visible_boards.append({
                        "id": board.id,
                        "tag": board.tag,
                        "name": board.name,
                        "description": board.description,
                        "nsfw": board.nsfw,
                        "is_visible": board.is_visible,
                        "is_locked": board.is_locked
                    })

            response.append({
                "id": cat.id,
                "name": cat.name,
                "is_visible": cat.is_visible,
                "is_nsfw": cat.is_nsfw,
                "boards": visible_boards
            })

        # Cache the result
        await redis.set("cache:boardlist", json.dumps(response), ex=CACHE_TTL)

        return response
    except Exception as e:
        logger.error(f"Error in get_categories_with_boards: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# Combine these two endpoints into one with a query parameter
@app.get("/boards/")
async def get_boards(
        sfw_only: bool = False,
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis)
):
    """
    Get all boards or only SFW boards based on the parameter.
    Using a more direct approach to avoid greenlet issues.
    """
    try:
        # Define cache key
        cache_key = "cache:boards:sfw" if sfw_only else "cache:boards:all"

        # Try cache first, with direct conversion
        cached_data = None
        cached = await redis.get(cache_key)
        if cached:
            try:
                cached_data = json.loads(cached)
                return cached_data
            except json.JSONDecodeError:
                # Continue to DB query if cache is invalid
                pass

        # Direct DB query with connection management
        async with db.begin():
            # Build query
            if sfw_only:
                stmt = sa_text("SELECT * FROM board WHERE nsfw = false")
            else:
                stmt = sa_text("SELECT * FROM board")

            # Use raw SQL to avoid SQLAlchemy greenlet issues
            result = await db.execute(stmt)
            rows = result.all()

            # Manual mapping to dictionaries
            boards_data = []
            for row in rows:
                board_dict = {}
                for key, value in row._mapping.items():
                    if isinstance(value, datetime):
                        board_dict[key] = value.isoformat()
                    else:
                        board_dict[key] = value
                boards_data.append(board_dict)

        # Cache with direct JSON
        if boards_data:
            json_data = json.dumps(boards_data)
            await redis.set(cache_key, json_data, ex=CACHE_TTL)

        return boards_data

    except Exception as e:
        logger.error(f"Error in get_boards: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# Replace the sfw_boards endpoint to avoid dependency on the main endpoint
@app.get("/sfw_boards/", include_in_schema=False)
async def get_sfw_boards(
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis)
):
    """
    Get only SFW boards using a direct SQL approach.
    """
    try:
        # Define cache key
        cache_key = "cache:boards:sfw"

        # Try cache first
        cached = await redis.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass

        # Direct DB query with explicit connection management
        async with db.begin():
            # Use raw SQL to avoid SQLAlchemy ORM greenlet issues
            stmt = sa_text("SELECT * FROM board WHERE nsfw = false")
            result = await db.execute(stmt)
            rows = result.all()

            # Manual mapping
            boards_data = []
            for row in rows:
                board_dict = {}
                for key, value in row._mapping.items():
                    if isinstance(value, datetime):
                        board_dict[key] = value.isoformat()
                    else:
                        board_dict[key] = value
                boards_data.append(board_dict)

        # Cache results
        if boards_data:
            json_data = json.dumps(boards_data)
            await redis.set(cache_key, json_data, ex=CACHE_TTL)

        return boards_data

    except Exception as e:
        logger.error(f"Error in get_sfw_boards: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/{tag}")
async def get_board(
        tag: str,
        db: AsyncSession = Depends(get_db)
):
    """Get a specific board by its tag."""
    try:
        # Query for board with explicit awaits and eager loading of relationships
        # Use joinedload to load the category relationship upfront
        query = select(Board).options(joinedload(Board.category)).where(Board.tag == tag)
        result = await db.execute(query)
        board = result.scalar_one_or_none()

        if not board:
            raise HTTPException(status_code=404, detail="Board not found")

        # Convert to dict without relying on lazy loading
        board_dict = {
            "id": board.id,
            "tag": board.tag,
            "name": board.name,
            "description": board.description,
            "nsfw": board.nsfw,
            "is_visible": board.is_visible,
            "is_locked": board.is_locked,
            "category_id": board.category_id,
            # Include category info if available
            "category": {
                "id": board.category.id,
                "name": board.category.name
            } if board.category else None
        }

        return board_dict

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error in get_board: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# SELECT * FROM post
# JOIN board ON post.board_id = board.id
# WHERE board.nsfw = false
# ORDER BY timestamp DESC LIMIT 8
@app.get("/boards/latest/")
async def get_users(db: AsyncSession = Depends(get_db)):
    query = (select(Post)
             .join(Board)
             .where(Board.nsfw == False)
             .where(Post.parent_id == 0)
             .order_by(Post.timestamp.desc())
             .limit(8)
             )

    result = await db.execute(query)
    boards = result.scalars().all()
    return boards


@app.get("/boards/{board_id}/posts/")
async def get_posts_by_board(
        board_id: int,
        db: AsyncSession = Depends(get_db),
):
    """Get all posts for a specific board."""
    try:
        # Check if board exists
        board_result = await db.execute(select(Board).where(Board.id == board_id))
        board = board_result.scalar_one_or_none()

        if not board:
            raise HTTPException(status_code=404, detail="Board not found")

        # Query for posts with explicit awaits
        query = (
            select(Post)
            .where(Post.board_id == board_id)
            .order_by(Post.timestamp.desc())
        )

        result = await db.execute(query)

        # Convert to list of dicts for safe serialization
        posts_data = []
        for post in result.scalars().all():
            posts_data.append(post.to_dict())

        return posts_data

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error in get_posts_by_board: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


## @app.get("/thread/{post_id}")

async def get_image_urls(image_ids: list[str]) -> list[str]:
    """Generates URLs for images that exist in Redis, with batch processing."""
    if not image_ids:
        logger.debug("No image IDs provided to get_image_urls")
        return []

    logger.info(f"Getting URLs for image IDs: {image_ids}")

    # Handle case where image_ids might be None or not a proper list
    if image_ids is None:
        return []

    # Convert to list if it's not already
    if not isinstance(image_ids, list):
        try:
            # Try to convert if it's some other sequence type
            image_ids = list(image_ids)
        except (TypeError, ValueError):
            logger.error(f"Could not convert image_ids to list: {image_ids}")
            return []

    try:
        urls = []
        # Get Redis connection from the global pool
        redis = await get_redis()

        # Use pipelining for faster Redis operations
        pipe = redis.pipeline()

        # Queue up all the exists commands
        for image_id in image_ids:
            if image_id:  # Skip empty IDs
                pipe.exists(f"file:{image_id}")

        # Execute the pipeline
        results = await pipe.execute()

        # Create URLs for existing files - use direct ID as URL
        urls = [f"{image_id}" for image_id, exists in zip(image_ids, results) if exists and image_id]

        logger.info(f"Generated {len(urls)} URLs for {len(image_ids)} image IDs")
        return urls

    except Exception as e:
        logger.error(f"Error in get_image_urls: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty list on error rather than failing
        return []


async def get_board_ops_with_replies(async_session: AsyncSession, board_id: int):
    """
    Get original posts with their first and last replies for a board.
    Uses optimized queries to avoid lazy loading issues.
    """
    try:
        logger.info(f"Fetching OPs for board ID: {board_id}")

        # Use a query that explicitly gets image_ids as a text array and casts it properly
        ops_query = sa_text("""
            SELECT p.id, p.title, p.text as text_, p.timestamp, p.parent_id, p.board_id, p.is_visible, 
                  p.image_ids::text[] as image_ids, p.child_ids 
            FROM post p
            WHERE p.parent_id = 0 AND p.board_id = :board_id
            ORDER BY p.timestamp DESC
        """)

        ops_result = await async_session.execute(ops_query, {"board_id": board_id})
        ops = ops_result.fetchall()

        logger.info(f"Found {len(ops)} OPs for board {board_id}")

        # Return empty if no OPs
        if not ops:
            return []

        # Get all replies in a single query
        op_ids = [op.id for op in ops]
        logger.info(f"Searching for replies to OP IDs: {op_ids}")

        # Handle SQL IN clause carefully for any number of IDs
        if len(op_ids) == 1:
            # Special case for a single ID to avoid syntax issues
            replies_query = sa_text("""
                SELECT r.id, r.title, r.text as text_, r.timestamp, r.parent_id, r.board_id, r.is_visible, 
                      r.image_ids::text[] as image_ids, r.child_ids
                FROM post r 
                WHERE r.parent_id = :op_id
                ORDER BY r.timestamp ASC
            """)
            replies_result = await async_session.execute(replies_query, {"op_id": op_ids[0]})
        else:
            # Multiple IDs case
            placeholder = ', '.join([f':id{i}' for i in range(len(op_ids))])
            replies_query = sa_text(f"""
                SELECT r.id, r.title, r.text as text_, r.timestamp, r.parent_id, r.board_id, r.is_visible, 
                      r.image_ids::text[] as image_ids, r.child_ids
                FROM post r 
                WHERE r.parent_id IN ({placeholder})
                ORDER BY r.timestamp ASC
            """)

            # Build parameters dictionary
            params = {f"id{i}": op_id for i, op_id in enumerate(op_ids)}
            replies_result = await async_session.execute(replies_query, params)

        replies = replies_result.fetchall()
        logger.info(f"Found {len(replies)} total replies")

        # Group replies by parent_id
        replies_by_parent = {}
        for reply in replies:
            parent_id = reply.parent_id
            if parent_id not in replies_by_parent:
                replies_by_parent[parent_id] = []
            replies_by_parent[parent_id].append(reply)

        # Format the results to match the expected output
        result = []
        for op in ops:
            # Add missing attributes to make the objects compatible with the ORM model
            op_dict = dict(op._mapping)

            # Log image_ids for debugging
            logger.info(f"Post {op.id} image_ids from DB: {op.image_ids}")

            # Create a proper object from the row
            op_obj = type('Post', (), {
                'id': op.id,
                'title': op.title,
                'text': op.text_,
                'timestamp': op.timestamp,
                'parent_id': op.parent_id,
                'board_id': op.board_id,
                'is_visible': op.is_visible,
                'image_ids': op.image_ids if hasattr(op, 'image_ids') else [],
                'child_ids': op.child_ids if hasattr(op, 'child_ids') else []
            })

            # Get first and last reply
            op_replies = replies_by_parent.get(op.id, [])

            first_reply = None
            last_reply = None

            if op_replies:
                # Create reply objects
                first_reply_data = op_replies[0]
                first_reply = type('Post', (), {
                    'id': first_reply_data.id,
                    'text': first_reply_data.text_,
                    'timestamp': first_reply_data.timestamp,
                    'image_ids': first_reply_data.image_ids if hasattr(first_reply_data, 'image_ids') else []
                })

                if len(op_replies) > 1:
                    last_reply_data = op_replies[-1]
                    last_reply = type('Post', (), {
                        'id': last_reply_data.id,
                        'text': last_reply_data.text_,
                        'timestamp': last_reply_data.timestamp,
                        'image_ids': last_reply_data.image_ids if hasattr(last_reply_data, 'image_ids') else []
                    })
                else:
                    last_reply = first_reply

            result.append((op_obj, first_reply, last_reply))

        logger.info(f"Returning {len(result)} OPs with their replies")
        return result

    except Exception as e:
        logger.error(f"Error in get_board_ops_with_replies: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty list on error to avoid cascading failures
        return []


@app.get("/{board_tag}/ops", response_model=BoardOpsResponse)
async def get_board_operations(
        board_tag: str,
        session: AsyncSession = Depends(get_db)
):
    try:
        # Get the board with eager loading to avoid lazy loading issues
        board_query = await session.execute(
            select(Board)
            .where(Board.tag == board_tag)
            .options(joinedload(Board.category))
        )
        board = board_query.scalar_one_or_none()

        if not board:
            raise HTTPException(status_code=404, detail="Board not found")
        if not board.is_visible:
            raise HTTPException(status_code=403, detail="Board is hidden")

        # Get OP posts with eager loading for child_ids and image_ids
        ops_data = await get_board_ops_with_replies(session, board.id)

        # Prepare the response
        formatted_ops = []
        for op, first_reply, last_reply in ops_data:
            # Get image IDs, looking for both attribute and property access
            image_ids = []

            # First try to get image_ids as a property
            if hasattr(op, "image_ids") and op.image_ids:
                # For objects where image_ids is a property
                image_ids = op.image_ids
            # Also check if it's a key in a dict-like object
            elif hasattr(op, "__getitem__") and "image_ids" in op:
                # For dict-like objects
                image_ids = op["image_ids"]

            logger.info(f"Post {op.id} has image_ids: {image_ids}")

            # Get detailed image data with thumbnails and full URLs
            op_image_data = await get_image_data(image_ids)

            # For backward compatibility also provide simple image_urls
            op_image_urls = [img["url"] for img in op_image_data]

            # Create the OP data dictionary with image URLs
            op_data = {
                "id": op.id,
                "title": op.title,
                "text": op.text,
                "timestamp": op.timestamp.isoformat(),
                "image_urls": op_image_urls,  # For backward compatibility
                "images": op_image_data,  # New field with complete image info
                "image_ids": image_ids,  # Original IDs for reference
                "board": {"tag": board.tag, "name": board.name}
            }

            # Process first reply if exists
            first_reply_data = None
            if first_reply:
                # Get image data for first reply
                reply_image_ids = []
                if hasattr(first_reply, "image_ids") and first_reply.image_ids:
                    reply_image_ids = first_reply.image_ids
                elif hasattr(first_reply, "__getitem__") and "image_ids" in first_reply:
                    reply_image_ids = first_reply["image_ids"]

                first_reply_image_data = await get_image_data(reply_image_ids)
                first_reply_urls = [img["url"] for img in first_reply_image_data]

                # Create first reply data
                first_reply_data = PostReply(
                    id=first_reply.id,
                    text=first_reply.text,
                    image_urls=first_reply_urls,
                    timestamp=first_reply.timestamp,
                    images=first_reply_image_data  # Add the detailed image data
                )

            # Process last reply if different from first
            last_reply_data = None
            if last_reply and last_reply.id != (first_reply.id if first_reply else None):
                # Get image data for last reply
                reply_image_ids = []
                if hasattr(last_reply, "image_ids") and last_reply.image_ids:
                    reply_image_ids = last_reply.image_ids
                elif hasattr(last_reply, "__getitem__") and "image_ids" in last_reply:
                    reply_image_ids = last_reply["image_ids"]

                last_reply_image_data = await get_image_data(reply_image_ids)
                last_reply_urls = [img["url"] for img in last_reply_image_data]

                # Create last reply data
                last_reply_data = PostReply(
                    id=last_reply.id,
                    text=last_reply.text,
                    image_urls=last_reply_urls,
                    timestamp=last_reply.timestamp,
                    images=last_reply_image_data  # Add the detailed image data
                )

            # Add to formatted ops
            formatted_ops.append(BoardOpResponse(
                op=op_data,
                first_reply=first_reply_data,
                last_reply=last_reply_data,
                replies_count=len(op.child_ids) if hasattr(op, "child_ids") and op.child_ids else 0
            ))

        # Create the final response
        return BoardOpsResponse(
            board_info={
                "tag": board.tag,
                "name": board.name,
                "description": board.description,
                "category": board.category.name if board.category else None
            },
            ops=formatted_ops
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error in get_board_operations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving board operations: {str(e)}")


@app.get("/{file_key:path}")
async def get_media(file_key: str, s3: S3Service = Depends(get_s3)):
    """
    Serve media files from S3. Uses path parameter to support nested paths.
    If file_key doesn't start with 'media/', prepend it automatically.
    """
    try:
        # Ensure file_key starts with media/
        if not file_key.startswith("media/"):
            file_key = f"media/{file_key}"

        logger.info(f"Generating presigned URL for: {file_key}")
        url = await s3.generate_presigned_url(file_key)
        return RedirectResponse(url)
    except Exception as e:
        logger.error(f"Error generating presigned URL for {file_key}: {str(e)}")
        raise HTTPException(404, "File not found")


@app.post("/new_thread/", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(
        board_tag: str = Form(...),
        title: Optional[str] = Form(None),
        text: Optional[str] = Form(None),
        is_visible: bool = Form(True),
        files: list[UploadFile] = File([]),
        db: AsyncSession = Depends(get_db),
        s3: S3Service = Depends(get_s3),
        redis: Redis = Depends(get_redis)
):
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif"}

    # Validate post content
    if not title and not text and not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Post must contain title, text, or at least one file"
        )

    # Validate board
    board_query = await db.execute(
        select(Board)
        .where(Board.tag == board_tag)
    )
    board = board_query.scalar()
    if not board:
        raise HTTPException(status_code=404, detail="Board not found")
    if board.is_locked:
        raise HTTPException(status_code=403, detail="Board is locked")

    # Validate files before upload
    for file in files:
        content = await file.read(1024)  # Read first 1KB to check type
        await file.seek(0)  # Reset file pointer

        # Check file type
        mime_type = imghdr.what(None, content)
        if not mime_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {file.filename} is not a valid image"
            )

        # Get file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()  # Get position (file size)
        await file.seek(0)  # Reset file pointer

        # Check file size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {file.filename} exceeds maximum size of {MAX_FILE_SIZE / 1024 / 1024}MB"
            )

    uploaded_files = []
    try:
        # Upload files first, outside of the DB transaction
        for file in files:
            try:
                # Generate a unique ID for the file
                file_id = str(uuid.uuid4())

                # Make a copy of the file in memory to avoid streaming issues
                file_copy = BytesIO(await file.read())
                await file.seek(0)

                # Attempt to upload to S3 with specific error handling for Timeweb
                file_key = await s3.upload_file(file)

                if not file_key:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to upload file to S3"
                    )

                # Get metadata - if this fails, we know the upload wasn't successful
                try:
                    metadata = await s3.get_file_metadata(file_key)
                    logger.info(f"Retrieved metadata from S3: {metadata}")
                except Exception as meta_error:
                    logger.error(f"Failed to get metadata for uploaded file: {str(meta_error)}")
                    # Try one more upload approach
                    file_key = await direct_s3_upload(s3, file)
                    # Create fallback metadata since we couldn't get it from S3
                    metadata = {
                        "filename": file.filename,
                        "size": file_copy.getbuffer().nbytes,
                        "content_type": file.content_type or "application/octet-stream",
                    }
                    logger.info(f"Using fallback metadata: {metadata}")

                # Generate thumbnail if image
                thumb_id = None
                try:
                    content_type = metadata.get("content_type", file.content_type)
                    if content_type and content_type.startswith("image/"):
                        await file.seek(0)
                        thumb_id = await generate_thumbnail(file, s3, file_id)
                        if thumb_id:
                            logger.info(f"Thumbnail generated successfully with ID: {thumb_id}")
                        else:
                            logger.warning(f"Thumbnail generation returned None for file: {file.filename}")
                except Exception as thumb_error:
                    logger.error(f"Thumbnail generation failed: {str(thumb_error)}")
                    logger.error(traceback.format_exc())
                    # Continue without thumbnail
                    thumb_id = None

                # Create a metadata dictionary with safe values
                redis_metadata = {
                    "s3_key": file_key,
                    "filename": metadata.get("filename", file.filename or ""),
                    "size": str(metadata.get("size", file_copy.getbuffer().nbytes)),
                    "content_type": metadata.get("content_type", file.content_type or "application/octet-stream"),
                }

                # Only add thumbnail if it was successfully generated
                if thumb_id:
                    redis_metadata["thumbnail_id"] = thumb_id
                    logger.info(f"Adding thumbnail ID {thumb_id} to Redis for file {file_key}")

                # Store metadata in Redis with expiration
                try:
                    # Store metadata for the file key
                    await redis.hset(f"file:{file_key}", mapping=redis_metadata)

                    # Set TTL for Redis keys (30 days)
                    await redis.expire(f"file:{file_key}", 60 * 60 * 24 * 30)

                    logger.info(f"Successfully stored metadata in Redis for file: {file_key}")

                    # Log what's actually stored in Redis for debugging
                    stored_data = await redis.hgetall(f"file:{file_key}")
                    logger.info(f"Data stored in Redis for {file_key}: {stored_data}")
                except Exception as redis_error:
                    logger.error(f"Failed to store metadata in Redis: {str(redis_error)}")
                    logger.error(traceback.format_exc())

                uploaded_files.append(file_key)
                logger.info(f"Successfully processed file {file.filename} with key {file_key}")

            except Exception as file_error:
                logger.error(f"Error processing file {file.filename}: {str(file_error)}")
                # Continue with next file
                continue

        # Once files are uploaded, create post in DB
        try:
            # Create the post without explicit transaction handling
            # to avoid any issues with the transaction state detection
            new_post = await Post.create(
                db=db,
                board_id=board.id,
                title=title,
                text=text,
                file_keys=uploaded_files,  # Ensure this list contains the uploaded file keys
                is_visible=is_visible
            )

            # Ensure changes are committed
            await db.commit()

            # Prepare the image metadata for the response
            images_data = []
            logger.info(f"Preparing response with uploaded files: {uploaded_files}")

            for file_key in uploaded_files:
                try:
                    # Get the data from Redis
                    redis_data = await redis.hgetall(f"file:{file_key}")

                    # Log the Redis data for debugging
                    logger.info(f"Redis data for {file_key}: {redis_data}")

                    # Create a proper ImageMeta object with all required fields
                    image_meta = {
                        "s3_key": file_key,
                        "filename": redis_data.get("filename", "unknown"),
                        "size": int(redis_data.get("size", "0")),
                        "content_type": redis_data.get("content_type", "application/octet-stream"),
                    }

                    # Add thumbnail ID if it exists
                    if "thumbnail_id" in redis_data:
                        image_meta["thumbnail_id"] = redis_data["thumbnail_id"]
                        logger.info(f"Found thumbnail_id in Redis for {file_key}: {redis_data['thumbnail_id']}")

                    # Create ImageMeta object and add to response
                    images_data.append(ImageMeta(**image_meta))

                except Exception as img_error:
                    logger.error(f"Error preparing image metadata for response: {str(img_error)}")
                    logger.error(traceback.format_exc())
                    # Add fallback data to ensure we have an entry
                    images_data.append(ImageMeta(
                        s3_key=file_key,
                        filename="error",
                        content_type="unknown",
                        size=0
                    ))

            # Print the final list of images data
            logger.info(f"Final images_data for response: {images_data}")

            # Ensure image_ids are set in the DB record
            if uploaded_files:
                # Construct a safe SQL query with proper quoting
                quoted_keys = [f"'{key}'" for key in uploaded_files]
                sql_query = sa_text(
                    f"UPDATE post SET image_ids = ARRAY[{','.join(quoted_keys)}] WHERE id = {new_post.id}")
                await db.execute(sql_query)
                await db.commit()
                logger.info(f"Updated image_ids in post {new_post.id} with {uploaded_files}")
            else:
                logger.warning(f"No files to update in post {new_post.id}")

            # Return the response with images data
            return PostResponse(
                id=new_post.id,
                board_id=new_post.board_id,
                title=new_post.title,
                text=new_post.text_,
                images=images_data,
                timestamp=new_post.timestamp,
                parent_id=new_post.parent_id,
                is_visible=new_post.is_visible
            )

        except Exception as db_error:
            # Rollback if there's an error
            await db.rollback()

            logger.error(f"Error creating post in database: {str(db_error)}")
            logger.error(traceback.format_exc())

            # Clean up any uploaded files
            for file_key in uploaded_files:
                try:
                    # Delete the S3 file
                    await s3.delete_file(file_key)

                    # Check if there's a thumbnail in Redis
                    thumb_id = await redis.hget(f"file:{file_key}", "thumbnail_id")
                    if thumb_id:
                        # Delete Redis thumbnail data and metadata
                        await redis.delete(f"thumb:{thumb_id}")
                        await redis.delete(f"thumb_meta:{thumb_id}")

                    # Delete the file metadata
                    await redis.delete(f"file:{file_key}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up file {file_key}: {cleanup_error}")

            # Re-raise the original exception
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating post: {str(db_error)}"
            )

    except Exception as e:
        if isinstance(e, HTTPException):
            # Pass through HTTPExceptions directly
            raise e

        logger.error(f"Unhandled error in create_thread: {str(e)}")
        logger.error(traceback.format_exc())

        # Clean up any uploaded files that weren't already cleaned up
        for file_key in uploaded_files:
            try:
                # Delete the S3 file
                await s3.delete_file(file_key)

                # Check if there's a thumbnail in Redis
                thumb_id = await redis.hget(f"file:{file_key}", "thumbnail_id")
                if thumb_id:
                    # Delete Redis thumbnail data and metadata
                    await redis.delete(f"thumb:{thumb_id}")
                    await redis.delete(f"thumb_meta:{thumb_id}")

                # Delete the file metadata
                await redis.delete(f"file:{file_key}")
            except Exception:
                # Just ignore errors during final cleanup
                pass

        # Re-raise as HTTPException
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


# Alternative implementation for getting boards with direct PostgreSQL access
async def _get_boards_direct_db(sfw_only: bool = False) -> List[Dict[str, Any]]:
    """
    Direct database access bypassing SQLAlchemy ORM completely.
    This is a fallback method when SQLAlchemy has greenlet issues.
    """
    import asyncpg
    from db.database import DATABASE_URL

    # Extract database connection info from SQLAlchemy URL
    # Assumes format: postgresql+asyncpg://user:password@host:port/dbname
    pg_url = DATABASE_URL.replace('postgresql+asyncpg://', 'postgresql://')

    try:
        # Connect directly to PostgreSQL
        conn = await asyncpg.connect(pg_url)

        try:
            # Execute query
            if sfw_only:
                query = "SELECT * FROM board WHERE nsfw = false"
            else:
                query = "SELECT * FROM board"

            rows = await conn.fetch(query)

            # Convert to list of dicts
            result = [dict(row) for row in rows]

            return result
        finally:
            # Always close connection
            await conn.close()
    except Exception as e:
        logger.error(f"Direct DB access error: {str(e)}")
        logger.error(traceback.format_exc())
        return []


@app.get("/boards_direct/")
async def get_boards_direct(
        sfw_only: bool = False,
        redis: Redis = Depends(get_redis)
):
    """
    Alternative endpoint that bypasses SQLAlchemy completely.
    Use this if the regular /boards/ endpoint is failing with greenlet errors.
    """
    try:
        # Try cache first
        cache_key = "cache:boards:sfw" if sfw_only else "cache:boards:all"
        cached = await redis.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass

        # Get directly from database
        boards_data = await _get_boards_direct_db(sfw_only)

        # Cache results
        if boards_data:
            json_data = json.dumps(boards_data)
            await redis.set(cache_key, json_data, ex=CACHE_TTL)

        return boards_data
    except Exception as e:
        logger.error(f"Error in get_boards_direct: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


async def direct_s3_upload(s3_service, file: UploadFile) -> str:
    """
    Alternative direct upload method as a fallback when the main S3 upload fails.
    Uses direct HTTP upload to bypass S3 SDK issues.
    """
    try:
        # Ensure file is at the beginning
        await file.seek(0)

        # Read file content 
        file_content = await file.read()
        content_length = len(file_content)

        # Reset file position
        await file.seek(0)

        # Check if file has content
        if content_length == 0:
            logger.error(f"File {file.filename} appears to be empty")
            raise ValueError("File is empty")

        # Try using the new direct HTTP upload method
        logger.info(f"Attempting direct HTTP upload for file: {file.filename} ({content_length} bytes)")
        file_key = await s3_service.upload_file_direct_http(file)

        if file_key:
            logger.info(f"Direct HTTP upload succeeded for {file.filename}")
            return file_key
        else:
            raise Exception("Direct upload returned no file key")
    except Exception as e:
        logger.error(f"Direct S3 upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"All S3 upload methods failed: {str(e)}"
        )


@app.get("/debug/post/{post_id}")
async def debug_post(
        post_id: int,
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis)
):
    """
    Diagnostic endpoint to debug image handling in posts.
    """
    try:
        # 1. Try to get the post using SQLAlchemy ORM
        query_orm = select(Post).where(Post.id == post_id)
        result_orm = await db.execute(query_orm)
        post_orm = result_orm.scalar_one_or_none()

        # 2. Also try with a direct SQL query
        query_sql = sa_text("""
            SELECT id, title, text as text_, image_ids
            FROM post
            WHERE id = :post_id
        """)
        result_sql = await db.execute(query_sql, {"post_id": post_id})
        post_sql = result_sql.fetchone()

        # Prepare the response
        debug_info = {
            "post_id": post_id,
            "post_exists_orm": post_orm is not None,
            "post_exists_sql": post_sql is not None,
            "orm_image_ids": post_orm.image_ids if post_orm else None,
            "orm_image_ids_type": str(type(post_orm.image_ids)) if post_orm else None,
            "sql_image_ids": post_sql.image_ids if post_sql else None,
            "sql_image_ids_type": str(type(post_sql.image_ids)) if post_sql else None,
        }

        # If we have image_ids, check Redis for their existence
        if post_orm and post_orm.image_ids:
            # Use pipelining for faster Redis operations
            pipe = redis.pipeline()
            for image_id in post_orm.image_ids:
                if image_id:
                    await pipe.exists(f"file:{image_id}")
                    await pipe.hgetall(f"file:{image_id}")

            # Execute the pipeline
            redis_results = await pipe.execute()

            # Process results
            redis_info = []
            i = 0
            while i < len(redis_results):
                exists = redis_results[i]
                metadata = redis_results[i + 1] if i + 1 < len(redis_results) else {}

                image_id = post_orm.image_ids[i // 2]
                full_url = f"{image_id}"

                # Check if we need to prepend media/
                if not image_id.startswith("media/"):
                    media_url = f"media/{image_id}"
                else:
                    media_url = image_id

                redis_info.append({
                    "image_id": image_id,
                    "exists_in_redis": exists,
                    "metadata": metadata,
                    "full_url": full_url,
                    "media_url": media_url
                })
                i += 2

            debug_info["redis_info"] = redis_info

            # Also generate URLs to see if they work
            image_urls = await get_image_urls(post_orm.image_ids)
            debug_info["generated_urls"] = image_urls

            # Show the actual S3 URLs we'd generate
            s3_service = get_s3_service()
            s3_urls = []
            for image_id in post_orm.image_ids:
                if image_id:
                    try:
                        # Ensure media/ prefix
                        if not image_id.startswith("media/"):
                            image_key = f"media/{image_id}"
                        else:
                            image_key = image_id

                        s3_url = await s3_service.generate_presigned_url(image_key)
                        s3_urls.append({"image_id": image_id, "s3_url": s3_url})
                    except Exception as e:
                        s3_urls.append({"image_id": image_id, "error": str(e)})

            debug_info["s3_urls"] = s3_urls

        return debug_info

    except Exception as e:
        logger.error(f"Error in debug_post: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}


async def get_image_data(image_ids: list[str]) -> list[dict]:
    """
    Generates comprehensive image data for the frontend, including both thumbnails and full-size URLs.
    
    Args:
        image_ids: List of image IDs from the database
        
    Returns:
        List of image data dictionaries with URLs and metadata
    """
    if not image_ids or image_ids is None:
        return []

    logger.info(f"Getting image data for IDs: {image_ids}")

    try:
        image_data = []
        redis = await get_redis()

        # Use pipelining for faster Redis operations
        pipe = redis.pipeline()

        # Queue up all operations - check existence and get metadata
        for image_id in image_ids:
            if image_id:
                # Always look up by the original image key
                await pipe.exists(f"file:{image_id}")
                await pipe.hgetall(f"file:{image_id}")

        # Execute the pipeline
        results = await pipe.execute()
        logger.info(f"Redis pipeline results count: {len(results)}")

        # Process results
        i = 0
        while i < len(results):
            exists = results[i]
            metadata = results[i + 1] if i + 1 < len(results) else {}

            image_id = image_ids[i // 2]
            logger.info(f"Processing image ID {image_id}, exists: {exists}, metadata: {metadata}")

            if exists and metadata:
                # Ensure media/ prefix for full image
                if not image_id.startswith("media/"):
                    full_path = f"media/{image_id}"
                else:
                    full_path = image_id

                # Get thumbnail URL if available - using the /thumb/{id} endpoint
                thumbnail_url = None
                if "thumbnail_id" in metadata and metadata["thumbnail_id"]:
                    thumb_id = metadata["thumbnail_id"]
                    logger.info(f"Found thumbnail ID: {thumb_id} for image: {image_id}")

                    # Use the /thumb endpoint for Redis thumbnails
                    thumbnail_url = f"/thumb/{thumb_id}"

                    # Verify this thumbnail exists in Redis for extra safety
                    thumbnail_exists = await redis.exists(f"thumb:{thumb_id}")
                    if not thumbnail_exists:
                        logger.warning(f"Thumbnail {thumb_id} referenced but not found in Redis")
                        thumbnail_url = None
                else:
                    logger.info(f"No thumbnail ID found for image: {image_id}")

                # Build the image info dict
                image_info = {
                    "id": image_id,
                    "url": full_path,  # S3 full image URL
                    "thumbnail": thumbnail_url or full_path,  # Redis thumbnail URL or fallback to full image
                    "filename": metadata.get("filename", ""),
                    "content_type": metadata.get("content_type", ""),
                    "size": metadata.get("size", "0")
                }
                logger.info(f"Created image info: {image_info}")
                image_data.append(image_info)
            else:
                logger.warning(f"Image {image_id} not found in Redis or has no metadata")

                # Still create a basic entry even if metadata is missing
                if exists or image_id:
                    # Use the image ID as a fallback
                    full_path = f"media/{image_id}" if not image_id.startswith("media/") else image_id
                    fallback_info = {
                        "id": image_id,
                        "url": full_path,
                        "thumbnail": full_path,  # Use full image as thumbnail too
                        "filename": "Unknown",
                        "content_type": "image/jpeg",  # Assume a generic type
                        "size": "0"
                    }
                    logger.info(f"Created fallback image info for {image_id}: {fallback_info}")
                    image_data.append(fallback_info)

            i += 2

        logger.info(f"Retrieved data for {len(image_data)} images")
        return image_data

    except Exception as e:
        logger.error(f"Error in get_image_data: {str(e)}")
        logger.error(traceback.format_exc())

        # Create a minimal fallback for each image ID
        fallback_data = []
        for image_id in image_ids:
            if image_id:
                # Ensure it has the media/ prefix
                full_path = f"media/{image_id}" if not image_id.startswith("media/") else image_id
                fallback_data.append({
                    "id": image_id,
                    "url": full_path,
                    "thumbnail": full_path,  # Use full image as thumbnail
                    "filename": "Unknown",
                    "content_type": "image/jpeg",
                    "size": "0"
                })

        logger.info(f"Created {len(fallback_data)} fallback entries in error handler")
        return fallback_data


@app.get("/debug/thumbnail/{thumb_id}")
async def debug_thumbnail(
        thumb_id: str,
        redis: Redis = Depends(get_redis)
):
    """
    Diagnostic endpoint to check thumbnail data in Redis.
    This now checks both the old S3-based thumbnails (thumbnail_key) and new Redis-based thumbnails.
    """
    try:
        # Check if it's a new Redis-stored thumbnail
        redis_thumb_exists = await redis.exists(f"thumb:{thumb_id}")
        redis_thumb_meta = await redis.hgetall(f"thumb_meta:{thumb_id}")

        # Also check if it's referenced from a file (old format with S3 thumbnail)
        # Find files that reference this thumbnail
        file_keys = []
        async for key in redis.scan_iter("file:*"):
            file_meta = await redis.hgetall(key)
            # Check if this file references our thumbnail
            if "thumbnail_id" in file_meta and file_meta["thumbnail_id"] == thumb_id:
                file_keys.append(key.decode('utf-8').replace("file:", ""))

        # Prepare the response
        response = {
            "thumbnail_id": thumb_id,
            "exists_in_redis_storage": redis_thumb_exists,
            "thumbnail_metadata": redis_thumb_meta,
            "referenced_by_files": file_keys,
            "direct_url": f"/thumb/{thumb_id}" if redis_thumb_exists else None,
        }

        # Add thumbnail size if it exists
        if redis_thumb_exists:
            thumb_data = await redis.get(f"thumb:{thumb_id}")
            response["thumbnail_size_bytes"] = len(thumb_data) if thumb_data else 0
            response["content_type"] = redis_thumb_meta.get("content_type", "image/jpeg")

        return response

    except Exception as e:
        logger.error(f"Error in debug_thumbnail: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.get("/thumb/{thumb_id}")
async def get_thumbnail(
        thumb_id: str,
        redis: Redis = Depends(get_redis)
):
    """
    Serve thumbnail images directly from Redis.
    """
    try:
        # Check if thumbnail exists
        thumb_exists = await redis.exists(f"thumb:{thumb_id}")
        if not thumb_exists:
            logger.warning(f"Thumbnail {thumb_id} not found in Redis")
            raise HTTPException(status_code=404, detail="Thumbnail not found")

        # Get thumbnail metadata
        metadata = await redis.hgetall(f"thumb_meta:{thumb_id}")

        # Get the actual thumbnail binary data
        thumb_data = await redis.get(f"thumb:{thumb_id}")

        # Determine content type from metadata, default to image/jpeg
        content_type = metadata.get("content_type", "image/jpeg")

        # Reset expiration on access (keeps frequently accessed thumbnails alive)
        await redis.expire(f"thumb:{thumb_id}", 60 * 60 * 24 * 30)  # 30 days
        await redis.expire(f"thumb_meta:{thumb_id}", 60 * 60 * 24 * 30)

        # Return the binary data with proper content type
        return Response(
            content=thumb_data,
            media_type=content_type
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error serving thumbnail {thumb_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving thumbnail: {str(e)}")


@app.get("/thread/{op_id}", response_model=dict)
async def get_thread_by_op(
        op_id: int,
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis)
):
    try:
        # Try to get from cache first
        cache_key = f"cache:thread:{op_id}"
        cached = await redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # Get the OP post
        op_query = select(Post).where(
            Post.id == op_id,
            Post.parent_id == 0,  # Ensure it's an OP post
            Post.is_visible == True
        )
        op_result = await db.execute(op_query)
        op_post = op_result.scalar_one_or_none()

        if not op_post:
            raise HTTPException(
                status_code=404,
                detail="Thread not found or is not visible"
            )

        # Get all replies
        replies_query = select(Post).where(
            Post.parent_id == op_id,
            Post.is_visible == True
        ).order_by(Post.timestamp)
        replies_result = await db.execute(replies_query)
        replies = replies_result.scalars().all()

        # Get image data for OP and replies
        all_image_ids = op_post.image_ids.copy()
        for reply in replies:
            all_image_ids.extend(reply.image_ids)

        # Get image data
        image_data = await get_image_data(all_image_ids)

        # Construct response
        response = {
            "op": {
                **op_post.to_dict(),
                "images": [img for img in image_data if img["id"] in op_post.image_ids]
            },
            "replies": [
                {
                    **reply.to_dict(),
                    "images": [img for img in image_data if img["id"] in reply.image_ids]
                }
                for reply in replies
            ]
        }

        # Cache the response
        await redis.set(cache_key, json.dumps(response), ex=CACHE_TTL)

        return response

    except Exception as e:
        logger.error(f"Error in get_thread_by_op: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving thread: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        app,  # Your FastAPI app
        host="0.0.0.0",
        port=8000,
        log_config=None,
        log_level=None,
    )
