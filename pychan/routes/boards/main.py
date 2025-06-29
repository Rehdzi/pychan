import traceback

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import json
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from pychan.db.database import get_db
from pychan.db.models import Category, Board, Post
from pychan.util.redis_config import get_redis

router = APIRouter(
    prefix="/boards",
    tags=["boards"],
)

@router.get("/list")
async def get_categories_with_boards(
        ttl: int,
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
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
        await redis.set("cache:boardlist", json.dumps(response), ex=ttl)

        return response
    except Exception as e:
        logger.error(f"Error in get_categories_with_boards: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/{tag}")
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

@router.get("/latest")
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

