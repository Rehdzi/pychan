import traceback

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import json
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from pychan.db.database import get_db, SessionDep
from pychan.db.models import Category, Board, Post
from pychan.repos.boards import BoardsRepository
from pychan.repos.posts import PostsRepository
from pychan.schemas.board import BoardSchema, BoardAddSchema
from pychan.schemas.post import PostAddSchema
from pychan.services.posts import PostService
from pychan.util.redis_config import get_redis

router = APIRouter(
    prefix="/posts",
    tags=["Posts"],
)

@router.get("/latest")
async def get_latest_posts(num: int, nsfw: bool):
    res = await PostsRepository().get_latest(num, nsfw)
    return res

@router.post("/new")
async def new_post(
        post: PostAddSchema
):
    post_id = await PostService(PostsRepository).add_post(post)
    return {"post_id": post_id}
