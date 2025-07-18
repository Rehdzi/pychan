from typing import Type

from pychan.schemas.post import PostAddSchema
from pychan.util.repository import AbstractRepository


class PostService:
    """Service for posts."""
    def __init__(self, posts_repo: AbstractRepository):
        self.posts_repo = posts_repo()

    async def add_post(self, post: PostAddSchema):
        posts_dict = post.model_dump()
        post_id = await self.posts_repo.add_one(posts_dict)
        return post_id

    async def get_latest_posts(self, num: int, nsfw: bool):
        posts = await self.posts_repo.get_latest(num, nsfw)
        return posts

    async def get_posts_by_board(self, board_id: int):
        posts = await self.posts_repo.get_posts_by_board(board_id)
        return posts


