from sqlalchemy import select

from pychan.db.database import new_session
from pychan.db.models import Post, Board
from pychan.util.repository import SQLAlchemyRepository


class PostsRepository(SQLAlchemyRepository):
    model = Post

    async def get_posts_by_board(self, board_id: int):
        async with new_session() as session:
            stmt = (select(self.model)
                 .join(Board)
                 .where(Board.id == board_id)
                 .order_by(Post.timestamp.desc())
                 )
            result = await session.execute(stmt)
            posts = result.scalars().all()
            return posts

    async def get_latest(self, num: int, nsfw: bool):
        async with new_session() as session:
            stmt = (select(self.model)
                 .join(Board)
                 .where(Board.nsfw == nsfw)
                 .where(self.model.parent_id.is_(None))
                 .order_by(Post.timestamp.desc())
                 .limit(num)
                 )

        result = await session.execute(stmt)
        posts = result.scalars().all()
        return posts
