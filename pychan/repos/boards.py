from sqlalchemy import select

from pychan.db.database import new_session
from pychan.db.models import Board
from pychan.util.repository import SQLAlchemyRepository


class BoardsRepository(SQLAlchemyRepository):
    model = Board

    async def find_by_tag(self, value: str):
        async with new_session() as session:
            stmt = select(self.model).where(self.model.tag == value)
            res = await session.execute(stmt)
            return res.scalar_one()

    async def find_with_filters(self, nsfw: bool, is_visible: bool):
        async with new_session() as session:
            stmt = select(self.model).where(
                self.model.nsfw == nsfw,
                self.model.is_visible == is_visible
            )
            res = await session.execute(stmt)
            res = [row.to_read_model() for row in res.scalars().all()]
            await session.commit()
            return res
