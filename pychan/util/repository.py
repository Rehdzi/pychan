from abc import ABC, abstractmethod

from sqlalchemy import insert, select

from pychan.db.database import new_session


class AbstractRepository(ABC):
    """Abstract base class for repository pattern."""
    @abstractmethod
    async def add_one(self):
        raise NotImplementedError

    @abstractmethod
    async def find_all(self):
        raise NotImplementedError


class SQLAlchemyRepository(AbstractRepository):
    """SQLAlchemy implementation of the repository pattern."""
    model = None

    async def add_one(self, data: dict) -> int:
        async with new_session() as session:
            stmt = insert(self.model).values(**data).returning(self.model.id)
            res = await session.execute(stmt)
            await session.commit()
            return res.scalar_one()

    async def find_all(self):
        async with new_session() as session:
            stmt = select(self.model)
            res = await session.execute(stmt)
            res = [row[0].to_read_model() for row in res.all()]
            await session.commit()
            return res
