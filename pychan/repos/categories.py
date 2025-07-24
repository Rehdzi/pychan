from pychan.db.models import Category
from pychan.util.repository import SQLAlchemyRepository


class CategoriesRepository(SQLAlchemyRepository):
    model = Category
    