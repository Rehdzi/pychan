from fastapi import APIRouter

from pychan.repos.categories import CategoriesRepository
from pychan.schemas.category import CategoryAddSchema


router = APIRouter(
    prefix="/categories",
    tags=["Categories"],
)

@router.get("/list")
async def get_categories():
    res = await CategoriesRepository().find_all()
    return res

@router.post("/add")
async def add_category(category: CategoryAddSchema):
    res = await CategoriesRepository().add_one(category.model_dump())
    return res