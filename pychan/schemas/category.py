from pydantic import BaseModel


class CategoryAddSchema(BaseModel):
    name: str
    is_visible: bool
    is_nsfw: bool

class CategorySchema(CategoryAddSchema):
    id: int