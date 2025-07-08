from pydantic import BaseModel, Field


class BoardAddSchema(BaseModel):
    tag: str = Field(max_length=5)
    category_id: int = Field(gt=0)
    nsfw: bool
    is_visible: bool
    is_locked: bool
    name: str = Field(max_length=150)
    description: str = Field(max_length=150)


class BoardSchema(BoardAddSchema):
    id: int = Field(gt=0)
