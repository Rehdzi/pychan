from pydantic import BaseModel
from datetime import datetime


class PostAddSchema(BaseModel):
    board_id: int
    title: str | None
    text: str
    image_ids: list[str] | None
    is_visible: bool
    parent_id: int | None


class PostSchema(PostAddSchema):
    id: int
    timestamp: datetime


