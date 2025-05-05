from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List

class PostReply(BaseModel):
    id: Optional[int] = None
    text: Optional[str] = None
    timestamp: Optional[datetime] = None

class BoardOpResponse(BaseModel):
    op: dict
    first_reply: Optional[PostReply] = None
    last_reply: Optional[PostReply] = None
    replies_count: int

class BoardOpsResponse(BaseModel):
    board_info: dict
    ops: List[BoardOpResponse]

class PostCreate(BaseModel):
    board_tag: str = Field(..., min_length=1, max_length=5)
    title: Optional[str] = Field(None, max_length=150)
    text: Optional[str] = Field(None, max_length=450)
    image_ids: Optional[List[int]] = Field(default_factory=list)
    is_visible: Optional[bool] = True


class PostResponse(BaseModel):
    id: int
    board_id: int
    title: Optional[str]
    text: Optional[str]
    image_ids: List[int]
    timestamp: datetime
    parent_id: Optional[int]
    is_visible: bool