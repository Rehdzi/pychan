from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List

class PostReply(BaseModel):
    id: Optional[int] = None
    text: Optional[str] = None
    image_urls: List[str] = Field(default_factory=list)
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
    image_urls: Optional[List[str]] = Field(default_factory=list)
    is_visible: Optional[bool] = True

class ImageCreate(BaseModel):
    filename: str
    content_type: str

class ImageResponse(BaseModel):
    id: str
    filename: str
    size: int

class ImageMeta(BaseModel):
    s3_key: str
    filename: str
    content_type: str
    size: int
    thumbnail_key: Optional[str] = None

class PostResponse(BaseModel):
    id: int
    board_id: int
    title: Optional[str]
    text: Optional[str]
    images: List[ImageMeta]
    timestamp: datetime
    parent_id: Optional[int]
    is_visible: bool
