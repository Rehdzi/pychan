from datetime import datetime
from pydantic import BaseModel
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