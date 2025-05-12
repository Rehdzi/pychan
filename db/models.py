from typing import List, Optional

from sqlalchemy import ARRAY, BigInteger, Boolean, DateTime, ForeignKeyConstraint, Identity, Integer, PrimaryKeyConstraint, String, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncAttrs
import datetime

class Base(AsyncAttrs, DeclarativeBase):
    def to_dict(self):
        """Convert model to dictionary for JSON serialization"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Category(Base):
    __tablename__ = 'category'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='untitled_table_pkey'),
        UniqueConstraint('name', name='untitled_table_name_key')
    )

    id: Mapped[int] = mapped_column(Integer, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=2147483647, cycle=False, cache=1), primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    is_visible: Mapped[bool] = mapped_column(Boolean, server_default=text('true'))
    is_nsfw: Mapped[bool] = mapped_column(Boolean, server_default=text('false'))

    board: Mapped[List['Board']] = relationship('Board', back_populates='category')
    boards = relationship("Board", back_populates="category")

    def to_dict(self):
        result = super().to_dict()
        # Don't include relationship data by default
        return result


class Image(Base):
    __tablename__ = 'image'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='image_pkey'),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    url: Mapped[str] = mapped_column(String)
    squared: Mapped[str] = mapped_column(String)
    filename: Mapped[str] = mapped_column(String)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    filesize: Mapped[int] = mapped_column(BigInteger)
    alt: Mapped[str] = mapped_column(String)


class Users(Base):
    __tablename__ = 'users'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='users_pkey'),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String)
    country: Mapped[Optional[str]] = mapped_column(String)
    ip: Mapped[Optional[str]] = mapped_column(String)
    hash: Mapped[Optional[str]] = mapped_column(String)
    admin: Mapped[Optional[bool]] = mapped_column(Boolean, server_default=text('false'))


class Board(Base):
    __tablename__ = 'board'
    __table_args__ = (
        ForeignKeyConstraint(['category_id'], ['category.id'], name='board_category_id_fkey'),
        PrimaryKeyConstraint('id', name='board_pkey'),
        UniqueConstraint('name', name='board_name_key'),
        UniqueConstraint('tag', name='board_tag_key')
    )

    id: Mapped[int] = mapped_column(Integer, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=2147483647, cycle=False, cache=1), primary_key=True)
    tag: Mapped[str] = mapped_column(String(5))
    category_id: Mapped[int] = mapped_column(Integer, server_default=text('1'))
    nsfw: Mapped[bool] = mapped_column(Boolean, server_default=text('false'))
    is_visible: Mapped[bool] = mapped_column(Boolean, server_default=text('true'))
    is_locked: Mapped[bool] = mapped_column(Boolean, server_default=text('false'))
    name: Mapped[str] = mapped_column(String(50))
    description: Mapped[Optional[str]] = mapped_column(String(150), server_default=text("''::character varying"))

    category: Mapped['Category'] = relationship('Category', back_populates='board')
    post: Mapped[List['Post']] = relationship('Post', back_populates='board')
    category = relationship("Category", back_populates="boards")

    def to_dict(self):
        result = super().to_dict()
        # Include the category name if available
        if self.category:
            result['category_name'] = self.category.name
        return result


class Post(Base):
    __tablename__ = 'post'
    __table_args__ = (
        ForeignKeyConstraint(['board_id'], ['board.id'], name='post_board_id_fkey'),
        PrimaryKeyConstraint('id', name='post_pkey')
    )

    id: Mapped[int] = mapped_column(Integer, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=2147483647, cycle=False, cache=1), primary_key=True)
    board_id: Mapped[int] = mapped_column(BigInteger)
    image_ids: Mapped[list] = mapped_column(ARRAY(String()), server_default=text("'{}'::character varying[]"))
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime)
    is_visible: Mapped[bool] = mapped_column(Boolean, server_default=text('true'))
    title: Mapped[Optional[str]] = mapped_column(String(150), server_default=text("''::character varying"))
    parent_id: Mapped[Optional[int]] = mapped_column(BigInteger, server_default=text("'0'::bigint"))
    text_: Mapped[Optional[str]] = mapped_column('text', String(450))
    child_ids: Mapped[Optional[list]] = mapped_column(ARRAY(Integer()), server_default=text("'{}'::integer[]"))

    board: Mapped['Board'] = relationship('Board', back_populates='post')

    def to_dict(self):
        result = super().to_dict()
        # Convert datetime to ISO format string
        if self.timestamp:
            result['timestamp'] = self.timestamp.isoformat()
        # Change text_ key to text for consistency
        if 'text_' in result:
            result['text'] = result.pop('text_')
        # Ensure image_ids is a list
        if 'image_ids' in result and result['image_ids'] is None:
            result['image_ids'] = []
        return result

    @classmethod
    async def create(cls, db, board_id, title=None, text=None, file_keys=None, is_visible=True, parent_id=0):
        """
        Create a new post
        
        Args:
            db: Database session
            board_id: Board ID
            title: Post title
            text: Post text
            file_keys: List of S3 file keys
            is_visible: Whether the post is visible
            parent_id: Parent post ID (0 for original posts)
            
        Returns:
            Post: The created post
        """
        # Ensure file_keys is a list, even if empty
        if file_keys is None:
            file_keys = []
            
        # Log the file keys for debugging
        print(f"Creating post with file keys: {file_keys}")
        
        new_post = cls(
            board_id=board_id,
            title=title,
            text_=text,
            image_ids=file_keys,  # Assign file_keys to image_ids
            timestamp=datetime.datetime.now(),
            is_visible=is_visible,
            parent_id=parent_id
        )
        
        db.add(new_post)
        await db.flush()  # To get the ID
        
        # If this is a reply, update parent's child_ids
        if parent_id != 0:
            parent = await db.get(cls, parent_id)
            if parent:
                if parent.child_ids:
                    parent.child_ids.append(new_post.id)
                else:
                    parent.child_ids = [new_post.id]
        
        return new_post
