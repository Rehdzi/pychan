from typing import List, Optional

from sqlalchemy import ARRAY, BigInteger, Boolean, DateTime, ForeignKeyConstraint, Identity, Integer, \
    PrimaryKeyConstraint, String, UniqueConstraint, text, ForeignKey, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncAttrs
import datetime

from pychan.schemas.board import BoardSchema
from pychan.schemas.post import PostSchema


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
    category_id: Mapped[int] = mapped_column(ForeignKey("category.id"), server_default=text('1'))
    nsfw: Mapped[bool] = mapped_column(Boolean, server_default=text('false'))
    is_visible: Mapped[bool] = mapped_column(Boolean, server_default=text('true'))
    is_locked: Mapped[bool] = mapped_column(Boolean, server_default=text('false'))
    name: Mapped[str] = mapped_column(String(50))
    description: Mapped[Optional[str]] = mapped_column(String(150), server_default=text("''::character varying"))

    category: Mapped['Category'] = relationship('Category', back_populates='board')
    post: Mapped[List['Post']] = relationship('Post', back_populates='board')

    def to_read_model(self) -> BoardSchema:
        return BoardSchema(
            id=self.id,
            tag=self.tag,
            category_id=self.category_id,
            nsfw=self.nsfw,
            is_visible=self.is_visible,
            is_locked=self.is_locked,
            name=self.name,
            description=self.description or ""
        )


class Post(Base):
    __tablename__ = 'post'
    __table_args__ = (
        ForeignKeyConstraint(['board_id'], ['board.id'], name='post_board_id_fkey'),
        PrimaryKeyConstraint('id', name='post_pkey')
    )

    id: Mapped[int] = mapped_column(Integer, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=2147483647, cycle=False, cache=1), primary_key=True)
    board_id: Mapped[int] = mapped_column(ForeignKey("board.id"))
    image_ids: Mapped[list] = mapped_column(ARRAY(String()), server_default=text("'{}'::character varying[]"))
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, index=True, server_default=text('current_timestamp'))
    is_visible: Mapped[bool] = mapped_column(Boolean, server_default=text('true'))
    title: Mapped[Optional[str]] = mapped_column(String(150), server_default=text("''::character varying"))
    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("post.id"), nullable=True, index=True
    )
    text: Mapped[Optional[str]] = mapped_column('text', String(450))

    # Отношение к родительскому посту
    parent: Mapped[Optional["Post"]] = relationship(
        "Post", remote_side="Post.id", backref="children", uselist=False
    )

    board: Mapped['Board'] = relationship('Board', back_populates='post')

    def to_read_model(self) -> PostSchema:
        return PostSchema(
            id=self.id,
            board_id=self.board_id,
            title=self.title,
            text=self.text,
            image_ids=self.image_ids,
            is_visible=self.is_visible,
            parent_id=self.parent_id,
            timestamp=self.timestamp if hasattr(self, 'timestamp') else None
        )
        