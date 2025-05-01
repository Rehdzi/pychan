from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from fastapi.middleware.cors import CORSMiddleware
from db.models import *
from db.database import get_db

app = FastAPI()

origins = [
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# @app.post("/users/")
# async def create_user(name: str, email: str, db: AsyncSession = Depends(get_db)):
#     new_user = User(name=name, email=email)
#     db.add(new_user)
#     await db.commit()
#     return new_user

@app.get("/categories/")
async def get_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Category))
    users = result.scalars().all()
    return users

@app.get("/boardlist")
async def get_categories_with_boards(db: AsyncSession = Depends(get_db)):
    # Выбираем все категории с предзагрузкой досок
    query = select(Category).options(selectinload(Category.boards)).order_by(Category.id)

    result = await db.execute(query)
    categories = result.scalars().all()

    return [
        {
            "id": cat.id,
            "name": cat.name,
            "is_visible": cat.is_visible,
            "is_nsfw": cat.is_nsfw,
            "boards": [
                {
                    "id": board.id,
                    "tag": board.tag,
                    "name": board.name,
                    "description": board.description,
                    "nsfw": board.nsfw,
                    "is_visible": board.is_visible,
                    "is_locked": board.is_locked
                }
                for board in cat.boards
                if board.is_visible  # Фильтрация невидимых досок
            ]
        }
        for cat in categories
        if cat.is_visible  # Фильтрация невидимых категорий
    ]

@app.get("/boards/")
async def get_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Board))
    boards = result.scalars().all()
    return boards

@app.get("/sfw_boards/")
async def get_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Board)
                              .where(Board.nsfw == False))
    boards = result.scalars().all()
    return boards


@app.get("/{tag}")
async def get_board(tag: str,
                    db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Board)
                              .where(Board.tag == tag))
    board = result.scalars().first()
    if board is None:
        raise HTTPException(status_code=404, detail="Board not found")
    return board

#SELECT * FROM post
# JOIN board ON post.board_id = board.id
# WHERE board.nsfw = false
# ORDER BY timestamp DESC LIMIT 8
@app.get("/boards/latest/")
async def get_users(db: AsyncSession = Depends(get_db)):
    query = (select(Post)
             .join(Board)
             .where(Board.nsfw == False)
             .where(Post.parent_id == 0)
             .order_by(Post.timestamp.desc())
             .limit(8)
             )

    result = await db.execute(query)
    boards = result.scalars().all()
    return boards

@app.get("/boards/{board_id}/posts/")
async def get_posts_by_board(
    board_id: int,
    db: AsyncSession = Depends(get_db),
):
    board = await db.get(Board, board_id)
    if not board:
        raise HTTPException(status_code=404, detail="Board not found")

    query = (
        select(Post)
        .where(Post.board_id == board_id)
        .order_by(Post.timestamp.desc())
    )

    result = await db.execute(query)
    posts = result.scalars().all()

    return posts