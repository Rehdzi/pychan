from fastapi import FastAPI, Depends, HTTPException, status
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from sqlalchemy.orm import selectinload, aliased, joinedload
from fastapi.middleware.cors import CORSMiddleware
from db.models import *
from db.database import get_db
from util.schemas import *

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

## @app.get("/thread/{post_id}")


async def get_board_ops_with_replies(async_session: AsyncSession, board_id: int):
    FirstReply = aliased(Post)
    LastReply = aliased(Post)

    # Подзапрос для агрегации ответов
    replies_subquery = (
        select(
            Post.parent_id,
            func.min(Post.id).label('first_reply_id'),
            func.max(Post.id).label('last_reply_id')
        )
        .where(Post.parent_id != 0)
        .group_by(Post.parent_id)
        .subquery()
    )

    # Основной запрос с джойнами
    stmt = (
        select(
            Post,
            FirstReply,
            LastReply
        )
        .outerjoin(
            replies_subquery,
            Post.id == replies_subquery.c.parent_id
        )
        .outerjoin(
            FirstReply,
            FirstReply.id == replies_subquery.c.first_reply_id
        )
        .outerjoin(
            LastReply,
            LastReply.id == replies_subquery.c.last_reply_id
        )
        .where(
            Post.parent_id == 0,
            Post.board_id == board_id
        )
        .options(
            ## joinedload(Post.board),
            ## joinedload(Post.image_ids)  # Если есть связь с изображениями
        )
        .order_by(Post.timestamp.desc())
    )

    result = await async_session.execute(stmt)
    return result.all()

@app.get("/{board_tag}/ops", response_model=BoardOpsResponse)
async def get_board_operations(
    board_tag: str,
    session: AsyncSession = Depends(get_db)
):
    # Получение доски
    board_query = await session.execute(
        select(Board)
        .where(Board.tag == board_tag)
        .options(joinedload(Board.category)))
    board = board_query.scalar()

    if not board:
        raise HTTPException(status_code=404, detail="Board not found")
    if not board.is_visible:
        raise HTTPException(status_code=403, detail="Board is hidden")

    # Получение OP-постов
    ops_data = await get_board_ops_with_replies(session, board.id)

    # Форматирование ответа
    formatted_ops = []
    for op, first_reply, last_reply in ops_data:
        formatted_ops.append(BoardOpResponse(
            op={
                "id": op.id,
                "title": op.title,
                "text": op.text_,
                "timestamp": op.timestamp,
                "image_ids": [img.id for img in op.image_ids],
                "board": {"tag": board.tag, "name": board.name}
            },
            first_reply=PostReply(
                id=first_reply.id if first_reply else None,
                text=first_reply.text_ if first_reply else None,
                timestamp=first_reply.timestamp if first_reply else None
            ) if first_reply else None,
            last_reply=PostReply(
                id=last_reply.id if last_reply else None,
                text=last_reply.text_ if last_reply else None,
                timestamp=last_reply.timestamp if last_reply else None
            ) if last_reply else None,
            replies_count=len(op.child_ids) if op.child_ids else 0
        ))

    return BoardOpsResponse(
        board_info={
            "tag": board.tag,
            "name": board.name,
            "description": board.description,
            "category": board.category.name if board.category else None
        },
        ops=formatted_ops
    )


@app.post("/new_thread/", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def new_thread(post_data: PostCreate,
                     db: AsyncSession = Depends(get_db)):
    # Получаем доску по тегу
    board_query = await db.execute(
        select(Board)
        .where(Board.tag == post_data.board_tag)
    )
    board = board_query.scalar()

    if not board:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Board not found"
        )

    if not board.is_visible or board.is_locked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Board is locked or hidden"
        )

    # Создаем новый пост
    new_post = Post(
        board_id=board.id,
        title=post_data.title,
        text_=post_data.text,
        image_ids=post_data.image_ids,
        parent_id=0,  # Указываем что это OP-пост
        timestamp=datetime.utcnow(),
        is_visible=post_data.is_visible,
        child_ids=[]
    )

    try:
        db.add(new_post)
        await db.commit()
        await db.refresh(new_post)
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating post: {str(e)}"
        )

    return {
        "id": new_post.id,
        "board_id": new_post.board_id,
        "title": new_post.title,
        "text": new_post.text_,
        "image_ids": new_post.image_ids,
        "timestamp": new_post.timestamp,
        "parent_id": new_post.parent_id,
        "is_visible": new_post.is_visible
    }