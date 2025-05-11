import uuid
from urllib.parse import quote

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from datetime import datetime

from fastapi.openapi.models import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from sqlalchemy.orm import selectinload, aliased, joinedload
from fastapi.middleware.cors import CORSMiddleware
from db.models import *
from db.database import get_db
from util.schemas import *
from util.redis_config import redis_client

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

async def get_image_urls(image_ids: list[str]) -> list[str]:
    """Генерирует URL для изображений, существующих в Redis"""
    urls = []
    for image_id in image_ids:
        if redis_client.exists(f"image:{image_id}"):
            urls.append(f"/images/{image_id}")
    return urls

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
        # Получаем URL для OP-поста
        op_image_urls = await get_image_urls(op.image_ids)

        # Формируем ответ с URL вместо ID
        op_data = {
            "id": op.id,
            "title": op.title,
            "text": op.text_,
            "timestamp": op.timestamp.isoformat(),
            "image_urls": op_image_urls,
            "board": {"tag": board.tag, "name": board.name}
        }

        # Обработка первого ответа
        first_reply_data = None
        if first_reply:
            first_images = await get_image_urls(first_reply.image_ids or [])
            first_reply_data = PostReply(
                id=first_reply.id,
                text=first_reply.text_,
                image_urls=first_images,
                timestamp=first_reply.timestamp
            )

        # Обработка последнего ответа
        last_reply_data = None
        if last_reply and last_reply.id != first_reply.id:
            last_images = await get_image_urls(last_reply.image_ids or [])
            last_reply_data = PostReply(
                id=last_reply.id,
                text=last_reply.text_,
                image_urls=last_images,
                timestamp=last_reply.timestamp
            )

        formatted_ops.append(BoardOpResponse(
            op=op_data,
            first_reply=first_reply_data,
            last_reply=last_reply_data,
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


@app.get("/images/{image_id}")
async def get_image(image_id: str):
    if not redis_client.exists(f"image:{image_id}"):
        raise HTTPException(status_code=404, detail="Image not found")

    # Получаем метаданные
    meta = redis_client.hgetall(f"image:{image_id}")
    content_type = meta.get(b'content_type', b'application/octet-stream').decode()
    filename = meta.get(b'filename', b'image').decode()

    # Получаем данные изображения
    image_data = redis_client.get(f"image:{image_id}:data")

    # Кодируем имя файла для заголовка
    filename_encoded = quote(filename, safe='')

    return Response(
        content=image_data,
        media_type=content_type,
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{filename_encoded}"
        }
    )

@app.post("/new_thread/", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def new_thread(
        board_tag: str = Form(...),
        title: Optional[str] = Form(None),
        text: Optional[str] = Form(None),
        is_visible: Optional[bool] = Form(True),
        files: list[UploadFile] = File([]),
        db: AsyncSession = Depends(get_db)
):
    # Получаем доску по тегу
    board_query = await db.execute(
        select(Board)
        .where(Board.tag == board_tag)
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

    # Обрабатываем изображения
    image_ids = []
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is not an image"
            )

        try:
            contents = await file.read()
            image_id = str(uuid.uuid4())

            # Сохраняем в Redis
            pipe = redis_client.pipeline()
            pipe.hset(f"image:{image_id}", mapping={
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(contents)
            })
            pipe.set(f"image:{image_id}:data", contents)
            pipe.expire(f"image:{image_id}", 3600 * 24 * 7)  # TTL 1 неделя
            pipe.execute()

            # with open(file.filename, "wb") as f:
            #     f.write(file.file.read())

            image_ids.append(image_id)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error uploading image {file.filename}: {str(e)}"
            )

    # Создаем новый пост с image_ids
    new_post = Post(
        board_id=board.id,
        title=title,
        text_=text,
        image_ids=image_ids,  # Используем ID из Redis
        parent_id=0,
        timestamp=datetime.utcnow(),
        is_visible=is_visible,
        child_ids=[]
    )

    try:
        db.add(new_post)
        await db.commit()
        await db.refresh(new_post)
    except Exception as e:
        # Откатываем загрузку изображений при ошибке
        for image_id in image_ids:
            redis_client.delete(f"image:{image_id}")
            redis_client.delete(f"image:{image_id}:data")

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
        "image_urls": new_post.image_ids,
        "timestamp": new_post.timestamp,
        "parent_id": new_post.parent_id,
        "is_visible": new_post.is_visible
    }