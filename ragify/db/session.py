"""Engine/session 工厂。默认数据库文件在 vectorstore/ragify.db（跟现有的
vectorstore/kbs.json 放在同一个目录，方便理解——都是"knowledge base 相关的
持久化状态"）。

get_engine() 按 database_url 缓存 engine（同一个 URL 只创建一次，连接池能
被复用）；测试用不同的 database_url 传进来就会拿到全新的、隔离的 engine。
"""

import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DB_PATH = Path("vectorstore") / "ragify.db"


def default_database_url() -> str:
    return os.environ.get("RAGIFY_DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")


@lru_cache(maxsize=8)
def get_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite:///") and database_url != "sqlite:///:memory:":
        db_file = database_url[len("sqlite:///"):]
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(database_url, connect_args={"check_same_thread": False})


def get_session(database_url: str | None = None) -> Session:
    url = database_url or default_database_url()
    engine = get_engine(url)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return factory()
