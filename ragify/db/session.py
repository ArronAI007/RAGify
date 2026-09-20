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
    # .env 里未填值时这个变量是空字符串而不是"不存在"，os.environ.get(key, default)
    # 只在 key 缺失时才会用 default，空字符串会被当成"已设置"直接返回，导致引擎
    # 拿一个空 URL 去连接——用 or 让空字符串也走默认值。
    return os.environ.get("RAGIFY_DATABASE_URL") or f"sqlite:///{DEFAULT_DB_PATH}"


@lru_cache(maxsize=8)
def get_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite:///") and database_url != "sqlite:///:memory:":
        db_file = database_url[len("sqlite:///"):]
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite:") else {}
    return create_engine(database_url, connect_args=connect_args)


def get_session(database_url: str | None = None) -> Session:
    url = database_url or default_database_url()
    engine = get_engine(url)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return factory()
