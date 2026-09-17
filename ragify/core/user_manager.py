"""用户账户管理。跟 ragify/core/kb_manager.py 的 KBManager 同一个模式：
DB 驱动、session-per-call、构造函数接受可选的 database_url 用于测试隔离。
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db.models import UserRow
from ..db.session import get_session
from .security import hash_password, verify_password


@dataclass
class User:
    id: str
    email: str
    name: str
    created_at: str


class UserManager:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url

    def _session(self) -> Session:
        return get_session(self.database_url)

    def create(self, email: str, password: str, name: str) -> User:
        email = email.strip().lower()
        if not email:
            raise ValueError("邮箱不能为空")
        if len(password) < 8:
            raise ValueError("密码至少需要 8 位")
        name = name.strip()
        if not name:
            raise ValueError("姓名不能为空")

        with self._session() as session:
            existing = session.query(UserRow).filter(UserRow.email == email).first()
            if existing is not None:
                raise ValueError(f"邮箱 '{email}' 已被注册")

            user_id = uuid.uuid4().hex[:12]
            created_at = datetime.now(timezone.utc).isoformat()
            password_hash = hash_password(password)
            session.add(UserRow(
                id=user_id, email=email, password_hash=password_hash,
                name=name, created_at=created_at,
            ))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                raise ValueError(f"邮箱 '{email}' 已被注册")

        return User(id=user_id, email=email, name=name, created_at=created_at)

    def get_by_email(self, email: str) -> User | None:
        email = email.strip().lower()
        with self._session() as session:
            row = session.query(UserRow).filter(UserRow.email == email).first()
            if row is None:
                return None
            return User(id=row.id, email=row.email, name=row.name, created_at=row.created_at)

    def get_by_id(self, user_id: str) -> User | None:
        with self._session() as session:
            row = session.get(UserRow, user_id)
            if row is None:
                return None
            return User(id=row.id, email=row.email, name=row.name, created_at=row.created_at)

    def verify_credentials(self, email: str, password: str) -> User | None:
        """邮箱不存在或密码错误都返回 None，不区分具体原因——避免被用来
        枚举出哪些邮箱已经注册过。"""
        email = email.strip().lower()
        with self._session() as session:
            row = session.query(UserRow).filter(UserRow.email == email).first()
            if row is None:
                return None
            if not verify_password(password, row.password_hash):
                return None
            return User(id=row.id, email=row.email, name=row.name, created_at=row.created_at)
