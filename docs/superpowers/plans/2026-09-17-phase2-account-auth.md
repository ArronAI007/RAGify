# Phase 2: 账户认证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 RAGify 加一套独立的账户认证基础设施——User 表、注册/登录、bcrypt 密码哈希、PyJWT 会话——但不改动任何现有接口的鉴权要求，也不做前端登录页 UI。

**Architecture:** 沿用 Phase 1 确立的分层模式：`ragify/core/security.py`（纯函数）→ `ragify/core/user_manager.py`（DB 驱动，跟 `KBManager` 同构）→ `ragify/api/routers/auth.py`（FastAPI 路由）。前端只加 4 个 Next.js 代理路由，把后端签发的 JWT 转成 httpOnly cookie；不做任何页面 UI。

**Tech Stack:** `bcrypt`（密码哈希）、`PyJWT`（会话令牌，HS256）、`email-validator`（配合 Pydantic `EmailStr`），其余复用 Phase 1 已经装好的 FastAPI/SQLAlchemy/Alembic 技术栈。

---

### Task 1: 添加 Python 依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 在 `dependencies` 列表末尾加入新依赖**

打开 `pyproject.toml`，在 `"httpx>=0.27.0",` 这一行后面加入：

```toml
  "bcrypt>=4.1.0",
  "PyJWT>=2.9.0",
  "email-validator>=2.2.0",
```

- [ ] **Step 2: 安装依赖**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify && uv pip install -e .
```

- [ ] **Step 3: 验证导入**

```bash
.venv/bin/python -c "import bcrypt, jwt, email_validator; print('ok')"
```

Expected: 打印 `ok`。

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: 添加 bcrypt/PyJWT/email-validator 依赖"
```

---

### Task 2: `UserRow` 数据模型 + Alembic 迁移

**Files:**
- Modify: `ragify/db/models.py`
- Create: `alembic/versions/xxxx_create_users_table.py`

- [ ] **Step 1: 在 `ragify/db/models.py` 末尾加 `UserRow`**

当前文件内容（供对照，不要动 `KnowledgeBaseRow` 那部分）：

```python
"""SQLAlchemy ORM models. Phase 1 只有 KnowledgeBaseRow 一张表——
不加 tenant_id/owner_id 之类的列，那是 Phase 4（数据隔离迁移）的职责，
账户/租户模型设计出来之后再加对应的外键和迁移脚本。
"""

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class KnowledgeBaseRow(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False, unique=True)
    description: Mapped[str] = mapped_column(nullable=False, default="")
    created_at: Mapped[str] = mapped_column(nullable=False)
```

在文件末尾追加（`KnowledgeBaseRow` 类定义之后）：

```python


class UserRow(Base):
    """Phase 2：账户表。跟 KnowledgeBaseRow 平级，互不关联——知识库的归属
    是 Phase 4（数据隔离迁移）的职责，Phase 2 只解决"这个人是谁"。
    """
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(nullable=False)
    name: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[str] = mapped_column(nullable=False)
```

- [ ] **Step 2: 生成迁移**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify && .venv/bin/alembic revision --autogenerate -m "create users table"
```

Expected: 在 `alembic/versions/` 下生成一个新文件（比如 `xxxx_create_users_table.py`），只包含 `users` 表的 `upgrade()`/`downgrade()`——不应该出现任何对 `knowledge_bases` 表的改动（如果出现了，说明当前数据库状态跟已有迁移不一致，先跑 `alembic upgrade head` 让数据库追上现有迁移再重新生成）。

- [ ] **Step 3: 检查生成的迁移内容**

打开生成的文件，确认 `upgrade()`/`downgrade()` 内容等价于：

```python
def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )


def downgrade() -> None:
    op.drop_table("users")
```

如果 autogenerate 生成的版本格式不完全一致（比如带了自动生成的注释、单行写法），把 `upgrade()`/`downgrade()` 替换成上面这段。`down_revision` 字段应该自动指向当前 `alembic/versions/` 下已有的最新一个 revision id（不用手动改，autogenerate 会正确设置）。

- [ ] **Step 4: 跑迁移，验证建表成功**

```bash
.venv/bin/alembic upgrade head
.venv/bin/python -c "
import sqlite3
conn = sqlite3.connect('vectorstore/ragify.db')
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
print(tables)
cols = conn.execute('PRAGMA table_info(users)').fetchall()
print(cols)
"
```

Expected: 第一行输出包含 `('users',)`（还有 `knowledge_bases`、`alembic_version`）；第二行输出 5 列：`id`（主键）、`email`、`password_hash`、`name`、`created_at`。

- [ ] **Step 5: 清理这次手动验证生成的数据库文件**

```bash
rm -f vectorstore/ragify.db
```

- [ ] **Step 6: Commit**

```bash
git add ragify/db/models.py alembic/versions/
git commit -m "feat: 新增 UserRow 模型和 users 表迁移"
```

---

### Task 3: `ragify/core/security.py`（密码哈希 + JWT）

**Files:**
- Create: `ragify/core/security.py`
- Test: `tests/test_security.py`

- [ ] **Step 1: 写失败的测试 `tests/test_security.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""密码哈希/校验 + JWT 编码/解码的单测。纯函数测试，不碰数据库。"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jwt as pyjwt

from ragify.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

TEST_SECRET = "test-secret-only-for-unit-tests"


class TestPasswordHashing(unittest.TestCase):
    def test_hash_is_not_the_plaintext(self):
        h = hash_password("correct horse battery staple")
        self.assertNotEqual(h, "correct horse battery staple")

    def test_same_password_hashes_differently_each_time(self):
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        self.assertNotEqual(h1, h2)  # bcrypt 每次用不同的随机 salt

    def test_verify_password_correct(self):
        h = hash_password("my-password")
        self.assertTrue(verify_password("my-password", h))

    def test_verify_password_wrong(self):
        h = hash_password("my-password")
        self.assertFalse(verify_password("wrong-password", h))


class TestJWT(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        token = create_access_token("user-123", "a@b.com", secret=TEST_SECRET)
        payload = decode_access_token(token, secret=TEST_SECRET)
        self.assertEqual(payload["sub"], "user-123")
        self.assertEqual(payload["email"], "a@b.com")

    def test_decode_with_wrong_secret_raises(self):
        token = create_access_token("user-123", "a@b.com", secret=TEST_SECRET)
        with self.assertRaises(pyjwt.PyJWTError):
            decode_access_token(token, secret="a-different-secret")

    def test_decode_expired_token_raises(self):
        # 手工构造一个已经过期的 token，绕开 create_access_token 固定 7 天
        # 有效期的默认值，直接测过期校验本身。
        expired_payload = {
            "sub": "user-123",
            "email": "a@b.com",
            "iat": datetime.now(timezone.utc) - timedelta(days=8),
            "exp": datetime.now(timezone.utc) - timedelta(days=1),
        }
        expired_token = pyjwt.encode(expired_payload, TEST_SECRET, algorithm="HS256")
        with self.assertRaises(pyjwt.ExpiredSignatureError):
            decode_access_token(expired_token, secret=TEST_SECRET)

    def test_decode_garbage_token_raises(self):
        with self.assertRaises(pyjwt.PyJWTError):
            decode_access_token("not-a-real-token", secret=TEST_SECRET)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_security -v 2>&1 | tail -15
```

Expected: `ModuleNotFoundError: No module named 'ragify.core.security'`。

- [ ] **Step 3: 写 `ragify/core/security.py`**

```python
"""密码哈希/校验 + JWT 编码/解码。纯函数，不依赖数据库。

JWT 密钥从环境变量 RAGIFY_JWT_SECRET 读取；如果没配置，进程启动后第一次
用到时生成一个随机密钥并打警告日志——这意味着不配置的话每次重启服务
都会让所有人的登录状态失效，这是刻意的安全默认行为（好过留一个硬编码
的、任何人都能伪造 token 的默认密钥）。生产部署应该在 .env 里配置
RAGIFY_JWT_SECRET，让重启服务不影响已登录用户。

create_access_token/decode_access_token 都接受一个可选的 secret 参数，
仅用于测试场景下传入一个已知的固定密钥，跳过上面这套"读环境变量/生成
随机密钥"的逻辑，让测试可以确定性地验证编码-解码往返。生产代码路径
永远不传这个参数，两处调用（签发 token 的 auth.py 和验证 token 的
get_current_user）都会一致地读到同一个进程内缓存的密钥。
"""

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

logger = logging.getLogger("ragify.core.security")

JWT_ALGORITHM = "HS256"
JWT_EXPIRES_DAYS = 7
BCRYPT_ROUNDS = 12

_fallback_secret: str | None = None


def _get_jwt_secret() -> str:
    global _fallback_secret
    secret = os.environ.get("RAGIFY_JWT_SECRET")
    if secret:
        return secret
    if _fallback_secret is None:
        _fallback_secret = secrets.token_hex(32)
        logger.warning(
            "未设置 RAGIFY_JWT_SECRET，已生成临时密钥——重启服务后所有登录状态"
            "会失效。生产使用请在 .env 里配置 RAGIFY_JWT_SECRET。"
        )
    return _fallback_secret


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user_id: str, email: str, secret: str | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(days=JWT_EXPIRES_DAYS),
    }
    return jwt.encode(payload, secret or _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, secret: str | None = None) -> dict:
    """token 无效/过期/签名不匹配都会抛 jwt.PyJWTError（或其子类）。"""
    return jwt.decode(token, secret or _get_jwt_secret(), algorithms=[JWT_ALGORITHM])
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_security -v
```

Expected: 8 个测试全部 `ok`。

- [ ] **Step 5: Commit**

```bash
git add ragify/core/security.py tests/test_security.py
git commit -m "feat: 新增密码哈希与 JWT 编码/解码（ragify/core/security.py）"
```

---

### Task 4: `ragify/core/user_manager.py`（DB 驱动的用户管理）

**Files:**
- Create: `ragify/core/user_manager.py`
- Test: `tests/test_user_manager.py`

- [ ] **Step 1: 写失败的测试 `tests/test_user_manager.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UserManager 测试
验证用户的增查、邮箱去重（含大小写归一化）、密码校验。
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.user_manager import UserManager
from ragify.db.models import Base
from ragify.db.session import get_engine


class TestUserManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = UserManager(database_url=self.db_url)

    def tearDown(self):
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_and_get_by_email(self):
        user = self.manager.create("Alice@Example.com", "password123", "Alice")
        self.assertTrue(user.id)
        self.assertEqual(user.email, "alice@example.com")  # 邮箱归一化成小写

        fetched = self.manager.get_by_email("alice@example.com")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "Alice")

    def test_get_by_email_is_case_insensitive(self):
        self.manager.create("Bob@Example.com", "password123", "Bob")
        fetched = self.manager.get_by_email("BOB@EXAMPLE.COM")
        self.assertIsNotNone(fetched)

    def test_create_duplicate_email_raises(self):
        self.manager.create("dup@example.com", "password123", "First")
        with self.assertRaises(ValueError):
            self.manager.create("dup@example.com", "password456", "Second")

    def test_create_duplicate_email_different_case_raises(self):
        self.manager.create("case@example.com", "password123", "First")
        with self.assertRaises(ValueError):
            self.manager.create("Case@Example.com", "password456", "Second")

    def test_create_short_password_raises(self):
        with self.assertRaises(ValueError):
            self.manager.create("short@example.com", "1234567", "Short")

    def test_create_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.manager.create("noname@example.com", "password123", "   ")

    def test_get_by_email_missing_returns_none(self):
        self.assertIsNone(self.manager.get_by_email("nobody@example.com"))

    def test_get_by_id_missing_returns_none(self):
        self.assertIsNone(self.manager.get_by_id("does-not-exist"))

    def test_get_by_id(self):
        user = self.manager.create("byid@example.com", "password123", "ById")
        fetched = self.manager.get_by_id(user.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.email, "byid@example.com")

    def test_verify_credentials_correct(self):
        self.manager.create("verify@example.com", "correct-password", "Verify")
        user = self.manager.verify_credentials("verify@example.com", "correct-password")
        self.assertIsNotNone(user)
        self.assertEqual(user.email, "verify@example.com")

    def test_verify_credentials_wrong_password_returns_none(self):
        self.manager.create("verify2@example.com", "correct-password", "Verify2")
        user = self.manager.verify_credentials("verify2@example.com", "wrong-password")
        self.assertIsNone(user)

    def test_verify_credentials_missing_email_returns_none(self):
        user = self.manager.verify_credentials("nobody2@example.com", "whatever123")
        self.assertIsNone(user)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_user_manager -v 2>&1 | tail -15
```

Expected: `ModuleNotFoundError: No module named 'ragify.core.user_manager'`。

- [ ] **Step 3: 写 `ragify/core/user_manager.py`**

```python
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
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_user_manager -v
```

Expected: 12 个测试全部 `ok`。

- [ ] **Step 5: 跑现有完整测试套件确认无回归**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -10
```

Expected: 全部通过（现有 94 个 + 本任务新增的测试，具体总数以实际输出为准），无 FAILED/ERROR。

- [ ] **Step 6: Commit**

```bash
git add ragify/core/user_manager.py tests/test_user_manager.py
git commit -m "feat: 新增 UserManager（用户增查、邮箱去重、密码校验）"
```

---

### Task 5: Schema 与依赖注入扩展

**Files:**
- Modify: `ragify/api/schemas.py`
- Modify: `ragify/api/dependencies.py`

- [ ] **Step 1: 在 `ragify/api/schemas.py` 末尾追加**

当前文件末尾是 `UpdateChunkRequest` 类，在它后面加：

```python


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
```

同时把文件顶部的 import 改成：

```python
from pydantic import BaseModel, EmailStr
```

- [ ] **Step 2: 验证 Pydantic 能正常导入（依赖 Task 1 装好的 email-validator）**

```bash
.venv/bin/python -c "from ragify.api.schemas import RegisterRequest, LoginRequest; print('ok')"
```

Expected: 打印 `ok`。如果报 `ImportError: email-validator is not installed`，说明 Task 1 的依赖没装上，回去检查。

- [ ] **Step 3: 在 `ragify/api/dependencies.py` 里加 `get_user_manager` 和 `get_current_user`**

当前文件顶部的 import 是：

```python
import os
import threading
from pathlib import Path

from ..config import get_config
from ..core.kb_manager import KBManager
```

改成：

```python
import os
import threading
from pathlib import Path

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import get_config
from ..core.kb_manager import KBManager
from ..core.security import decode_access_token
from ..core.user_manager import User, UserManager
```

在文件末尾（`resolve_kb_path` 函数之后）追加：

```python


_bearer_scheme = HTTPBearer(auto_error=False)


def get_user_manager() -> UserManager:
    return UserManager()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    manager: UserManager = Depends(get_user_manager),
) -> User:
    if credentials is None:
        raise HTTPException(status_code=401, detail="缺少登录凭证")
    try:
        payload = decode_access_token(credentials.credentials)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="登录凭证无效或已过期")
    user = manager.get_by_id(payload["sub"])
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user
```

- [ ] **Step 4: 验证能正常导入**

```bash
.venv/bin/python -c "from ragify.api.dependencies import get_current_user, get_user_manager; print('ok')"
```

Expected: 打印 `ok`。

- [ ] **Step 5: 跑现有完整测试套件确认没有破坏 KB 相关的路由（它们也在这个文件里）**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -10
```

Expected: 全部通过，无 FAILED/ERROR。

- [ ] **Step 6: Commit**

```bash
git add ragify/api/schemas.py ragify/api/dependencies.py
git commit -m "feat: 新增注册/登录请求 schema 和 get_current_user 依赖"
```

---

### Task 6: `/api/auth/*` 路由

**Files:**
- Create: `ragify/api/routers/auth.py`
- Modify: `ragify/api/main.py`
- Test: `tests/test_api_auth.py`

- [ ] **Step 1: 写失败的测试 `tests/test_api_auth.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/auth/register、/api/auth/login、/api/auth/me 路由测试。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_user_manager
from ragify.api.main import app
from ragify.core.user_manager import UserManager
from ragify.db.models import Base
from ragify.db.session import get_engine


class TestAuthRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = UserManager(database_url=self.db_url)
        app.dependency_overrides[get_user_manager] = lambda: self.manager
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_register_returns_token_and_user(self):
        res = self.client.post("/api/auth/register", json={
            "email": "new@example.com", "password": "password123", "name": "New User",
        })
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("access_token", body)
        self.assertEqual(body["user"]["email"], "new@example.com")
        self.assertEqual(body["user"]["name"], "New User")

    def test_register_duplicate_email_rejected(self):
        self.client.post("/api/auth/register", json={
            "email": "dup@example.com", "password": "password123", "name": "First",
        })
        res = self.client.post("/api/auth/register", json={
            "email": "dup@example.com", "password": "password456", "name": "Second",
        })
        self.assertEqual(res.status_code, 400)

    def test_register_short_password_rejected(self):
        res = self.client.post("/api/auth/register", json={
            "email": "shortpw@example.com", "password": "short", "name": "Short",
        })
        self.assertEqual(res.status_code, 400)

    def test_register_invalid_email_rejected(self):
        res = self.client.post("/api/auth/register", json={
            "email": "not-an-email", "password": "password123", "name": "Bad Email",
        })
        self.assertEqual(res.status_code, 422)  # Pydantic EmailStr 校验，路由函数都还没进

    def test_login_correct_credentials(self):
        self.client.post("/api/auth/register", json={
            "email": "login@example.com", "password": "password123", "name": "Login User",
        })
        res = self.client.post("/api/auth/login", json={
            "email": "login@example.com", "password": "password123",
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn("access_token", res.json())

    def test_login_wrong_password_rejected(self):
        self.client.post("/api/auth/register", json={
            "email": "login2@example.com", "password": "password123", "name": "Login User 2",
        })
        res = self.client.post("/api/auth/login", json={
            "email": "login2@example.com", "password": "wrong-password",
        })
        self.assertEqual(res.status_code, 401)

    def test_login_unknown_email_rejected(self):
        res = self.client.post("/api/auth/login", json={
            "email": "ghost@example.com", "password": "whatever123",
        })
        self.assertEqual(res.status_code, 401)

    def test_me_with_valid_token(self):
        register_res = self.client.post("/api/auth/register", json={
            "email": "me@example.com", "password": "password123", "name": "Me User",
        })
        token = register_res.json()["access_token"]
        res = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["email"], "me@example.com")

    def test_me_without_token_rejected(self):
        res = self.client.get("/api/auth/me")
        self.assertEqual(res.status_code, 401)

    def test_me_with_garbage_token_rejected(self):
        res = self.client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
        self.assertEqual(res.status_code, 401)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_api_auth -v 2>&1 | tail -15
```

Expected: 报错（`ModuleNotFoundError` 或者一堆 404），因为路由和 `auth.py` 还不存在。

- [ ] **Step 3: 写 `ragify/api/routers/auth.py`**

```python
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_current_user, get_user_manager
from ..schemas import LoginRequest, RegisterRequest
from ...core.security import create_access_token
from ...core.user_manager import User, UserManager

router = APIRouter()


def _user_out(user: User) -> dict:
    return {"id": user.id, "email": user.email, "name": user.name, "created_at": user.created_at}


@router.post("/api/auth/register")
def register(body: RegisterRequest, manager: UserManager = Depends(get_user_manager)) -> dict:
    try:
        user = manager.create(body.email, body.password, body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = create_access_token(user.id, user.email)
    return {"access_token": token, "token_type": "bearer", "user": _user_out(user)}


@router.post("/api/auth/login")
def login(body: LoginRequest, manager: UserManager = Depends(get_user_manager)) -> dict:
    user = manager.verify_credentials(body.email, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    token = create_access_token(user.id, user.email)
    return {"access_token": token, "token_type": "bearer", "user": _user_out(user)}


@router.get("/api/auth/me")
def me(current_user: User = Depends(get_current_user)) -> dict:
    return _user_out(current_user)
```

- [ ] **Step 4: 在 `ragify/api/main.py` 里挂载这个 router**

当前 `main.py` 完整内容：

```python
from fastapi import FastAPI

from .routers import documents, health, kb, query
from ..core.kb_manager import KBManager

app = FastAPI(title="RAGify API")

app.include_router(kb.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(health.router)


@app.on_event("startup")
def _migrate_legacy_json_on_startup() -> None:
    KBManager().migrate_json_if_needed()
```

改成：

```python
from fastapi import FastAPI

from .routers import auth, documents, health, kb, query
from ..core.kb_manager import KBManager

app = FastAPI(title="RAGify API")

app.include_router(kb.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(auth.router)
app.include_router(health.router)


@app.on_event("startup")
def _migrate_legacy_json_on_startup() -> None:
    KBManager().migrate_json_if_needed()
```

- [ ] **Step 5: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_api_auth -v
```

Expected: 10 个测试全部 `ok`。

- [ ] **Step 6: 跑全量测试套件确认无回归**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -10
```

- [ ] **Step 7: Commit**

```bash
git add ragify/api/routers/auth.py ragify/api/main.py tests/test_api_auth.py
git commit -m "feat: /api/auth/register, /api/auth/login, /api/auth/me 路由"
```

---

### Task 7: `callBackend` 支持自定义请求头

**Files:**
- Modify: `frontend/src/lib/backend.ts`

- [ ] **Step 1: 修改 `frontend/src/lib/backend.ts`**

当前完整内容：

```typescript
const API_BASE = process.env.RAGIFY_API_URL || "http://localhost:8000";

interface CallBackendOptions {
  method?: string;
  timeout?: number;
}

export async function callBackend<T>(
  path: string,
  body?: Record<string, unknown>,
  opts: CallBackendOptions = {}
): Promise<T> {
  const method = opts.method ?? (body !== undefined ? "POST" : "GET");
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(opts.timeout ?? 30_000),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(
      typeof data.detail === "string" ? data.detail : `${res.status} ${res.statusText}`
    );
  }
  return data as T;
}
```

改成（只加了 `headers?: Record<string, string>` 这个可选字段和它的合并逻辑，向后兼容——现有所有调用点不传 `headers` 时行为完全不变）：

```typescript
const API_BASE = process.env.RAGIFY_API_URL || "http://localhost:8000";

interface CallBackendOptions {
  method?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

export async function callBackend<T>(
  path: string,
  body?: Record<string, unknown>,
  opts: CallBackendOptions = {}
): Promise<T> {
  const method = opts.method ?? (body !== undefined ? "POST" : "GET");
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json", ...opts.headers },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(opts.timeout ?? 30_000),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(
      typeof data.detail === "string" ? data.detail : `${res.status} ${res.statusText}`
    );
  }
  return data as T;
}
```

- [ ] **Step 2: 类型检查**

```bash
cd frontend && npx tsc --noEmit --pretty false
```

Expected: 无输出。

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/backend.ts
git commit -m "feat: callBackend 支持传自定义请求头（为 Authorization: Bearer 做准备）"
```

---

### Task 8: 前端共享 cookie 辅助模块

**Files:**
- Create: `frontend/src/lib/auth-cookie.ts`

- [ ] **Step 1: 写 `frontend/src/lib/auth-cookie.ts`**

```typescript
import { NextResponse } from "next/server";

export const AUTH_COOKIE_NAME = "ragify_token";

// 7 天，需要跟后端 ragify/core/security.py 里 JWT_EXPIRES_DAYS 保持一致——
// 两边都改的话记得同步改。
const AUTH_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 7;

export function setAuthCookie(response: NextResponse, token: string): void {
  response.cookies.set(AUTH_COOKIE_NAME, token, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    maxAge: AUTH_COOKIE_MAX_AGE_SECONDS,
    path: "/",
  });
}
```

- [ ] **Step 2: 类型检查**

```bash
cd frontend && npx tsc --noEmit --pretty false
```

Expected: 无输出（还没有任何文件引用这个新模块）。

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/auth-cookie.ts
git commit -m "feat: 新增前端 auth cookie 共享辅助模块"
```

---

### Task 9: 4 个前端 `route.ts`

**Files:**
- Create: `frontend/src/app/api/auth/register/route.ts`
- Create: `frontend/src/app/api/auth/login/route.ts`
- Create: `frontend/src/app/api/auth/logout/route.ts`
- Create: `frontend/src/app/api/auth/me/route.ts`

- [ ] **Step 1: 创建目录并写 `frontend/src/app/api/auth/register/route.ts`**

```bash
mkdir -p frontend/src/app/api/auth/register frontend/src/app/api/auth/login frontend/src/app/api/auth/logout frontend/src/app/api/auth/me
```

```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { setAuthCookie } from "@/lib/auth-cookie";

interface AuthResult {
  access_token: string;
  user: Record<string, unknown>;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const result = await callBackend<AuthResult>(
      "/api/auth/register",
      { email: body.email, password: body.password, name: body.name },
      { timeout: 15_000 }
    );
    const response = NextResponse.json({ user: result.user });
    setAuthCookie(response, result.access_token);
    return response;
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
```

- [ ] **Step 2: 写 `frontend/src/app/api/auth/login/route.ts`**

```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { setAuthCookie } from "@/lib/auth-cookie";

interface AuthResult {
  access_token: string;
  user: Record<string, unknown>;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const result = await callBackend<AuthResult>(
      "/api/auth/login",
      { email: body.email, password: body.password },
      { timeout: 15_000 }
    );
    const response = NextResponse.json({ user: result.user });
    setAuthCookie(response, result.access_token);
    return response;
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
```

- [ ] **Step 3: 写 `frontend/src/app/api/auth/logout/route.ts`**

这个接口不调用后端——JWT 是无状态的，服务端没有 session 可以失效，登出就是把浏览器里的 cookie 清掉。

```typescript
import { NextResponse } from "next/server";
import { AUTH_COOKIE_NAME } from "@/lib/auth-cookie";

export async function POST() {
  const response = NextResponse.json({ success: true });
  response.cookies.delete(AUTH_COOKIE_NAME);
  return response;
}
```

- [ ] **Step 4: 写 `frontend/src/app/api/auth/me/route.ts`**

```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { AUTH_COOKIE_NAME } from "@/lib/auth-cookie";

export async function GET(req: NextRequest) {
  const token = req.cookies.get(AUTH_COOKIE_NAME)?.value;
  if (!token) {
    return NextResponse.json({ error: "未登录" }, { status: 401 });
  }
  try {
    const result = await callBackend(
      "/api/auth/me",
      undefined,
      { method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` } }
    );
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 5: 类型检查和 lint**

```bash
cd frontend && npx tsc --noEmit --pretty false && npx eslint src/app/api/auth
```

Expected: 都无输出/无错误。

- [ ] **Step 6: Commit**

```bash
git add frontend/src/app/api/auth/
git commit -m "feat: 新增注册/登录/登出/me 四个前端代理路由"
```

---

### Task 10: 端到端验证 + 全量测试 + 收尾

**Files:** 无新文件，只做验证

- [ ] **Step 1: 启动完整服务栈**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify
./start.sh start
sleep 3
```

- [ ] **Step 2: 用 curl 走一遍完整的注册 → 登录 → me 流程，验证 Next.js 代理层 + FastAPI 后端 + cookie 全部串联正确**

```bash
# 注册，把响应头和响应体都存下来看
curl -i -s -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"e2e-test@example.com","password":"password123","name":"E2E Test User"}' \
  -c /tmp/ragify-e2e-cookies.txt
```

Expected: HTTP 200，响应体是 `{"user":{"id":"...","email":"e2e-test@example.com","name":"E2E Test User","created_at":"..."}}`（不包含 `access_token`——那个只存在于 httpOnly cookie 里，不暴露给客户端 JS），响应头里有 `Set-Cookie: ragify_token=...; HttpOnly; ...`。

```bash
# 带着刚才存的 cookie 去调 /api/auth/me，验证 session 生效
curl -s http://localhost:3000/api/auth/me -b /tmp/ragify-e2e-cookies.txt
```

Expected: `{"id":"...","email":"e2e-test@example.com","name":"E2E Test User","created_at":"..."}`。

```bash
# 不带 cookie 调 /api/auth/me，验证确实需要登录
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:3000/api/auth/me
```

Expected: `401`。

```bash
# 登出，验证 cookie 被清空后 /api/auth/me 又变回 401
curl -s -X POST http://localhost:3000/api/auth/logout -b /tmp/ragify-e2e-cookies.txt -c /tmp/ragify-e2e-cookies.txt
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:3000/api/auth/me -b /tmp/ragify-e2e-cookies.txt
```

Expected: `401`（cookie 已经被清空/过期）。

```bash
# 重复注册同一个邮箱，验证冲突正确处理
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"e2e-test@example.com","password":"password123","name":"Dup"}'
```

Expected: `500`（Next.js 层对所有后端错误统一包装成 500，这是 Phase 1 就确立的既有约定——参见 `frontend/src/lib/backend.ts` 的错误处理逻辑）。

```bash
# 登录刚注册的账号
curl -s -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"e2e-test@example.com","password":"password123"}' \
  -c /tmp/ragify-e2e-cookies2.txt
curl -s http://localhost:3000/api/auth/me -b /tmp/ragify-e2e-cookies2.txt
```

Expected: 登录返回 `{"user": {...}}`；带上新 cookie 调 `/api/auth/me` 返回同一个用户信息。

- [ ] **Step 3: 确认现有功能完全不受影响（这是本阶段最重要的不变量：Phase 2 不改变任何现有接口的行为）**

```bash
curl -s -o /dev/null -w "kb list: %{http_code}\n" http://localhost:3000/api/knowledge-bases
curl -s -o /dev/null -w "health: %{http_code}\n" http://localhost:3000/api/health
```

Expected: 两个都是 `200`——完全不需要登录，跟 Phase 2 开始之前一模一样。

- [ ] **Step 4: 清理测试数据**

```bash
rm -f /tmp/ragify-e2e-cookies.txt /tmp/ragify-e2e-cookies2.txt
```

（测试过程中通过 `POST /api/auth/register` 建的 `e2e-test@example.com` 这个用户留在数据库里也无妨——Phase 2 没有任何"删除用户"的接口，这是预期状态，不用清理。）

- [ ] **Step 5: 跑全量 Python 测试套件**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -15
```

Expected: 全部通过，无 FAILED/ERROR。

- [ ] **Step 6: 前端类型检查和 lint**

```bash
cd frontend && npx tsc --noEmit --pretty false && npx eslint src/app/api
```

Expected: 都无输出/无错误。

- [ ] **Step 7: 停止服务，确认工作区干净**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify
./start.sh stop
git status --short
```

Expected: `start.sh status` 显示两个服务都未运行；`git status --short` 只剩下已知的、跟本次任务无关的历史遗留改动（如果有 `ragify.egg-info/*`、`__pycache__/*.pyc` 这类构建产物噪音，用 `git restore` 丢弃，不要提交）。

- [ ] **Step 8: 最终确认所有提交都在**

```bash
git log --oneline <Task 1 之前的 commit>..HEAD
```

Expected: 能看到本计划 Task 1-9 对应的全部 commit。
