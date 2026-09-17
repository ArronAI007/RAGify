# Phase 1: 后端服务化 + 数据库基础设施 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 RAGify 现有的无状态 Python 子进程后端（`frontend/scripts/bridge.py`，每次请求 spawn 一个新进程，元数据存在 `vectorstore/kbs.json`）迁移成常驻 FastAPI 服务 + SQLAlchemy/SQLite 数据库，用户可见行为（建知识库、传文档索引、标准/Agentic 问答）保持完全不变。这是多租户账户系统 5 阶段规划的第 1 阶段，本阶段不引入任何账户/租户概念。

**Architecture:** Next.js API 路由（瘦代理层，`frontend/src/app/api/*/route.ts`）通过 HTTP 调用一个常驻的 FastAPI 服务（`ragify/api/`），FastAPI 路由复用现有的 `ragify/mcp`（Pipeline）、`ragify/agentic`（AgenticRAG）核心逻辑，只是把 `KBManager` 的存储从 JSON 文件换成 SQLAlchemy + SQLite（`ragify/db/`）。`start.sh` 同时管理 uvicorn 和 Next.js 两个后台进程。

**Tech Stack:** FastAPI, uvicorn, SQLAlchemy 2.0, Alembic, SQLite, httpx（测试用），现有的 LangChain/FAISS/DashScope 技术栈不变。

**⚠️ 一个必须理解的设计前提（第一个任务前必读）：** 现有 `_resolve_kb_path()`（在 `bridge.py` 里）会修改 `ragify.config.get_config()` 返回的全局单例配置对象的 `vectorstore.persist_directory` 字段，然后再构造 `VectorStoreManager`/`IndexingPipeline`/`QueryPipeline`/`AgenticRAG`。在"每请求一个新子进程"的旧模型下这是安全的（每个请求有自己独立的进程内存，不会互相干扰）。但在常驻 FastAPI 服务里，多个请求共享同一个 Python 进程和同一个全局配置单例——如果两个请求并发地"改配置 → 构造 manager"，会出现读到对方 KB 路径的竞态条件（跨知识库数据串扰）。

本计划的解决方式：用一个进程级 `threading.Lock`（`KB_LOCK`，定义在 `ragify/api/dependencies.py`）包住"改配置 + 构造 manager/pipeline"这一小段代码。两种情况区别对待：
- `QueryPipeline`/`IndexingPipeline`：它们的组件在 `__init__` 时就立即读一次全局配置并把 `persist_directory` 存成实例属性（验证见 `ragify/mcp/base.py:18` 的 `PipelineComponent.__init__` 会立即调用 `_setup()`），构造完之后可以安全地把锁释放掉，再执行真正耗时的 `.run()`（可能调用 LLM，不应该被全局锁卡住并发）。
- `AgenticRAG`：它的 `retrieve_docs` 工具是在 `.run()` 循环*执行期间*、每次被 LLM 调用时才现读一次全局配置（见 `ragify/agentic/agent.py` 里 `_retrieve_docs` 的调用时机），所以锁必须包住整个 `.run()` 调用，不能提前释放。

这意味着 Agentic 问答请求之间是全局串行的（同一时刻只能有一个 Agentic 请求在跑）。这是本阶段"先保证正确性、不引入隐藏 bug"的有意取舍——当前系统还没有真正的多用户并发场景（Phase 3/4 加账户体系之后才会有），等那时候有了真实的并发需求，再考虑把这个全局可变配置改造成显式传参（更大的重构，不在本阶段范围内）。

**与设计文档的三处偏差（写计划前逐个核对代码后发现的，都不是遗漏）：**

1. **不做 `settings.py` 路由。** 设计文档里提到过一个 `settings.py`（对应「系统设置」页）。核对代码后发现 `frontend/src/app/settings/page.tsx` 现在完全是本地假状态——没有任何 `fetch`/`useEffect` 调后端，`bridge.py` 的 `HANDLERS` 里也从来没有过 `get_settings`/`update_settings` 这个 action。既然没有真实行为可以"保持不变"，本阶段就不建这个路由，避免凭空造一个没有真实需求的接口。
2. **不做 `list_skills` 路由。** `bridge.py` 里有 `handle_list_skills`，但搜遍 `frontend/src/app/api/` 和 `frontend/src/lib/api.ts` 都没有任何调用点——是从没被接上过的死代码（跟我们之前修过的 `handle_agentic_query` 里那段死代码是同类问题）。`ragify/mcp_server/server.py` 有自己独立的 `_list_skills()` 实现，不依赖 bridge.py，不受影响。
3. **新增 Task 5，修 `ragify/mcp_server/server.py`。** 这个文件有两处直接调用 `KBManager.migrate_if_needed()`（跟 bridge.py 无关，是 MCP stdio 协议服务自己的调用）。Task 4 把这个方法改名成 `migrate_json_if_needed()` 之后，如果不同步改这两处，MCP server 会在运行时抛 `AttributeError`。

---

### Task 1: 添加 Python 依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 在 `dependencies` 列表末尾加入新依赖**

打开 `pyproject.toml`，找到第 14-28 行的 `dependencies = [...]` 列表，在 `"docx2txt>=0.8",` 这一行后面加入：

```toml
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "sqlalchemy>=2.0.0",
  "alembic>=1.13.0",
  "httpx>=0.27.0",
```

- [ ] **Step 2: 安装依赖**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify && uv pip install -e .
```

Expected: 输出里能看到 `fastapi`、`uvicorn`、`sqlalchemy`、`alembic`、`httpx` 被安装。

- [ ] **Step 3: 验证导入**

```bash
.venv/bin/python -c "import fastapi, uvicorn, sqlalchemy, alembic, httpx; print('ok')"
```

Expected: 打印 `ok`，无报错。

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: 添加 FastAPI/SQLAlchemy/Alembic 依赖"
```

---

### Task 2: 创建数据库基础设施（`ragify/db/`）

**Files:**
- Create: `ragify/db/__init__.py`
- Create: `ragify/db/models.py`
- Create: `ragify/db/session.py`

- [ ] **Step 1: 创建包目录和空 `__init__.py`**

```bash
mkdir -p ragify/db
touch ragify/db/__init__.py
```

- [ ] **Step 2: 写 `ragify/db/models.py`**

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

- [ ] **Step 3: 写 `ragify/db/session.py`**

```python
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
```

- [ ] **Step 4: 写一个临时验证脚本确认建表/读写正常，然后删除**

```bash
.venv/bin/python -c "
from ragify.db.models import Base
from ragify.db.session import get_engine, get_session
from ragify.db.models import KnowledgeBaseRow

engine = get_engine('sqlite:///:memory:')
Base.metadata.create_all(bind=engine)

session = get_session('sqlite:///:memory:')
# 注意：:memory: 每次 get_engine 调用同一个 url 会命中 lru_cache，拿到同一个
# engine，所以下面这个 session 用的是刚刚 create_all 的那个内存库
session.add(KnowledgeBaseRow(id='k1', name='test', description='', created_at='now'))
session.commit()
row = session.get(KnowledgeBaseRow, 'k1')
assert row.name == 'test'
print('models/session smoke test OK')
"
```

Expected: 打印 `models/session smoke test OK`，无报错。

- [ ] **Step 5: Commit**

```bash
git add ragify/db/
git commit -m "feat: 新增 ragify/db 包（SQLAlchemy models + session 工厂）"
```

---

### Task 3: 设置 Alembic，生成初始迁移

**Files:**
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/script.py.mako`（Alembic 自动生成）
- Create: `alembic/versions/xxxx_create_knowledge_bases_table.py`

- [ ] **Step 1: 初始化 Alembic**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify && .venv/bin/alembic init alembic
```

Expected: 生成 `alembic.ini` 和 `alembic/` 目录（含 `env.py`、`script.py.mako`、`versions/`）。

- [ ] **Step 2: 编辑 `alembic/env.py`，让它指向我们的 metadata 和数据库 URL**

把生成的 `alembic/env.py` 顶部（`config = context.config` 那一行之后、`target_metadata = None` 那一行）替换成：

```python
from ragify.db.models import Base
from ragify.db.session import default_database_url

config = context.config
config.set_main_option("sqlalchemy.url", default_database_url())

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
```

（即：在原有 `config = context.config` 后面插入 `config.set_main_option(...)` 这一行，并把原来的 `target_metadata = None` 改成 `target_metadata = Base.metadata`，两个 import 加在文件最上面 `from alembic import context` 后面。)

- [ ] **Step 3: 生成初始迁移**

```bash
.venv/bin/alembic revision --autogenerate -m "create knowledge_bases table"
```

Expected: 在 `alembic/versions/` 下生成一个新文件，打印类似 `Generating .../alembic/versions/xxxx_create_knowledge_bases_table.py ... done`。

- [ ] **Step 4: 检查生成的迁移文件内容**

打开刚生成的文件，确认 `upgrade()` 函数内容等价于（字段名、类型、约束都要对上；如果 autogenerate 生成的版本不完全一致，直接把 `upgrade()`/`downgrade()` 替换成下面这段）：

```python
def upgrade() -> None:
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )


def downgrade() -> None:
    op.drop_table("knowledge_bases")
```

- [ ] **Step 5: 跑迁移，验证建表成功**

```bash
.venv/bin/alembic upgrade head
.venv/bin/python -c "
import sqlite3
conn = sqlite3.connect('vectorstore/ragify.db')
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
print(tables)
cols = conn.execute('PRAGMA table_info(knowledge_bases)').fetchall()
print(cols)
"
```

Expected: 第一行输出包含 `('knowledge_bases',)` 和 `('alembic_version',)`；第二行输出 4 列：`id`（主键）、`name`、`description`、`created_at`。

- [ ] **Step 6: 清理这次手动验证生成的数据库文件（后面的自动化测试会用自己的临时库，不依赖这个文件）**

```bash
rm -f vectorstore/ragify.db
```

- [ ] **Step 7: Commit**

```bash
git add alembic.ini alembic/
git commit -m "feat: 引入 Alembic，创建 knowledge_bases 表的初始迁移"
```

---

### Task 4: 把 `KBManager` 改造成 DB 驱动

**Files:**
- Modify: `ragify/core/kb_manager.py`（整体重写，公开方法签名不变）
- Test: `tests/test_kb_manager.py`（新建）

- [ ] **Step 1: 写失败的测试 `tests/test_kb_manager.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KBManager 测试（DB 驱动版）
验证 KB 的增删查、名称去重，以及从 kbs.json / 旧版扁平索引迁移进数据库的逻辑。
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.kb_manager import KBManager
from ragify.db.session import get_engine


class TestKBManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        self.vectorstore_dir = Path(self.tmp_dir) / "vectorstore"
        self.manager = KBManager(database_url=self.db_url, vectorstore_dir=self.vectorstore_dir)

    def tearDown(self):
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_and_get(self):
        kb = self.manager.create("测试知识库", "描述")
        self.assertTrue(kb.id)
        fetched = self.manager.get(kb.id)
        self.assertEqual(fetched.name, "测试知识库")
        self.assertEqual(fetched.description, "描述")

    def test_create_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.manager.create("   ")

    def test_create_duplicate_name_raises(self):
        self.manager.create("重复名称")
        with self.assertRaises(ValueError):
            self.manager.create("重复名称")

    def test_list_all_empty(self):
        self.assertEqual(self.manager.list_all(), [])

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.manager.get("does-not-exist"))

    def test_delete_removes_kb_and_directory(self):
        kb = self.manager.create("待删除")
        kb_dir = Path(self.manager.get_persist_dir(kb.id))
        self.assertTrue(kb_dir.exists())

        ok = self.manager.delete(kb.id)

        self.assertTrue(ok)
        self.assertIsNone(self.manager.get(kb.id))
        self.assertFalse(kb_dir.exists())

    def test_delete_missing_returns_false(self):
        self.assertFalse(self.manager.delete("does-not-exist"))

    def test_migrate_json_if_needed_imports_existing_file(self):
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        kbs_file = self.vectorstore_dir / "kbs.json"
        kbs_file.write_text(json.dumps({"kbs": [
            {"id": "legacy1", "name": "旧知识库", "description": "", "created_at": "2024-01-01T00:00:00"}
        ]}), encoding="utf-8")

        migrated = self.manager.migrate_json_if_needed()

        self.assertTrue(migrated)
        kb = self.manager.get("legacy1")
        self.assertIsNotNone(kb)
        self.assertEqual(kb.name, "旧知识库")
        self.assertFalse(kbs_file.exists())
        self.assertTrue((self.vectorstore_dir / "kbs.json.migrated").exists())

    def test_migrate_json_if_needed_noop_when_db_has_rows(self):
        self.manager.create("已有数据")
        self.assertFalse(self.manager.migrate_json_if_needed())

    def test_migrate_json_if_needed_noop_when_nothing_to_migrate(self):
        self.assertFalse(self.manager.migrate_json_if_needed())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败（因为 `KBManager` 还是旧的 JSON 实现，构造函数不接受 `database_url`/`vectorstore_dir` 参数）**

```bash
.venv/bin/python -m unittest tests.test_kb_manager -v 2>&1 | tail -20
```

Expected: 报错，类似 `TypeError: KBManager.__init__() got an unexpected keyword argument 'database_url'`。

- [ ] **Step 3: 重写 `ragify/core/kb_manager.py`**

用下面的内容整体替换 `ragify/core/kb_manager.py`：

```python
import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from ..db.models import KnowledgeBaseRow
from ..db.session import get_session

logger = logging.getLogger("ragify.core.kb_manager")

DEFAULT_VECTORSTORE_DIR = Path("vectorstore")


@dataclass
class KnowledgeBase:
    id: str
    name: str
    description: str
    created_at: str


class KBManager:
    def __init__(
        self,
        database_url: str | None = None,
        vectorstore_dir: str | Path | None = None,
    ):
        self.database_url = database_url
        self.vectorstore_dir = Path(vectorstore_dir) if vectorstore_dir else DEFAULT_VECTORSTORE_DIR
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)

    def _session(self) -> Session:
        return get_session(self.database_url)

    @property
    def _kbs_json_file(self) -> Path:
        return self.vectorstore_dir / "kbs.json"

    def migrate_json_if_needed(self) -> bool:
        """One-time startup migration: bring existing on-disk state into the DB.

        Handles two legacy states:
        - kbs.json exists (post-KB-support installs): import its rows into the DB,
          then rename the file to kbs.json.migrated so it is not re-imported.
        - No kbs.json but a flat index.faiss + index.pkl exist directly under
          vectorstore_dir (pre-KB-support installs): move them into a new per-KB
          directory and insert one row for it.

        Returns True if a migration ran, False if there was nothing to migrate
        (including when the DB already has rows).
        """
        if self.list_all():
            return False

        kbs_file = self._kbs_json_file
        if kbs_file.exists():
            data = json.loads(kbs_file.read_text(encoding="utf-8"))
            with self._session() as session:
                for item in data.get("kbs", []):
                    session.add(KnowledgeBaseRow(
                        id=item["id"],
                        name=item["name"],
                        description=item.get("description", ""),
                        created_at=item.get("created_at", ""),
                    ))
                session.commit()
            kbs_file.rename(kbs_file.with_suffix(".json.migrated"))
            logger.info("已将 %s 迁移进数据库", kbs_file)
            return True

        old_index = self.vectorstore_dir / "index.faiss"
        old_pkl = self.vectorstore_dir / "index.pkl"
        if old_index.exists() and old_pkl.exists():
            kb_id = "default-" + uuid.uuid4().hex[:8]
            kb_dir = self.vectorstore_dir / kb_id
            kb_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_index), str(kb_dir / "index.faiss"))
            shutil.move(str(old_pkl), str(kb_dir / "index.pkl"))

            with self._session() as session:
                session.add(KnowledgeBaseRow(
                    id=kb_id,
                    name="默认知识库",
                    description="迁移自旧版本数据",
                    created_at=datetime.now(timezone.utc).isoformat(),
                ))
                session.commit()
            logger.info("已迁移旧索引到 KB '默认知识库' (%s)", kb_id)
            return True

        return False

    def create(self, name: str, description: str = "") -> KnowledgeBase:
        name = name.strip()
        if not name:
            raise ValueError("知识库名称不能为空")

        with self._session() as session:
            existing = {row.name.lower() for row in session.query(KnowledgeBaseRow).all()}
            if name.lower() in existing:
                raise ValueError(f"知识库 '{name}' 已存在")

            kb_id = uuid.uuid4().hex[:12]
            created_at = datetime.now(timezone.utc).isoformat()
            description = description.strip()
            session.add(KnowledgeBaseRow(
                id=kb_id, name=name, description=description, created_at=created_at,
            ))
            session.commit()
            result = KnowledgeBase(id=kb_id, name=name, description=description, created_at=created_at)

        kb_dir = self.vectorstore_dir / kb_id
        kb_dir.mkdir(parents=True, exist_ok=True)
        logger.info("创建知识库 '%s' (%s)", name, kb_id)
        return result

    def delete(self, kb_id: str) -> bool:
        with self._session() as session:
            row = session.get(KnowledgeBaseRow, kb_id)
            if row is None:
                return False
            name = row.name
            session.delete(row)
            session.commit()

        kb_dir = self.vectorstore_dir / kb_id
        if kb_dir.exists():
            shutil.rmtree(str(kb_dir))
        logger.info("删除知识库 '%s' (%s)", name, kb_id)
        return True

    def list_all(self) -> list[KnowledgeBase]:
        with self._session() as session:
            rows = session.query(KnowledgeBaseRow).all()
            return [
                KnowledgeBase(id=r.id, name=r.name, description=r.description, created_at=r.created_at)
                for r in rows
            ]

    def get(self, kb_id: str) -> KnowledgeBase | None:
        with self._session() as session:
            row = session.get(KnowledgeBaseRow, kb_id)
            if row is None:
                return None
            return KnowledgeBase(id=row.id, name=row.name, description=row.description, created_at=row.created_at)

    def get_persist_dir(self, kb_id: str) -> str:
        return str(self.vectorstore_dir / kb_id)
```

注意这个版本比旧版少了一个 `update_doc_count()` 方法——那是旧代码里一个从没被调用过的空 no-op（`pass`），删掉它之前已用 `grep -rn "update_doc_count"` 确认过全代码库没有任何调用点。

- [ ] **Step 4: 跑测试，确认全部通过**

```bash
.venv/bin/python -m unittest tests.test_kb_manager -v
```

Expected: 全部 10 个测试 `ok`。

- [ ] **Step 5: 跑现有的完整 Python 测试套件，确认没有回归**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -15
```

Expected: `Ran 6X tests ... OK`（现有 5 个测试文件 61 个用例 + 新增的 `test_kb_manager.py` 10 个用例，全部通过；具体总数以实际输出为准，但不能有 FAILED/ERROR）。

- [ ] **Step 6: Commit**

```bash
git add ragify/core/kb_manager.py tests/test_kb_manager.py
git commit -m "feat: KBManager 存储层从 JSON 文件迁移到 SQLAlchemy"
```

---

### Task 5: 修复 `ragify/mcp_server/server.py` 里的方法名引用

`KBManager.migrate_if_needed()` 在上一个任务里改名成了 `migrate_json_if_needed()`。`ragify/mcp_server/server.py` 有两处直接调用这个方法（跟 `bridge.py` 无关，是 MCP stdio 协议服务自己的调用），必须同步改名，否则会在运行时报 `AttributeError`。

**Files:**
- Modify: `ragify/mcp_server/server.py:39`
- Modify: `ragify/mcp_server/server.py:103`

- [ ] **Step 1: 改第 39 行**

在 `_list_resources()` 函数里，把：
```python
    manager = KBManager()
    manager.migrate_if_needed()
```
改成：
```python
    manager = KBManager()
    manager.migrate_json_if_needed()
```

- [ ] **Step 2: 改第 103 行**

在 `_call_tool()` 函数的 `ragify_list_kbs` 分支里，把：
```python
        manager = KBManager()
        manager.migrate_if_needed()
```
改成：
```python
        manager = KBManager()
        manager.migrate_json_if_needed()
```

- [ ] **Step 3: 验证没有遗漏的旧方法名引用**

```bash
grep -rn "migrate_if_needed" --include="*.py" ragify/
```

Expected: 无输出（`ragify/` 目录下已经没有任何地方还在调用旧名字；`frontend/scripts/bridge.py` 里还有几处，但那个文件会在 Task 11 里被整个删除，暂时不用管）。

- [ ] **Step 4: Commit**

```bash
git add ragify/mcp_server/server.py
git commit -m "fix: 同步 mcp_server 对 KBManager.migrate_json_if_needed 改名的引用"
```

---

### Task 6: FastAPI 依赖注入与请求 Schema

**Files:**
- Create: `ragify/api/__init__.py`
- Create: `ragify/api/dependencies.py`
- Create: `ragify/api/schemas.py`

- [ ] **Step 1: 创建包目录**

```bash
mkdir -p ragify/api/routers
touch ragify/api/__init__.py ragify/api/routers/__init__.py
```

- [ ] **Step 2: 写 `ragify/api/dependencies.py`**

```python
"""FastAPI 依赖注入 + 并发安全辅助函数。

见本计划文档开头的并发说明：KB_LOCK 用来保护"改全局 vectorstore 配置 +
构造读这个配置的对象"这一小段临界区，调用方式见各 router 文件。
"""

import os
import threading
from pathlib import Path

from ..config import get_config
from ..core.kb_manager import KBManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

KB_LOCK = threading.Lock()


def get_kb_manager() -> KBManager:
    return KBManager()


def resolve_kb_path(manager: KBManager, kb_id: str | None) -> str:
    """解析 kb_id 对应的 persist_directory，并把它写进全局 vectorstore 配置。

    调用方必须已经持有 KB_LOCK。如果 kb_id 为 None，回退到第一个可用知识库。
    """
    if kb_id:
        kb = manager.get(kb_id)
        if kb is None:
            raise ValueError(f"知识库 '{kb_id}' 不存在")
    else:
        all_kbs = manager.list_all()
        if not all_kbs:
            raise ValueError("没有可用知识库，请先创建知识库")
        kb_id = all_kbs[0].id

    persist_dir = manager.get_persist_dir(kb_id)
    get_config().update("vectorstore.persist_directory", persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir
```

- [ ] **Step 3: 写 `ragify/api/schemas.py`**

```python
"""Pydantic 请求体模型，一一对应 frontend route.ts 发过来的 JSON 形状。"""

from pydantic import BaseModel


class CreateKBRequest(BaseModel):
    name: str
    description: str = ""


class QueryRequest(BaseModel):
    query: str
    k: int = 3
    score_threshold: float | None = None
    kb_id: str | None = None


class AgenticQueryRequest(BaseModel):
    query: str
    kb_id: str | None = None
    chat_history: list[dict] | None = None
    max_iterations: int | None = None


class IndexRequest(BaseModel):
    directory_path: str | None = None
    file_paths: list[str] | None = None
    clear_vectorstore: bool | None = None
    kb_id: str | None = None


class ClearIndexRequest(BaseModel):
    kb_id: str | None = None


class DeleteDocRequest(BaseModel):
    kb_id: str
    source: str


class UpdateChunkRequest(BaseModel):
    kb_id: str | None = None
    chunk_id: str
    content: str
```

- [ ] **Step 4: 验证能正常导入**

```bash
.venv/bin/python -c "from ragify.api.dependencies import KB_LOCK, get_kb_manager, resolve_kb_path; from ragify.api import schemas; print('ok')"
```

Expected: 打印 `ok`。

- [ ] **Step 5: Commit**

```bash
git add ragify/api/__init__.py ragify/api/routers/__init__.py ragify/api/dependencies.py ragify/api/schemas.py
git commit -m "feat: 新增 ragify/api 依赖注入与请求 schema"
```

---

### Task 7: health 路由 + 最小可跑的 FastAPI app

**Files:**
- Create: `ragify/api/routers/health.py`
- Create: `ragify/api/main.py`
- Test: `tests/test_api_health.py`

- [ ] **Step 1: 写失败的测试 `tests/test_api_health.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI /api/health 路由测试。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.main import app


class TestHealthRoute(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_returns_200_with_expected_shape(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("status", body)
        self.assertIn("version", body)
        self.assertIn("llm_provider", body)
        self.assertIn("vectorstore_type", body)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_api_health -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'ragify.api.main'`。

- [ ] **Step 3: 写 `ragify/api/routers/health.py`**

```python
from fastapi import APIRouter

from ...config import get_config

router = APIRouter()


@router.get("/api/health")
def get_health() -> dict:
    cfg = get_config()
    return {
        "status": "healthy",
        "version": cfg.get("base.version", "0.1.0"),
        "llm_provider": cfg.get("llm.provider", "unknown"),
        "vectorstore_type": cfg.get("vectorstore.type", "unknown"),
    }
```

- [ ] **Step 4: 写 `ragify/api/main.py`**

```python
from fastapi import FastAPI

from .routers import health

app = FastAPI(title="RAGify API")

app.include_router(health.router)
```

- [ ] **Step 5: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_api_health -v
```

Expected: 1 个测试 `ok`。

- [ ] **Step 6: Commit**

```bash
git add ragify/api/routers/health.py ragify/api/main.py tests/test_api_health.py
git commit -m "feat: FastAPI app 骨架 + /api/health 路由"
```

---

### Task 8: 知识库管理路由（`/api/kb`）

**Files:**
- Create: `ragify/api/routers/kb.py`
- Modify: `ragify/api/main.py`
- Test: `tests/test_api_kb.py`

- [ ] **Step 1: 写失败的测试 `tests/test_api_kb.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/kb 路由测试：建/查/删知识库。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_kb_manager
from ragify.api.main import app
from ragify.core.kb_manager import KBManager


class TestKBRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.manager = KBManager(
            database_url=f"sqlite:///{self.tmp_dir}/test.db",
            vectorstore_dir=Path(self.tmp_dir) / "vectorstore",
        )
        app.dependency_overrides[get_kb_manager] = lambda: self.manager
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_list_kbs_empty(self):
        res = self.client.get("/api/kb")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"knowledge_bases": []})

    def test_create_and_list_kb(self):
        res = self.client.post("/api/kb", json={"name": "测试库", "description": "desc"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["name"], "测试库")

        res = self.client.get("/api/kb")
        kbs = res.json()["knowledge_bases"]
        self.assertEqual(len(kbs), 1)
        self.assertEqual(kbs[0]["name"], "测试库")
        self.assertEqual(kbs[0]["doc_count"], 0)

    def test_create_kb_empty_name_rejected(self):
        res = self.client.post("/api/kb", json={"name": "   "})
        self.assertEqual(res.status_code, 400)

    def test_create_kb_duplicate_name_rejected(self):
        self.client.post("/api/kb", json={"name": "重复"})
        res = self.client.post("/api/kb", json={"name": "重复"})
        self.assertEqual(res.status_code, 400)

    def test_delete_kb(self):
        create_res = self.client.post("/api/kb", json={"name": "待删除"})
        kb_id = create_res.json()["id"]
        res = self.client.delete(f"/api/kb/{kb_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"success": True})

    def test_delete_missing_kb_returns_404(self):
        res = self.client.delete("/api/kb/does-not-exist")
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败（404，因为路由还不存在）**

```bash
.venv/bin/python -m unittest tests.test_api_kb -v 2>&1 | tail -15
```

- [ ] **Step 3: 写 `ragify/api/routers/kb.py`**

```python
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import KB_LOCK, get_kb_manager
from ..schemas import CreateKBRequest
from ...config import get_config
from ...core.kb_manager import KBManager
from ...core.vectorstores import VectorStoreManager

router = APIRouter()


@router.get("/api/kb")
def list_kbs(manager: KBManager = Depends(get_kb_manager)) -> dict:
    kbs = manager.list_all()
    kbs_out = []
    with KB_LOCK:
        for kb in kbs:
            doc_count = 0
            try:
                persist_dir = manager.get_persist_dir(kb.id)
                get_config().update("vectorstore.persist_directory", persist_dir)
                vm = VectorStoreManager()
                doc_count = vm.get_document_count()
            except Exception:
                pass
            kbs_out.append({
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "created_at": kb.created_at,
                "doc_count": doc_count,
            })
    return {"knowledge_bases": kbs_out}


@router.post("/api/kb")
def create_kb(body: CreateKBRequest, manager: KBManager = Depends(get_kb_manager)) -> dict:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="知识库名称不能为空")
    try:
        kb = manager.create(name, body.description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "created_at": kb.created_at,
    }


@router.delete("/api/kb/{kb_id}")
def delete_kb(kb_id: str, manager: KBManager = Depends(get_kb_manager)) -> dict:
    ok = manager.delete(kb_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"知识库 '{kb_id}' 不存在")
    return {"success": True}
```

- [ ] **Step 4: 在 `ragify/api/main.py` 里挂载这个 router**

把 `ragify/api/main.py` 改成：

```python
from fastapi import FastAPI

from .routers import health, kb

app = FastAPI(title="RAGify API")

app.include_router(kb.router)
app.include_router(health.router)
```

- [ ] **Step 5: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_api_kb -v
```

Expected: 6 个测试全部 `ok`。

- [ ] **Step 6: Commit**

```bash
git add ragify/api/routers/kb.py ragify/api/main.py tests/test_api_kb.py
git commit -m "feat: /api/kb 知识库管理路由"
```

---

### Task 9: 查询路由（`/api/query`、`/api/query/agentic`）

**Files:**
- Create: `ragify/api/routers/query.py`
- Modify: `ragify/api/main.py`
- Test: `tests/test_api_query.py`

- [ ] **Step 1: 写失败的测试 `tests/test_api_query.py`**

这里用 `unittest.mock.patch` 挡住真正的 LLM/向量库调用（跟 `tests/test_agentic_agent.py` 里的做法一致），只验证路由层的参数透传、错误处理和响应 reshape，不是重新测一遍 RAG 逻辑本身（那部分已经被 `tests/test_pipeline.py`、`tests/test_agentic_agent.py` 覆盖了）。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/query 和 /api/query/agentic 路由测试。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_kb_manager
from ragify.api.main import app
from ragify.core.kb_manager import KBManager


class TestQueryRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.manager = KBManager(
            database_url=f"sqlite:///{self.tmp_dir}/test.db",
            vectorstore_dir=Path(self.tmp_dir) / "vectorstore",
        )
        self.manager.create("默认知识库")
        app.dependency_overrides[get_kb_manager] = lambda: self.manager
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_query_without_available_kb_returns_400(self):
        # 删掉 setUp 里创建的那个默认知识库，制造"没有可用知识库"的情况
        kbs = self.manager.list_all()
        self.manager.delete(kbs[0].id)

        res = self.client.post("/api/query", json={"query": "test"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("没有可用知识库", res.json()["detail"])

    @patch("ragify.api.routers.query.QueryPipeline")
    def test_query_reshapes_pipeline_result(self, mock_pipeline_cls):
        mock_doc = MagicMock()
        mock_doc.page_content = "内容"
        mock_doc.metadata = {"source": "a.txt", "file_type": "txt", "retrieval_score": 0.9}

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "response": "答案",
            "response_generated": True,
            "retrieved_documents": [mock_doc],
            "query_summary": {"documents_retrieved": 1},
        }
        mock_pipeline_cls.return_value = mock_pipeline

        res = self.client.post("/api/query", json={"query": "什么是RAG", "k": 3})

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["response"], "答案")
        self.assertEqual(body["retrieved_documents"][0]["metadata"]["source"], "a.txt")
        mock_pipeline.run.assert_called_once_with({
            "query": "什么是RAG", "k": 3, "score_threshold": None,
        })

    @patch("ragify.api.routers.query.AgenticRAG")
    def test_agentic_query_delegates_to_agent(self, mock_agent_cls):
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "response": "答案", "tool_calls": [], "sources": [], "iterations": 1,
        }
        mock_agent_cls.return_value = mock_agent

        res = self.client.post("/api/query/agentic", json={"query": "问题"})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["response"], "答案")
        mock_agent.run.assert_called_once_with("问题", chat_history=None)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_api_query -v 2>&1 | tail -15
```

- [ ] **Step 3: 写 `ragify/api/routers/query.py`**

```python
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import KB_LOCK, get_kb_manager, resolve_kb_path
from ..schemas import AgenticQueryRequest, QueryRequest
from ...agentic import AgenticRAG
from ...core.kb_manager import KBManager
from ...mcp import QueryPipeline

router = APIRouter()


@router.post("/api/query")
def query(body: QueryRequest, manager: KBManager = Depends(get_kb_manager)) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        pipeline = QueryPipeline()

    result = pipeline.run({
        "query": body.query,
        "k": body.k,
        "score_threshold": body.score_threshold,
    })

    retrieved = result.get("retrieved_documents", [])
    docs_out = []
    for doc in retrieved:
        docs_out.append({
            "page_content": doc.page_content,
            "metadata": {
                "source": doc.metadata.get("source", ""),
                "file_type": doc.metadata.get("file_type", ""),
                "retrieval_score": doc.metadata.get("retrieval_score", 0),
            },
        })

    return {
        "response": result.get("response", ""),
        "response_generated": result.get("response_generated", False),
        "retrieved_documents": docs_out,
        "query_summary": result.get("query_summary", {}),
    }


@router.post("/api/query/agentic")
def agentic_query(body: AgenticQueryRequest, manager: KBManager = Depends(get_kb_manager)) -> dict:
    # 整个 AgenticRAG 构造 + .run() 都在锁内——它的 retrieve_docs 工具在
    # run() 执行期间（不是构造时）才现读一次全局 vectorstore 配置，所以
    # 不能像 query() 那样提前把锁放掉。见本文档开头的并发说明。
    with KB_LOCK:
        if body.kb_id:
            try:
                resolve_kb_path(manager, body.kb_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        agent = AgenticRAG(kb_id=body.kb_id, max_iterations=body.max_iterations)
        result = agent.run(body.query, chat_history=body.chat_history)
    return result
```

- [ ] **Step 4: 在 `ragify/api/main.py` 里挂载这个 router**

```python
from fastapi import FastAPI

from .routers import health, kb, query

app = FastAPI(title="RAGify API")

app.include_router(kb.router)
app.include_router(query.router)
app.include_router(health.router)
```

- [ ] **Step 5: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_api_query -v
```

Expected: 3 个测试全部 `ok`。

- [ ] **Step 6: Commit**

```bash
git add ragify/api/routers/query.py ragify/api/main.py tests/test_api_query.py
git commit -m "feat: /api/query 和 /api/query/agentic 路由"
```

---

### Task 10: 文档/索引/分块管理路由

**Files:**
- Create: `ragify/api/routers/documents.py`
- Modify: `ragify/api/main.py`
- Test: `tests/test_api_documents.py`

- [ ] **Step 1: 写失败的测试 `tests/test_api_documents.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/index、/api/stats、/api/documents、/api/chunks 路由测试。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_kb_manager
from ragify.api.main import app
from ragify.core.kb_manager import KBManager


class TestDocumentsRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.manager = KBManager(
            database_url=f"sqlite:///{self.tmp_dir}/test.db",
            vectorstore_dir=Path(self.tmp_dir) / "vectorstore",
        )
        self.kb = self.manager.create("默认知识库")
        app.dependency_overrides[get_kb_manager] = lambda: self.manager
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch("ragify.api.routers.documents.IndexingPipeline")
    def test_index_with_directory_path(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"indexing_summary": {"total_documents_indexed": 2}}
        mock_pipeline_cls.return_value = mock_pipeline

        res = self.client.post("/api/index", json={
            "directory_path": "/some/dir", "kb_id": self.kb.id,
        })

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["indexing_summary"]["total_documents_indexed"], 2)
        called_payload = mock_pipeline.run.call_args[0][0]
        self.assertEqual(called_payload["directory_path"], "/some/dir")
        self.assertTrue(called_payload["clear_vectorstore"])

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_clear_index(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.request("DELETE", "/api/index", json={"kb_id": self.kb.id})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"success": True})
        mock_vsm.clear.assert_called_once()

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_get_stats(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.get_document_count.return_value = 5
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.get(f"/api/stats?kb_id={self.kb.id}")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["doc_count"], 5)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_list_documents(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.get_sources.return_value = [{"name": "a.txt", "source": "a.txt"}]
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.get(f"/api/documents?kb_id={self.kb.id}")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["total"], 1)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_delete_document(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.delete_by_source.return_value = 3
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.request("DELETE", "/api/documents", json={
            "kb_id": self.kb.id, "source": "nonexistent.txt",
        })

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["chunks_removed"], 3)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_list_chunks_requires_source(self, mock_vsm_cls):
        res = self.client.get(f"/api/chunks?kb_id={self.kb.id}")
        self.assertEqual(res.status_code, 422)  # FastAPI 校验 source 是必填 query 参数

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_update_chunk(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.update_chunk_content.return_value = True
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.put("/api/chunks", json={
            "kb_id": self.kb.id, "chunk_id": "c1", "content": "新内容",
        })

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"success": True})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_api_documents -v 2>&1 | tail -15
```

- [ ] **Step 3: 写 `ragify/api/routers/documents.py`**

```python
import os

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import KB_LOCK, PROJECT_ROOT, get_kb_manager, resolve_kb_path
from ..schemas import ClearIndexRequest, DeleteDocRequest, IndexRequest, UpdateChunkRequest
from ...core.kb_manager import KBManager
from ...core.vectorstores import VectorStoreManager
from ...config import get_config
from ...mcp import IndexingPipeline

router = APIRouter()


@router.post("/api/index")
def index_documents(body: IndexRequest, manager: KBManager = Depends(get_kb_manager)) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        pipeline = IndexingPipeline()

    payload: dict = {}
    if body.directory_path:
        payload["directory_path"] = body.directory_path
    elif body.file_paths:
        payload["file_paths"] = body.file_paths
    elif body.kb_id:
        kb_data_dir = os.path.join(PROJECT_ROOT, "data", body.kb_id)
        if os.path.isdir(kb_data_dir):
            payload["directory_path"] = kb_data_dir

    if body.clear_vectorstore is not None:
        payload["clear_vectorstore"] = body.clear_vectorstore
    elif payload.get("directory_path"):
        payload["clear_vectorstore"] = True

    result = pipeline.run(payload)
    return {"indexing_summary": result.get("indexing_summary", {})}


@router.delete("/api/index")
def clear_index(body: ClearIndexRequest, manager: KBManager = Depends(get_kb_manager)) -> dict:
    with KB_LOCK:
        if body.kb_id:
            try:
                resolve_kb_path(manager, body.kb_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    vm.clear()
    return {"success": True}


@router.get("/api/stats")
def get_stats(kb_id: str | None = None, manager: KBManager = Depends(get_kb_manager)) -> dict:
    with KB_LOCK:
        if kb_id:
            try:
                resolve_kb_path(manager, kb_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        cfg = get_config()
        vm = VectorStoreManager()
        store_type = cfg.get("vectorstore.type", "unknown")
        collection_name = cfg.get("vectorstore.collection_name", "")
        persist_directory = cfg.get("vectorstore.persist_directory", "")

    return {
        "store_type": store_type,
        "collection_name": collection_name,
        "persist_directory": persist_directory,
        "doc_count": vm.get_document_count(),
    }


@router.get("/api/documents")
def list_documents(kb_id: str | None = None, manager: KBManager = Depends(get_kb_manager)) -> dict:
    with KB_LOCK:
        if kb_id:
            try:
                resolve_kb_path(manager, kb_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    sources = vm.get_sources()
    return {"documents": sources, "total": len(sources)}


@router.delete("/api/documents")
def delete_document(body: DeleteDocRequest, manager: KBManager = Depends(get_kb_manager)) -> dict:
    source = body.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="缺少 source 参数")

    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    deleted = False
    candidates = [source, os.path.join(PROJECT_ROOT, "data", body.kb_id, os.path.basename(source))]
    for candidate in candidates:
        if os.path.isfile(candidate):
            os.remove(candidate)
            deleted = True

    removed = vm.delete_by_source(source)
    return {"success": True, "deleted": deleted, "chunks_removed": removed}


@router.get("/api/chunks")
def list_chunks(source: str, kb_id: str | None = None, manager: KBManager = Depends(get_kb_manager)) -> dict:
    if not source.strip():
        raise HTTPException(status_code=400, detail="缺少 source 参数")

    with KB_LOCK:
        if kb_id:
            try:
                resolve_kb_path(manager, kb_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    chunks = vm.get_chunks_by_source(source)
    return {"chunks": chunks, "total": len(chunks)}


@router.put("/api/chunks")
def update_chunk(body: UpdateChunkRequest, manager: KBManager = Depends(get_kb_manager)) -> dict:
    if not body.chunk_id:
        raise HTTPException(status_code=400, detail="缺少 chunk_id 参数")

    with KB_LOCK:
        if body.kb_id:
            try:
                resolve_kb_path(manager, body.kb_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    ok = vm.update_chunk_content(body.chunk_id, body.content)
    return {"success": ok}
```

- [ ] **Step 4: 在 `ragify/api/main.py` 里挂载这个 router**

```python
from fastapi import FastAPI

from .routers import documents, health, kb, query

app = FastAPI(title="RAGify API")

app.include_router(kb.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(health.router)
```

- [ ] **Step 5: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_api_documents -v
```

Expected: 7 个测试全部 `ok`。

- [ ] **Step 6: Commit**

```bash
git add ragify/api/routers/documents.py ragify/api/main.py tests/test_api_documents.py
git commit -m "feat: 文档/索引/分块管理路由"
```

---

### Task 11: 应用启动时自动迁移 `kbs.json`

**Files:**
- Modify: `ragify/api/main.py`
- Test: `tests/test_api_startup_migration.py`

- [ ] **Step 1: 写失败的测试 `tests/test_api_startup_migration.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 FastAPI 启动事件会调用一次 KBManager.migrate_json_if_needed()。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.main import app


class TestStartupMigration(unittest.TestCase):
    @patch("ragify.api.main.KBManager")
    def test_startup_runs_migration(self, mock_kb_manager_cls):
        mock_manager = mock_kb_manager_cls.return_value
        with TestClient(app):
            pass  # 进入/退出 with 块会触发 startup/shutdown 事件
        mock_manager.migrate_json_if_needed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_api_startup_migration -v 2>&1 | tail -15
```

Expected: 报错，因为 `ragify.api.main` 里还没有 `KBManager` 这个名字可供 patch。

- [ ] **Step 3: 在 `ragify/api/main.py` 里加启动事件**

把 `ragify/api/main.py` 改成：

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

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_api_startup_migration -v
```

Expected: 1 个测试 `ok`。

- [ ] **Step 5: 跑一遍全部 Python 测试，确认没有回归**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -15
```

Expected: 全部通过，无 FAILED/ERROR。

- [ ] **Step 6: Commit**

```bash
git add ragify/api/main.py tests/test_api_startup_migration.py
git commit -m "feat: FastAPI 启动时自动执行 kbs.json 迁移"
```

---

### Task 12: 前端 `callBackend` 封装

**Files:**
- Create: `frontend/src/lib/backend.ts`

- [ ] **Step 1: 写 `frontend/src/lib/backend.ts`**

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

- [ ] **Step 2: 类型检查**

```bash
cd frontend && npx tsc --noEmit --pretty false
```

Expected: 无输出（暂时没有任何文件引用 `callBackend`，纯类型检查通过）。

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/backend.ts
git commit -m "feat: 前端新增 callBackend fetch 封装，对接 FastAPI"
```

---

### Task 13: 迁移 KB 管理 + health 相关的 3 个 route.ts

**Files:**
- Modify: `frontend/src/app/api/health/route.ts`
- Modify: `frontend/src/app/api/knowledge-bases/route.ts`
- Modify: `frontend/src/app/api/knowledge-bases/[id]/route.ts`

- [ ] **Step 1: 整体替换 `frontend/src/app/api/health/route.ts`**

```typescript
import { NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function GET() {
  try {
    const result = await callBackend("/api/health", undefined, { method: "GET", timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json(
      { status: "degraded", error: String(e) },
      { status: 500 }
    );
  }
}
```

- [ ] **Step 2: 整体替换 `frontend/src/app/api/knowledge-bases/route.ts`**

```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function GET() {
  try {
    const result = await callBackend("/api/kb", undefined, { method: "GET", timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    if (!body.name || typeof body.name !== "string" || !body.name.trim()) {
      return NextResponse.json(
        { error: "知识库名称不能为空" },
        { status: 400 }
      );
    }
    const result = await callBackend("/api/kb", {
      name: body.name.trim(),
      description: typeof body.description === "string" ? body.description : "",
    }, { timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
```

- [ ] **Step 3: 整体替换 `frontend/src/app/api/knowledge-bases/[id]/route.ts`**

```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const result = await callBackend(`/api/kb/${id}`, undefined, { method: "DELETE", timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
```

- [ ] **Step 4: 类型检查**

```bash
cd frontend && npx tsc --noEmit --pretty false
```

Expected: 无输出。

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/api/health/route.ts frontend/src/app/api/knowledge-bases/
git commit -m "feat: KB 管理与 health 路由改为代理到 FastAPI"
```

---

### Task 14: 迁移查询相关的 2 个 route.ts

**Files:**
- Modify: `frontend/src/app/api/query/route.ts`
- Modify: `frontend/src/app/api/query/agentic/route.ts`

- [ ] **Step 1: 整体替换 `frontend/src/app/api/query/route.ts`**

```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    if (!body.query || typeof body.query !== "string") {
      return NextResponse.json(
        { error: "缺少 query 参数" },
        { status: 400 }
      );
    }

    const payload: Record<string, unknown> = {
      query: body.query,
    };
    if (body.k !== undefined) payload.k = Number(body.k);
    if (body.score_threshold !== undefined) {
      payload.score_threshold = Number(body.score_threshold);
    }
    if (body.kb_id) {
      payload.kb_id = body.kb_id;
    }

    const result = await callBackend("/api/query", payload, { timeout: 60_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
```

- [ ] **Step 2: 整体替换 `frontend/src/app/api/query/agentic/route.ts`**

```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    if (!body.query || typeof body.query !== "string") {
      return NextResponse.json(
        { error: "缺少 query 参数" },
        { status: 400 }
      );
    }

    const payload: Record<string, unknown> = {
      query: body.query,
    };
    if (body.kb_id) payload.kb_id = body.kb_id;
    if (body.chat_history) payload.chat_history = body.chat_history;
    if (body.max_iterations) payload.max_iterations = Number(body.max_iterations);

    const result = await callBackend("/api/query/agentic", payload, { timeout: 120_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
```

- [ ] **Step 3: 类型检查**

```bash
cd frontend && npx tsc --noEmit --pretty false
```

Expected: 无输出。

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/api/query/
git commit -m "feat: 标准/Agentic 问答路由改为代理到 FastAPI"
```

---

### Task 15: 迁移索引/统计/文档/分块相关的 4 个 route.ts

**Files:**
- Modify: `frontend/src/app/api/index/route.ts`
- Modify: `frontend/src/app/api/stats/route.ts`
- Modify: `frontend/src/app/api/documents/route.ts`
- Modify: `frontend/src/app/api/chunks/route.ts`

- [ ] **Step 1: 整体替换 `frontend/src/app/api/index/route.ts`**

```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const payload: Record<string, unknown> = {};

    if (body.file_paths) {
      payload.file_paths = body.file_paths;
    } else if (body.directory_path) {
      payload.directory_path = body.directory_path;
    }
    if (body.clear_vectorstore !== undefined) {
      payload.clear_vectorstore = Boolean(body.clear_vectorstore);
    }
    if (body.kb_id) {
      payload.kb_id = body.kb_id;
    }

    const result = await callBackend("/api/index", payload, { timeout: 120_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const result = await callBackend("/api/index", { kb_id: body.kb_id }, { method: "DELETE", timeout: 30_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
```

- [ ] **Step 2: 整体替换 `frontend/src/app/api/stats/route.ts`**

```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function GET(req: NextRequest) {
  try {
    const kbId = req.nextUrl.searchParams.get("kb_id");
    const params = new URLSearchParams();
    if (kbId) params.set("kb_id", kbId);
    const query = params.toString() ? `?${params.toString()}` : "";
    const result = await callBackend(`/api/stats${query}`, undefined, { method: "GET", timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
```

- [ ] **Step 3: 整体替换 `frontend/src/app/api/documents/route.ts`**

```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function GET(req: NextRequest) {
  try {
    const kbId = req.nextUrl.searchParams.get("kb_id");
    const params = new URLSearchParams();
    if (kbId) params.set("kb_id", kbId);
    const query = params.toString() ? `?${params.toString()}` : "";
    const result = await callBackend(`/api/documents${query}`, undefined, { method: "GET", timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const { kb_id, source } = await req.json();
    if (!kb_id || !source) {
      return NextResponse.json({ error: "缺少 kb_id 或 source 参数" }, { status: 400 });
    }
    const result = await callBackend("/api/documents", { kb_id, source }, { method: "DELETE", timeout: 60_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
```

- [ ] **Step 4: 整体替换 `frontend/src/app/api/chunks/route.ts`**

```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function GET(req: NextRequest) {
  try {
    const source = req.nextUrl.searchParams.get("source");
    const kbId = req.nextUrl.searchParams.get("kb_id");
    if (!source) {
      return NextResponse.json({ error: "缺少 source 参数" }, { status: 400 });
    }
    const params = new URLSearchParams({ source });
    if (kbId) params.set("kb_id", kbId);
    const result = await callBackend(`/api/chunks?${params.toString()}`, undefined, { method: "GET", timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}

export async function PUT(req: NextRequest) {
  try {
    const { kb_id, chunk_id, content } = await req.json();
    if (!chunk_id) {
      return NextResponse.json({ error: "缺少 chunk_id 参数" }, { status: 400 });
    }
    const result = await callBackend("/api/chunks", { kb_id, chunk_id, content }, { method: "PUT", timeout: 30_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
```

- [ ] **Step 5: 类型检查和 lint**

```bash
cd frontend && npx tsc --noEmit --pretty false && npx eslint src/app/api
```

Expected: 都无输出/无错误。

- [ ] **Step 6: Commit**

```bash
git add frontend/src/app/api/index/ frontend/src/app/api/stats/ frontend/src/app/api/documents/ frontend/src/app/api/chunks/
git commit -m "feat: 索引/统计/文档/分块路由改为代理到 FastAPI"
```

---

### Task 16: 删除 bridge.py 子进程模式的遗留代码

**Files:**
- Delete: `frontend/scripts/bridge.py`
- Delete: `frontend/src/lib/bridge.ts`

- [ ] **Step 1: 确认没有任何文件还在 import 这两个文件**

```bash
grep -rln "lib/bridge\|scripts/bridge" frontend/src/ 2>/dev/null
```

Expected: 无输出（Task 13-15 已经把所有 route.ts 换成了 `callBackend`）。

- [ ] **Step 2: 删除文件**

```bash
rm frontend/scripts/bridge.py frontend/src/lib/bridge.ts
rmdir frontend/scripts 2>/dev/null || true
```

- [ ] **Step 3: 类型检查确认没有破坏任何引用**

```bash
cd frontend && npx tsc --noEmit --pretty false
```

Expected: 无输出。

- [ ] **Step 4: Commit**

```bash
git add -A frontend/scripts frontend/src/lib/bridge.ts
git commit -m "chore: 删除已废弃的 bridge.py 子进程桥接代码"
```

---

### Task 17: 改造 `start.sh` 同时管理 uvicorn 和 Next.js

**Files:**
- Modify: `start.sh`（整体重写）

- [ ] **Step 1: 用下面的内容整体替换 `start.sh`**

```bash
#!/usr/bin/env bash
# 一键启动/停止/重启 RAGify（API + 前端两个后台进程）。
#
# 用法:
#   ./start.sh            # 启动（若已在运行，先自动停止旧进程再启动，幂等）
#   ./start.sh start      # 同上
#   ./start.sh restart    # 显式重启
#   ./start.sh stop       # 停止
#   ./start.sh status     # 查看运行状态
#
# 日志: tail -f .run/api.log 或 .run/frontend.log
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
VENV_DIR="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
RUN_DIR="$PROJECT_ROOT/.run"

PID_FILE="$RUN_DIR/frontend.pid"
LOG_FILE="$RUN_DIR/frontend.log"
PORT="${PORT:-3000}"

API_PID_FILE="$RUN_DIR/api.pid"
API_LOG_FILE="$RUN_DIR/api.log"
API_PORT="${API_PORT:-8000}"

info() { printf '\033[1;34m[start]\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[start]\033[0m %s\n' "$1"; }
err()  { printf '\033[1;31m[start]\033[0m %s\n' "$1" >&2; }

mkdir -p "$RUN_DIR"

is_service_running() {
  [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null
}

stop_service() {
  local label="$1" pid_file="$2" port="$3"
  if is_service_running "$pid_file"; then
    local pid
    pid="$(cat "$pid_file")"
    info "停止${label} (PID $pid) ..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"

  if command -v lsof >/dev/null 2>&1; then
    local port_pids
    port_pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
    if [ -n "$port_pids" ]; then
      warn "端口 ${port} 仍被占用 (PID: $port_pids)，一并终止"
      kill $port_pids 2>/dev/null || true
    fi
  fi
}

stop_all() {
  stop_service "前端服务" "$PID_FILE" "$PORT"
  stop_service "API 服务" "$API_PID_FILE" "$API_PORT"
}

status_all() {
  if is_service_running "$PID_FILE"; then
    info "前端服务运行中 (PID $(cat "$PID_FILE"))，http://localhost:${PORT}"
  else
    info "前端服务未运行"
  fi
  if is_service_running "$API_PID_FILE"; then
    info "API 服务运行中 (PID $(cat "$API_PID_FILE"))，http://localhost:${API_PORT}"
  else
    info "API 服务未运行"
  fi
}

ensure_backend_ready() {
  # 1. Python 环境
  if ! command -v uv >/dev/null 2>&1; then
    err "未找到 uv，请先安装：pip install uv"
    exit 1
  fi

  if [ ! -x "$VENV_PYTHON" ]; then
    info "未发现虚拟环境，正在创建 .venv ..."
    (cd "$PROJECT_ROOT" && uv venv)
  fi

  if ! "$VENV_PYTHON" -c "import ragify" >/dev/null 2>&1; then
    info "安装/同步 Python 依赖 ..."
    (cd "$PROJECT_ROOT" && uv pip install -e .)
  fi

  # 2. 配置文件
  if [ ! -f "$PROJECT_ROOT/config/config.yaml" ]; then
    if [ -f "$PROJECT_ROOT/config/config.yaml.example" ]; then
      warn "config/config.yaml 不存在，从示例文件复制一份"
      cp "$PROJECT_ROOT/config/config.yaml.example" "$PROJECT_ROOT/config/config.yaml"
    else
      err "缺少 config/config.yaml，请先手动创建配置文件"
      exit 1
    fi
  fi

  if [ ! -f "$PROJECT_ROOT/.env" ]; then
    warn ".env 不存在，请确认已配置所需的 API Key（如 DASHSCOPE_API_KEY）"
  fi

  # 3. 前端依赖
  if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    info "安装前端依赖 ..."
    (cd "$FRONTEND_DIR" && npm install)
  fi

  # 4. 数据库迁移
  info "执行数据库迁移 ..."
  (cd "$PROJECT_ROOT" && "$VENV_PYTHON" -m alembic upgrade head)
}

start_api() {
  info "启动 API 服务（后台运行，http://localhost:${API_PORT}）..."
  (
    cd "$PROJECT_ROOT"
    nohup "$VENV_PYTHON" -m uvicorn ragify.api.main:app --host 0.0.0.0 --port "$API_PORT" >"$API_LOG_FILE" 2>&1 &
    echo $! > "$API_PID_FILE"
  )
  sleep 1
  if is_service_running "$API_PID_FILE"; then
    info "API 已启动 (PID $(cat "$API_PID_FILE"))，日志: tail -f $API_LOG_FILE"
  else
    err "API 启动失败，请查看日志: $API_LOG_FILE"
    exit 1
  fi
}

start_frontend() {
  info "启动前端开发服务器（后台运行，http://localhost:${PORT}）..."
  (
    cd "$FRONTEND_DIR"
    nohup npm run dev -- -p "$PORT" >"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
  )
  sleep 1
  if is_service_running "$PID_FILE"; then
    info "前端已启动 (PID $(cat "$PID_FILE"))，日志: tail -f $LOG_FILE"
  else
    err "前端启动失败，请查看日志: $LOG_FILE"
    exit 1
  fi
}

start_all() {
  stop_all
  ensure_backend_ready
  start_api
  start_frontend
}

ACTION="${1:-start}"
case "$ACTION" in
  start)
    start_all
    ;;
  stop)
    stop_all
    info "已停止"
    ;;
  restart)
    start_all
    ;;
  status)
    status_all
    ;;
  *)
    err "未知参数: ${ACTION}（支持 start|stop|restart|status）"
    exit 1
    ;;
esac
```

- [ ] **Step 2: 语法检查**

```bash
bash -n start.sh && echo "syntax ok"
```

Expected: `syntax ok`。

- [ ] **Step 3: 实测启停一整轮**

```bash
chmod +x start.sh
./start.sh start
sleep 3
./start.sh status
curl -s http://localhost:8000/api/health
echo
curl -s -o /dev/null -w "frontend HTTP %{http_code}\n" http://localhost:3000
./start.sh restart
sleep 3
./start.sh status
./start.sh stop
./start.sh status
```

Expected:
- 第一次 `status` 显示两个服务都"运行中"
- `curl .../api/health` 返回 JSON（`status: healthy` 等字段）
- 前端返回 `HTTP 200`
- `restart` 之后 PID 变化但两个服务仍"运行中"
- 最终 `stop` 之后 `status` 显示两个服务都"未运行"

- [ ] **Step 4: Commit**

```bash
git add start.sh
git commit -m "feat: start.sh 同时管理 API(uvicorn) 与前端两个后台进程"
```

---

### Task 18: 端到端功能验证 + 全量测试 + 收尾

**Files:** 无新文件，只做验证

- [ ] **Step 1: 启动完整服务栈**

```bash
./start.sh start
sleep 3
```

- [ ] **Step 2: 用 browse 工具走一遍真实界面，逐项核对跟迁移前行为一致**

用 `browse` 技能依次：
1. `goto http://localhost:3000` — 仪表盘应正常显示统计卡片（索引文档数、向量库类型等）
2. `goto http://localhost:3000/knowledge-base` — 知识库列表正常显示，新建一个知识库测试
3. 上传一个文档并点击索引，确认索引成功、文档数增加
4. `goto http://localhost:3000/qa` — 标准模式下对刚索引的文档提问，确认能拿到回答和来源
5. 切到 Agentic 模式提问，确认推理时间线正常显示、有回答
6. 删除刚才新建的测试知识库，确认删除成功

Expected: 每一步都跟 Phase 1 开始之前（子进程 bridge.py 时代）的行为完全一致，没有报错、没有 500。

- [ ] **Step 3: 检查两个服务的日志确认没有异常报错**

```bash
tail -50 .run/api.log
tail -50 .run/frontend.log
```

Expected: 没有 Python traceback、没有未捕获的异常堆栈（HMR/webpack 的常规日志除外）。

- [ ] **Step 4: 跑全量 Python 测试套件**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -20
```

Expected: 全部通过（现有 61 个 + 本计划新增的 kb_manager/api_health/api_kb/api_query/api_documents/api_startup_migration 测试，总数以实际为准），无 FAILED/ERROR。

- [ ] **Step 5: 前端类型检查和 lint**

```bash
cd frontend && npx tsc --noEmit --pretty false && npx eslint src/
```

Expected: 都无输出/无错误。

- [ ] **Step 6: 清理测试过程中产生的临时文件**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify
rm -rf test_vectorstore
./start.sh stop
```

- [ ] **Step 7: 最终确认 git 状态干净、所有改动都已提交**

```bash
git status --short
git log --oneline -20
```

Expected: 工作区干净（`git status --short` 无输出），能看到本计划每个任务对应的 commit。
