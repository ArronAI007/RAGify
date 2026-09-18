# Phase 4: 数据隔离迁移 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `KnowledgeBase` 加 `tenant_id` 归属，迁移现有数据，给 `/api/kb`、`/api/query`、`/api/documents` 加登录门禁并把 URL 改成显式带 `tenant_id`，同步更新前端代理路由，补一个最粗粝的登录页让浏览器端能拿到登录态，并给 MCP 服务入口设计对应的租户识别机制。

**Architecture:** `ragify/core/kb_manager.py` 的 `KBManager` 全部方法加 `tenant_id` 参数（`get`/`delete` 做真正的归属校验，不是简单过滤）；两个新的幂等启动迁移函数（`migrate_tenant_id_if_needed`/`migrate_vectorstore_layout_if_needed`）跟在 Phase 1/3 已有的迁移函数后面依次跑；`ragify/api/routers/{kb,query,documents}.py` 的路由全部挪到 `/api/tenants/{tenant_id}/...` 下面，用 Phase 3 已有的 `require_membership`/`require_role` 依赖做权限门禁；前端 7 个代理路由改成读 cookie、查当前用户的工作区、转发到新 URL；新增一个最简登录页 + `middleware.ts` 补上登录入口；`ragify/mcp_server/server.py`（Phase 1 的独立 stdio 入口，不走 HTTP）启动时读 `RAGIFY_MCP_TOKEN` 环境变量（复用已有的登录 JWT），解出 user_id 再解出这个用户的第一个工作区，作为整个 MCP 会话生命周期内固定使用的 `tenant_id`。

**Tech Stack:** 复用 Phase 1-3 已经装好的 FastAPI/SQLAlchemy/Alembic/PyJWT 技术栈，不新增任何 pyproject.toml 依赖；前端不新增依赖。

---

### Task 1: 数据模型 + Alembic 迁移

**Files:**
- Modify: `ragify/db/models.py`
- Create: `alembic/versions/xxxx_add_tenant_id_to_knowledge_bases.py`

- [ ] **Step 1: 修改 `ragify/db/models.py` 的 `KnowledgeBaseRow`**

当前内容：
```python
class KnowledgeBaseRow(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False, unique=True)
    description: Mapped[str] = mapped_column(nullable=False, default="")
    created_at: Mapped[str] = mapped_column(nullable=False)
```

改成：
```python
class KnowledgeBaseRow(Base):
    """Phase 4：加 tenant_id 归属，名称唯一性从全局收窄成工作区内唯一。
    tenant_id 数据库层面允许 NULL——不是业务上允许知识库没有归属，而是迁移
    窗口期需要：Alembic 迁移只管加列，不知道"默认工作区"的 id（那是应用
    启动时才创建的），实际回填由 KBManager.migrate_tenant_id_if_needed()
    在应用启动时完成。回填完成、应用真正开始对外提供服务之前，不会有任何
    一行知识库停留在 tenant_id IS NULL 的状态。
    """
    __tablename__ = "knowledge_bases"
    __table_args__ = (UniqueConstraint("tenant_id", "name"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(nullable=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False, default="")
    created_at: Mapped[str] = mapped_column(nullable=False)
```

（`UniqueConstraint` 已经在文件顶部 import 过，不需要改 import。）

- [ ] **Step 2: 生成迁移**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify && .venv/bin/alembic revision --autogenerate -m "add tenant_id to knowledge_bases"
```

Expected: 在 `alembic/versions/` 下生成一个新文件，`down_revision` 自动指向 `b50496f6382f`（当前最新的 revision，即 tenant 表迁移）。

- [ ] **Step 3: 检查生成的迁移内容**

打开生成的文件，确认 `upgrade()`/`downgrade()` 内容等价于：

```python
def upgrade() -> None:
    with op.batch_alter_table("knowledge_bases", schema=None) as batch_op:
        batch_op.add_column(sa.Column("tenant_id", sa.String(), nullable=True))
        batch_op.drop_constraint("knowledge_bases_name_key", type_="unique")
        batch_op.create_unique_constraint(
            "uq_knowledge_bases_tenant_id_name", ["tenant_id", "name"]
        )


def downgrade() -> None:
    with op.batch_alter_table("knowledge_bases", schema=None) as batch_op:
        batch_op.drop_constraint("uq_knowledge_bases_tenant_id_name", type_="unique")
        batch_op.create_unique_constraint("knowledge_bases_name_key", ["name"])
        batch_op.drop_column("tenant_id")
```

用 `batch_alter_table`（Alembic 对 SQLite 的标准做法——SQLite 不支持大部分 `ALTER TABLE` 变体，Alembic 会自动建临时表、拷数据、改名字来模拟）。如果 autogenerate 生成的约束名字跟示例不完全一样（比如 SQLite 自动生成的旧唯一约束名字不是 `knowledge_bases_name_key`），**不要**手改成示例里的名字硬凑——改成 autogenerate 实际检测到的名字，用 `alembic downgrade -1` 后 `alembic upgrade head` 跑一遍确认真实生效，而不是假设名字。如果 autogenerate 没能自动检测出旧的唯一约束需要删除（SQLite 的隐式索引经常检测不全），手动在 `upgrade()`/`downgrade()` 里按上面的结构补全。

- [ ] **Step 4: 跑迁移，验证结构变化**

```bash
.venv/bin/alembic upgrade head
.venv/bin/python -c "
import sqlite3
conn = sqlite3.connect('vectorstore/ragify.db')
print(conn.execute('PRAGMA table_info(knowledge_bases)').fetchall())
print(conn.execute(\"SELECT sql FROM sqlite_master WHERE type='table' AND name='knowledge_bases'\").fetchone())
"
```

Expected: 第一行输出里出现 `tenant_id`（`notnull` 值为 0，即允许 NULL）；第二行输出的 `CREATE TABLE` 语句里没有单独的 `name UNIQUE`，能看到一个包含 `tenant_id, name` 两列的 `UNIQUE` 约束。

- [ ] **Step 5: 清理这次手动验证生成的数据库文件**

```bash
rm -f vectorstore/ragify.db
```

- [ ] **Step 6: Commit**

```bash
git add ragify/db/models.py alembic/versions/
git commit -m "feat: KnowledgeBaseRow 新增 tenant_id 归属，名称唯一性改成工作区内唯一"
```

---

### Task 2: `KBManager` — create/list_all/get/delete 加 tenant_id（含跨租户隔离测试）

**Files:**
- Modify: `ragify/core/kb_manager.py`
- Modify: `tests/test_kb_manager.py`

- [ ] **Step 1: 改写 `tests/test_kb_manager.py`**

当前文件的测试全部假设知识库全局唯一、没有 `tenant_id`。改成：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KBManager 测试（DB 驱动版，Phase 4 加了 tenant_id 归属）
验证 KB 的增删查、工作区内名称去重、跨工作区隔离，以及从 kbs.json / 旧版
扁平索引迁移进数据库的逻辑。
"""

import json
import shutil
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.kb_manager import KBManager
from ragify.db.models import Base, KnowledgeBaseRow
from ragify.db.session import get_engine, get_session

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"


class TestKBManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        self.vectorstore_dir = Path(self.tmp_dir) / "vectorstore"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = KBManager(database_url=self.db_url, vectorstore_dir=self.vectorstore_dir)

    def tearDown(self):
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_and_get(self):
        kb = self.manager.create("测试知识库", "描述", TENANT_A)
        self.assertTrue(kb.id)
        self.assertEqual(kb.tenant_id, TENANT_A)
        fetched = self.manager.get(kb.id, TENANT_A)
        self.assertEqual(fetched.name, "测试知识库")
        self.assertEqual(fetched.description, "描述")

    def test_create_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.manager.create("   ", "", TENANT_A)

    def test_create_duplicate_name_within_same_tenant_raises(self):
        self.manager.create("重复名称", "", TENANT_A)
        with self.assertRaises(ValueError):
            self.manager.create("重复名称", "", TENANT_A)

    def test_create_same_name_in_different_tenants_both_succeed(self):
        kb_a = self.manager.create("同名知识库", "", TENANT_A)
        kb_b = self.manager.create("同名知识库", "", TENANT_B)
        self.assertNotEqual(kb_a.id, kb_b.id)
        self.assertEqual(self.manager.get(kb_a.id, TENANT_A).name, "同名知识库")
        self.assertEqual(self.manager.get(kb_b.id, TENANT_B).name, "同名知识库")

    def test_create_race_condition_raises_value_error(self):
        """Exercises the `except IntegrityError` branch in create(), not the
        in-Python pre-check (that path is already covered by
        test_create_duplicate_name_within_same_tenant_raises).

        Same technique as before Phase 4: patch Session.commit so that the
        *first* time create() calls it, we commit a colliding
        (tenant_id, name) row through a *separate* session first, simulating
        another process's create() call finishing in that exact window.
        """
        name = "并发冲突名称"
        original_commit = Session.commit
        state = {"injected": False}

        def racing_commit(session_self, *args, **kwargs):
            if not state["injected"]:
                state["injected"] = True
                other_session = get_session(self.db_url)
                try:
                    other_session.add(KnowledgeBaseRow(
                        id=uuid.uuid4().hex[:12], tenant_id=TENANT_A, name=name, description="",
                        created_at=datetime.now(timezone.utc).isoformat(),
                    ))
                    original_commit(other_session)
                finally:
                    other_session.close()
            return original_commit(session_self, *args, **kwargs)

        with patch.object(Session, "commit", racing_commit):
            with self.assertRaises(ValueError):
                self.manager.create(name, "", TENANT_A)

    def test_list_all_empty(self):
        self.assertEqual(self.manager.list_all(TENANT_A), [])

    def test_list_all_only_returns_own_tenant(self):
        self.manager.create("A的知识库", "", TENANT_A)
        self.manager.create("B的知识库", "", TENANT_B)

        kbs_a = self.manager.list_all(TENANT_A)
        self.assertEqual(len(kbs_a), 1)
        self.assertEqual(kbs_a[0].name, "A的知识库")

        kbs_b = self.manager.list_all(TENANT_B)
        self.assertEqual(len(kbs_b), 1)
        self.assertEqual(kbs_b[0].name, "B的知识库")

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.manager.get("does-not-exist", TENANT_A))

    def test_get_with_wrong_tenant_returns_none(self):
        """跨租户隔离的核心断言：即使 kb_id 是真实存在的（比如从别的工作区
        的旧链接、日志里泄露出来），用别的 tenant_id 去 get 也必须表现得
        跟"这个 kb_id 根本不存在"完全一样——不能因为 kb_id 本身合法就泄露
        任何信息（哪怕只是 404 vs 拿到数据的区别）。"""
        kb = self.manager.create("A的知识库", "", TENANT_A)
        self.assertIsNone(self.manager.get(kb.id, TENANT_B))
        self.assertIsNotNone(self.manager.get(kb.id, TENANT_A))

    def test_delete_removes_kb_and_directory(self):
        kb = self.manager.create("待删除", "", TENANT_A)
        kb_dir = Path(self.manager.get_persist_dir(TENANT_A, kb.id))
        self.assertTrue(kb_dir.exists())

        ok = self.manager.delete(kb.id, TENANT_A)

        self.assertTrue(ok)
        self.assertIsNone(self.manager.get(kb.id, TENANT_A))
        self.assertFalse(kb_dir.exists())

    def test_delete_missing_returns_false(self):
        self.assertFalse(self.manager.delete("does-not-exist", TENANT_A))

    def test_delete_with_wrong_tenant_returns_false_and_does_not_delete(self):
        """同 test_get_with_wrong_tenant_returns_none 的隔离要求：用别的
        tenant_id 删不掉别人的知识库，行为跟"不存在"一样，且知识库本身
        必须还在。"""
        kb = self.manager.create("A的知识库", "", TENANT_A)
        ok = self.manager.delete(kb.id, TENANT_B)
        self.assertFalse(ok)
        self.assertIsNotNone(self.manager.get(kb.id, TENANT_A))

    def test_migrate_json_if_needed_imports_existing_file(self):
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        kbs_file = self.vectorstore_dir / "kbs.json"
        kbs_file.write_text(json.dumps({"kbs": [
            {"id": "legacy1", "name": "旧知识库", "description": "", "created_at": "2024-01-01T00:00:00"}
        ]}), encoding="utf-8")

        migrated = self.manager.migrate_json_if_needed()

        self.assertTrue(migrated)
        # 迁移进来的行还没有 tenant_id（那是 migrate_tenant_id_if_needed 的
        # 职责，Task 3 才会加），这里用底层查询确认行本身确实进了库。
        with self.manager._session() as session:
            row = session.get(KnowledgeBaseRow, "legacy1")
            self.assertIsNotNone(row)
            self.assertEqual(row.name, "旧知识库")
            self.assertIsNone(row.tenant_id)
        self.assertFalse(kbs_file.exists())
        self.assertTrue((self.vectorstore_dir / "kbs.json.migrated").exists())

    def test_migrate_json_if_needed_noop_when_db_has_rows(self):
        self.manager.create("已有数据", "", TENANT_A)
        self.assertFalse(self.manager.migrate_json_if_needed())

    def test_migrate_json_if_needed_noop_when_nothing_to_migrate(self):
        self.assertFalse(self.manager.migrate_json_if_needed())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_kb_manager -v 2>&1 | tail -25
```

Expected: 大量 `TypeError`（现有 `create`/`list_all`/`get`/`delete` 都还不接受 `tenant_id` 参数）。

- [ ] **Step 3: 改写 `ragify/core/kb_manager.py` 的 `KnowledgeBase` dataclass 和四个方法**

`KnowledgeBase` dataclass 改成：
```python
@dataclass
class KnowledgeBase:
    id: str
    tenant_id: str | None
    name: str
    description: str
    created_at: str
```

（`tenant_id: str | None` 而不是 `str`——迁移窗口期、以及 `migrate_json_if_needed` 刚导入但还没跑 `migrate_tenant_id_if_needed` 的那一小段时间里，行确实可能没有 tenant_id，dataclass 要能诚实表达这个状态，不能假装总是有。）

`migrate_json_if_needed` 里所有构造 `KnowledgeBase(...)`/`KnowledgeBaseRow(...)` 的地方补上 `tenant_id=None`（不改变这个方法本身的返回值/幂等判断逻辑，只是因为 dataclass/row 都多了一个字段）：

```python
    def migrate_json_if_needed(self) -> bool:
        with self._session() as session:
            has_rows = session.query(KnowledgeBaseRow).first() is not None
        if has_rows:
            return False

        kbs_file = self._kbs_json_file
        if kbs_file.exists():
            data = json.loads(kbs_file.read_text(encoding="utf-8"))
            with self._session() as session:
                for item in data.get("kbs", []):
                    session.add(KnowledgeBaseRow(
                        id=item["id"],
                        tenant_id=None,
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
                    tenant_id=None,
                    name="默认知识库",
                    description="迁移自旧版本数据",
                    created_at=datetime.now(timezone.utc).isoformat(),
                ))
                session.commit()
            logger.info("已迁移旧索引到 KB '默认知识库' (%s)", kb_id)
            return True

        return False
```

（注意：`self.list_all()` 原来这里没有 `tenant_id` 参数就能调，现在 `list_all` 要求 `tenant_id` 了，所以这里改成直接查 `session.query(KnowledgeBaseRow).first() is not None`——这个方法要检查的是"数据库里是否已经有任何一行"，跟租户完全无关，语义不变，只是不能再借用即将变成租户过滤的 `list_all()` 来做这件事了。）

`create`/`delete`/`list_all`/`get` 改成：

```python
    def create(self, name: str, description: str, tenant_id: str) -> KnowledgeBase:
        name = name.strip()
        if not name:
            raise ValueError("知识库名称不能为空")

        with self._session() as session:
            existing = {
                row.name.lower() for row in
                session.query(KnowledgeBaseRow).filter(KnowledgeBaseRow.tenant_id == tenant_id).all()
            }
            if name.lower() in existing:
                raise ValueError(f"知识库 '{name}' 已存在")

            kb_id = uuid.uuid4().hex[:12]
            created_at = datetime.now(timezone.utc).isoformat()
            description = description.strip()
            session.add(KnowledgeBaseRow(
                id=kb_id, tenant_id=tenant_id, name=name, description=description, created_at=created_at,
            ))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                raise ValueError(f"知识库 '{name}' 已存在")
            result = KnowledgeBase(
                id=kb_id, tenant_id=tenant_id, name=name, description=description, created_at=created_at,
            )

        kb_dir = self.vectorstore_dir / tenant_id / kb_id
        kb_dir.mkdir(parents=True, exist_ok=True)
        logger.info("创建知识库 '%s' (%s)，工作区 %s", name, kb_id, tenant_id)
        return result

    def delete(self, kb_id: str, tenant_id: str) -> bool:
        with self._session() as session:
            row = session.get(KnowledgeBaseRow, kb_id)
            if row is None or row.tenant_id != tenant_id:
                return False
            name = row.name
            session.delete(row)
            session.commit()

        kb_dir = self.vectorstore_dir / tenant_id / kb_id
        if kb_dir.exists():
            shutil.rmtree(str(kb_dir))
        logger.info("删除知识库 '%s' (%s)，工作区 %s", name, kb_id, tenant_id)
        return True

    def list_all(self, tenant_id: str) -> list[KnowledgeBase]:
        with self._session() as session:
            rows = session.query(KnowledgeBaseRow).filter(KnowledgeBaseRow.tenant_id == tenant_id).all()
            return [
                KnowledgeBase(
                    id=r.id, tenant_id=r.tenant_id, name=r.name,
                    description=r.description, created_at=r.created_at,
                )
                for r in rows
            ]

    def get(self, kb_id: str, tenant_id: str) -> KnowledgeBase | None:
        with self._session() as session:
            row = session.get(KnowledgeBaseRow, kb_id)
            if row is None or row.tenant_id != tenant_id:
                return None
            return KnowledgeBase(
                id=row.id, tenant_id=row.tenant_id, name=row.name,
                description=row.description, created_at=row.created_at,
            )
```

（`get_persist_dir` 保持暂时不动——Step 3 只改这四个方法，`get_persist_dir` 在 Task 3 才改成分层路径。`create`/`delete` 里操作磁盘目录的两行提前换成 `self.vectorstore_dir / tenant_id / kb_id`，跟 Task 3 要交付的分层布局保持一致，避免同一个方法被两个任务改两次。）

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_kb_manager -v
```

Expected: 16 个测试全部 `ok`。

- [ ] **Step 5: Commit**

```bash
git add ragify/core/kb_manager.py tests/test_kb_manager.py
git commit -m "feat: KBManager 的 create/list_all/get/delete 加 tenant_id，get/delete 校验归属实现真正隔离"
```

---

### Task 3: `KBManager.get_persist_dir` 分层路径 + `migrate_tenant_id_if_needed`

**Files:**
- Modify: `ragify/core/tenant_manager.py`
- Modify: `ragify/core/kb_manager.py`
- Modify: `tests/test_tenant_manager.py`
- Modify: `tests/test_kb_manager.py`

- [ ] **Step 1: 在 `tests/test_tenant_manager.py` 的 `TestTenantManager` 类末尾（`_add_member` 辅助方法之前）追加失败的测试**

```python

    def test_get_earliest_tenant(self):
        t1 = self.manager.create_tenant("工作区1", "user-1")
        self._insert_user("user-2", "2024-02-01T00:00:00+00:00")
        t2 = self.manager.create_tenant("工作区2", "user-2")
        earliest = self.manager.get_earliest_tenant()
        self.assertEqual(earliest.id, t1.id)

    def test_get_earliest_tenant_none_when_no_tenants(self):
        self.assertIsNone(self.manager.get_earliest_tenant())
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_tenant_manager -v 2>&1 | tail -10
```

Expected: `AttributeError: 'TenantManager' object has no attribute 'get_earliest_tenant'`。

- [ ] **Step 3: 在 `ragify/core/tenant_manager.py` 的 `TenantManager` 类末尾（`migrate_default_tenant_if_needed` 之后）追加**

```python

    def get_earliest_tenant(self) -> Tenant | None:
        with self._session() as session:
            row = session.query(TenantRow).order_by(TenantRow.created_at.asc()).first()
            if row is None:
                return None
            return Tenant(id=row.id, name=row.name, created_at=row.created_at)
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_tenant_manager -v
```

Expected: 28 个测试全部 `ok`。

- [ ] **Step 5: Commit（先单独提交这个小方法，再继续 KBManager 那部分）**

```bash
git add ragify/core/tenant_manager.py tests/test_tenant_manager.py
git commit -m "feat: TenantManager 新增 get_earliest_tenant（供 KBManager 的 tenant_id 回填迁移使用）"
```

- [ ] **Step 6: 在 `tests/test_kb_manager.py` 顶部 import 里加 `TenantManager`，并在文件末尾（`test_migrate_json_if_needed_noop_when_nothing_to_migrate` 之后，`if __name__ ==` 之前）追加失败的测试**

顶部 import 加一行：
```python
from ragify.core.tenant_manager import TenantManager
```

追加：
```python

    def test_get_persist_dir_is_nested_by_tenant(self):
        path = self.manager.get_persist_dir(TENANT_A, "some-kb-id")
        self.assertEqual(Path(path), self.vectorstore_dir / TENANT_A / "some-kb-id")

    def test_migrate_tenant_id_backfills_existing_rows(self):
        # 模拟 Task 1 迁移窗口期的状态：先插入一行没有 tenant_id 的知识库
        # （比如 migrate_json_if_needed 刚导入、还没回填的状态），再建一个
        # 真实的工作区，验证回填能把这行认领过去。
        with self.manager._session() as session:
            session.add(KnowledgeBaseRow(
                id="legacy-kb", tenant_id=None, name="旧知识库", description="",
                created_at="2024-01-01T00:00:00+00:00",
            ))
            session.commit()

        tenant_manager = TenantManager(database_url=self.db_url)
        tenant_manager.create_tenant("默认工作区", "user-1")

        migrated = self.manager.migrate_tenant_id_if_needed(tenant_manager)

        self.assertTrue(migrated)
        with self.manager._session() as session:
            row = session.get(KnowledgeBaseRow, "legacy-kb")
            tenants = tenant_manager.list_tenants_for_user("user-1")
            self.assertEqual(row.tenant_id, tenants[0].id)

    def test_migrate_tenant_id_noop_when_no_tenant_exists_yet(self):
        with self.manager._session() as session:
            session.add(KnowledgeBaseRow(
                id="legacy-kb", tenant_id=None, name="旧知识库", description="",
                created_at="2024-01-01T00:00:00+00:00",
            ))
            session.commit()

        tenant_manager = TenantManager(database_url=self.db_url)
        migrated = self.manager.migrate_tenant_id_if_needed(tenant_manager)

        self.assertFalse(migrated)
        with self.manager._session() as session:
            row = session.get(KnowledgeBaseRow, "legacy-kb")
            self.assertIsNone(row.tenant_id)

    def test_migrate_tenant_id_noop_when_already_backfilled(self):
        tenant_manager = TenantManager(database_url=self.db_url)
        tenant_manager.create_tenant("工作区", "user-1")
        self.manager.create("知识库", "", tenant_manager.list_tenants_for_user("user-1")[0].id)

        migrated = self.manager.migrate_tenant_id_if_needed(tenant_manager)
        self.assertFalse(migrated)

    def test_migrate_tenant_id_is_idempotent_across_restarts(self):
        with self.manager._session() as session:
            session.add(KnowledgeBaseRow(
                id="legacy-kb", tenant_id=None, name="旧知识库", description="",
                created_at="2024-01-01T00:00:00+00:00",
            ))
            session.commit()
        tenant_manager = TenantManager(database_url=self.db_url)
        tenant_manager.create_tenant("默认工作区", "user-1")

        first_run = self.manager.migrate_tenant_id_if_needed(tenant_manager)
        second_run = self.manager.migrate_tenant_id_if_needed(tenant_manager)

        self.assertTrue(first_run)
        self.assertFalse(second_run)
```

- [ ] **Step 7: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_kb_manager -v 2>&1 | tail -15
```

Expected: `AttributeError: 'KBManager' object has no attribute 'migrate_tenant_id_if_needed'`（`test_get_persist_dir_is_nested_by_tenant` 这一条也会失败，因为 `get_persist_dir` 还是老的单层签名）。

- [ ] **Step 8: 修改 `ragify/core/kb_manager.py` 顶部 import，并改写 `get_persist_dir` + 新增 `migrate_tenant_id_if_needed`**

顶部 import 里加一行（`tenant_manager.py` 不 import `kb_manager.py`，所以这个方向的直接 import 不会产生循环 import，不需要 `TYPE_CHECKING` 这类延迟引用的写法）：
```python
from .tenant_manager import TenantManager
```

（`from ..db.models import KnowledgeBaseRow` 这一行不用改——`migrate_tenant_id_if_needed` 只通过 `tenant_manager.get_earliest_tenant()` 拿工作区信息，不需要直接查 `TenantRow`。）

`get_persist_dir` 改成：
```python
    def get_persist_dir(self, tenant_id: str, kb_id: str) -> str:
        return str(self.vectorstore_dir / tenant_id / kb_id)
```

在 `KBManager` 类末尾（`get_persist_dir` 之后）追加：
```python

    def migrate_tenant_id_if_needed(self, tenant_manager: TenantManager) -> bool:
        earliest_tenant = tenant_manager.get_earliest_tenant()
        if earliest_tenant is None:
            return False

        with self._session() as session:
            orphans = session.query(KnowledgeBaseRow).filter(KnowledgeBaseRow.tenant_id.is_(None)).all()
            if not orphans:
                return False
            for row in orphans:
                row.tenant_id = earliest_tenant.id
            session.commit()
        logger.info("已将 %d 个知识库回填到默认工作区 %s", len(orphans), earliest_tenant.id)
        return True
```

- [ ] **Step 9: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_kb_manager -v
```

Expected: 21 个测试全部 `ok`。

- [ ] **Step 10: 跑现有完整测试套件确认无回归**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -10
```

Expected: 全部通过（现有 193 个测试里，`test_tenant_manager.py`/`test_kb_manager.py` 的测试数已经变化，具体总数以实际输出为准，不应有 FAILED/ERROR）。

- [ ] **Step 11: Commit**

```bash
git add ragify/core/kb_manager.py tests/test_kb_manager.py
git commit -m "feat: KBManager 新增 get_persist_dir 分层路径 + migrate_tenant_id_if_needed（幂等回填默认工作区）"
```

---

### Task 4: `KBManager.migrate_vectorstore_layout_if_needed`（物理文件迁移）

**Files:**
- Modify: `ragify/core/kb_manager.py`
- Modify: `tests/test_kb_manager.py`

- [ ] **Step 1: 在 `tests/test_kb_manager.py` 末尾追加失败的测试**

```python

    def test_migrate_vectorstore_layout_moves_flat_dirs_to_nested(self):
        # 模拟 Phase 4 之前的扁平布局：vectorstore/{kb_id}/ 直接放在
        # vectorstore_dir 下面，不经过 tenant_id 那一层。
        kb = self.manager.create("知识库", "", TENANT_A)
        nested_dir = Path(self.manager.get_persist_dir(TENANT_A, kb.id))
        # create() 在 Task 2/3 之后已经直接建分层目录了，这里手动模拟"还是
        # 旧布局"的场景：把内容搬回扁平路径，删掉分层目录。
        flat_dir = self.vectorstore_dir / kb.id
        shutil.move(str(nested_dir), str(flat_dir))
        (flat_dir / "index.faiss").write_text("fake-index", encoding="utf-8")

        migrated = self.manager.migrate_vectorstore_layout_if_needed()

        self.assertTrue(migrated)
        self.assertFalse(flat_dir.exists())
        self.assertTrue(nested_dir.exists())
        self.assertEqual((nested_dir / "index.faiss").read_text(encoding="utf-8"), "fake-index")

    def test_migrate_vectorstore_layout_noop_when_already_nested(self):
        self.manager.create("知识库", "", TENANT_A)
        self.assertFalse(self.manager.migrate_vectorstore_layout_if_needed())

    def test_migrate_vectorstore_layout_skips_rows_without_tenant_id(self):
        # 还没跑 migrate_tenant_id_if_needed 的行没法知道要搬到哪个
        # tenant_id 目录下面，这次迁移应该跳过它们，不报错、不误搬。
        with self.manager._session() as session:
            session.add(KnowledgeBaseRow(
                id="legacy-kb", tenant_id=None, name="旧知识库", description="",
                created_at="2024-01-01T00:00:00+00:00",
            ))
            session.commit()
        flat_dir = self.vectorstore_dir / "legacy-kb"
        flat_dir.mkdir(parents=True, exist_ok=True)
        (flat_dir / "index.faiss").write_text("fake-index", encoding="utf-8")

        migrated = self.manager.migrate_vectorstore_layout_if_needed()

        self.assertFalse(migrated)
        self.assertTrue(flat_dir.exists())
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_kb_manager -v 2>&1 | tail -10
```

Expected: `AttributeError: 'KBManager' object has no attribute 'migrate_vectorstore_layout_if_needed'`。

- [ ] **Step 3: 在 `ragify/core/kb_manager.py` 类末尾追加**

```python

    def migrate_vectorstore_layout_if_needed(self) -> bool:
        with self._session() as session:
            rows = session.query(KnowledgeBaseRow).filter(KnowledgeBaseRow.tenant_id.isnot(None)).all()
            kb_infos = [(row.id, row.tenant_id) for row in rows]

        migrated_any = False
        for kb_id, tenant_id in kb_infos:
            flat_dir = self.vectorstore_dir / kb_id
            nested_dir = self.vectorstore_dir / tenant_id / kb_id
            if not flat_dir.exists() or nested_dir.exists():
                continue
            nested_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(flat_dir), str(nested_dir))
            logger.info("已将知识库 %s 的向量库文件从扁平路径迁移到 %s", kb_id, nested_dir)
            migrated_any = True

        return migrated_any
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_kb_manager -v
```

Expected: 24 个测试全部 `ok`。

- [ ] **Step 5: 跑现有完整测试套件确认无回归**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -10
```

Expected: 全部通过，无 FAILED/ERROR。

- [ ] **Step 6: Commit**

```bash
git add ragify/core/kb_manager.py tests/test_kb_manager.py
git commit -m "feat: KBManager 新增 migrate_vectorstore_layout_if_needed（扁平路径迁移到按 tenant_id 分层）"
```

---

### Task 5: `ragify/api/main.py` 挂载新迁移调用

**Files:**
- Modify: `ragify/api/main.py`

- [ ] **Step 1: 修改启动事件**

当前内容：
```python
@app.on_event("startup")
def _migrate_legacy_json_on_startup() -> None:
    KBManager().migrate_json_if_needed()
    TenantManager().migrate_default_tenant_if_needed()
```

改成：
```python
@app.on_event("startup")
def _migrate_legacy_json_on_startup() -> None:
    KBManager().migrate_json_if_needed()
    tenant_manager = TenantManager()
    tenant_manager.migrate_default_tenant_if_needed()
    KBManager().migrate_tenant_id_if_needed(tenant_manager)
    KBManager().migrate_vectorstore_layout_if_needed()
```

文件其余部分（import、`app.include_router(...)` 那几行）不变。

- [ ] **Step 2: 验证能正常导入 + 启动**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify && .venv/bin/python -c "
from ragify.api.main import app
print('ok')
"
```

Expected: 打印 `ok`。

- [ ] **Step 3: 改写 `tests/test_api_startup_migration.py`**

当前内容只断言了 `migrate_json_if_needed`/`migrate_default_tenant_if_needed` 各被调用一次。`KBManager` 被 mock 成类之后，`mock_kb_manager_cls.return_value` 是固定的同一个 Mock 对象——不管代码里构造几次 `KBManager()`，拿到的都是这同一个 mock 实例，所以可以直接对它补断言，不需要改 mock 的构造方式：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 FastAPI 启动事件会依次调用 KBManager.migrate_json_if_needed()、
TenantManager.migrate_default_tenant_if_needed()、
KBManager.migrate_tenant_id_if_needed()、
KBManager.migrate_vectorstore_layout_if_needed()。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.main import app


class TestStartupMigration(unittest.TestCase):
    @patch("ragify.api.main.TenantManager")
    @patch("ragify.api.main.KBManager")
    def test_startup_runs_migration(self, mock_kb_manager_cls, mock_tenant_manager_cls):
        mock_kb_manager = mock_kb_manager_cls.return_value
        mock_tenant_manager = mock_tenant_manager_cls.return_value
        with TestClient(app):
            pass  # 进入/退出 with 块会触发 startup/shutdown 事件
        mock_kb_manager.migrate_json_if_needed.assert_called_once()
        mock_tenant_manager.migrate_default_tenant_if_needed.assert_called_once()
        mock_kb_manager.migrate_tenant_id_if_needed.assert_called_once_with(mock_tenant_manager)
        mock_kb_manager.migrate_vectorstore_layout_if_needed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_api_startup_migration -v
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -10
```

Expected: 全部通过，无 FAILED/ERROR。

- [ ] **Step 5: Commit**

```bash
git add ragify/api/main.py tests/test_api_startup_migration.py
git commit -m "feat: main.py 启动事件补上 KBManager 的 tenant_id 回填 + 向量库布局迁移"
```

---

### Task 6: `dependencies.py` 的 `resolve_kb_path` 加 tenant_id

**Files:**
- Modify: `ragify/api/dependencies.py`

- [ ] **Step 1: 修改 `resolve_kb_path`**

当前内容：
```python
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

改成：
```python
def resolve_kb_path(manager: KBManager, kb_id: str | None, tenant_id: str) -> str:
    """解析 kb_id 对应的 persist_directory，并把它写进全局 vectorstore 配置。

    调用方必须已经持有 KB_LOCK。如果 kb_id 为 None，回退到这个工作区里第一个
    可用知识库——Phase 4 之后"第一个可用知识库"的范围收窄到当前 tenant_id
    下面，不再是全局第一个。
    """
    if kb_id:
        kb = manager.get(kb_id, tenant_id)
        if kb is None:
            raise ValueError(f"知识库 '{kb_id}' 不存在")
    else:
        all_kbs = manager.list_all(tenant_id)
        if not all_kbs:
            raise ValueError("没有可用知识库，请先创建知识库")
        kb_id = all_kbs[0].id

    persist_dir = manager.get_persist_dir(tenant_id, kb_id)
    get_config().update("vectorstore.persist_directory", persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir
```

文件其余部分（`KB_LOCK`、`get_kb_manager`、`get_current_user`、`get_tenant_manager`、`require_membership`、`require_role` 等）不变——这个文件里已经有 `require_membership`/`require_role`，Task 7/8/9 的路由改造会直接复用，不需要在这里新增任何依赖。

- [ ] **Step 2: 跑测试，确认按预期报错（此时 `query.py`/`documents.py` 还在用旧的两参数调用方式，还没改，属于预期中的中间态失败）**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify && .venv/bin/python -m unittest discover -s tests 2>&1 | tail -20
```

Expected: `tests/test_api_query.py`、`tests/test_api_documents.py` 里所有间接调用了 `resolve_kb_path` 的测试会报 `TypeError: resolve_kb_path() missing 1 required positional argument: 'tenant_id'`——这是预期的、跨越多个任务的中间态失败，Task 7/8 改完对应路由后才会恢复绿。**不要在这一步尝试修复这些测试**，它们的修复属于 Task 7/8 的范围。

- [ ] **Step 3: Commit**

```bash
git add ragify/api/dependencies.py
git commit -m "feat: resolve_kb_path 加 tenant_id 参数，KB 查找范围收窄到当前工作区"
```

（这个 commit 之后到 Task 8 完成之前，`test_api_query.py`/`test_api_documents.py` 会持续处于失败状态——这是刻意的、跨任务的中间态，跟 Phase 1/2/3 里"先加参数、再逐个路由文件跟上"的节奏一致，不是这个任务本身没做完。）

---

### Task 7: `kb.py` 路由改形 + 权限门禁

**Files:**
- Modify: `ragify/api/routers/kb.py`
- Modify: `tests/test_api_kb.py`

- [ ] **Step 1: 改写 `tests/test_api_kb.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/tenants/{tenant_id}/kb 路由测试：建/查/删知识库，登录门禁 + 权限矩阵。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_kb_manager, get_tenant_manager, get_user_manager
from ragify.api.main import app
from ragify.core.kb_manager import KBManager
from ragify.core.tenant_manager import TenantManager
from ragify.core.user_manager import UserManager
from ragify.db.models import Base, TenantAccountJoinRow
from ragify.db.session import get_engine


class TestKBRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = KBManager(
            database_url=self.db_url,
            vectorstore_dir=Path(self.tmp_dir) / "vectorstore",
        )
        self.user_manager = UserManager(database_url=self.db_url)
        self.tenant_manager = TenantManager(database_url=self.db_url)
        app.dependency_overrides[get_kb_manager] = lambda: self.manager
        app.dependency_overrides[get_user_manager] = lambda: self.user_manager
        app.dependency_overrides[get_tenant_manager] = lambda: self.tenant_manager
        self.client = TestClient(app)

        self.owner_token = self._register("owner@example.com", "Owner")
        self.tenant_id = self._create_tenant()

    def tearDown(self):
        app.dependency_overrides.clear()
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _register(self, email: str, name: str) -> str:
        res = self.client.post("/api/auth/register", json={
            "email": email, "password": "password123", "name": name,
        })
        return res.json()["access_token"]

    def _auth(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def _create_tenant(self) -> str:
        res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        return res.json()["id"]

    def _add_member(self, email: str, name: str, role: str) -> str:
        token = self._register(email, name)
        user_id = self.user_manager.get_by_email(email).id
        import uuid
        from datetime import datetime, timezone
        with self.tenant_manager._session() as session:
            session.add(TenantAccountJoinRow(
                id=uuid.uuid4().hex[:12], tenant_id=self.tenant_id, user_id=user_id,
                role=role, created_at=datetime.now(timezone.utc).isoformat(),
            ))
            session.commit()
        return token

    def test_list_kbs_empty(self):
        res = self.client.get(f"/api/tenants/{self.tenant_id}/kb", headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"knowledge_bases": []})

    def test_create_and_list_kb(self):
        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb",
            json={"name": "测试库", "description": "desc"},
            headers=self._auth(self.owner_token),
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["name"], "测试库")

        res = self.client.get(f"/api/tenants/{self.tenant_id}/kb", headers=self._auth(self.owner_token))
        kbs = res.json()["knowledge_bases"]
        self.assertEqual(len(kbs), 1)
        self.assertEqual(kbs[0]["name"], "测试库")
        self.assertEqual(kbs[0]["doc_count"], 0)

    def test_create_kb_empty_name_rejected(self):
        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb", json={"name": "   "}, headers=self._auth(self.owner_token)
        )
        self.assertEqual(res.status_code, 400)

    def test_create_kb_duplicate_name_rejected(self):
        self.client.post(f"/api/tenants/{self.tenant_id}/kb", json={"name": "重复"}, headers=self._auth(self.owner_token))
        res = self.client.post(f"/api/tenants/{self.tenant_id}/kb", json={"name": "重复"}, headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 400)

    def test_delete_kb(self):
        create_res = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb", json={"name": "待删除"}, headers=self._auth(self.owner_token)
        )
        kb_id = create_res.json()["id"]
        res = self.client.delete(f"/api/tenants/{self.tenant_id}/kb/{kb_id}", headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"success": True})

    def test_delete_missing_kb_returns_404(self):
        res = self.client.delete(
            f"/api/tenants/{self.tenant_id}/kb/does-not-exist", headers=self._auth(self.owner_token)
        )
        self.assertEqual(res.status_code, 404)

    def test_no_token_rejected(self):
        res = self.client.get(f"/api/tenants/{self.tenant_id}/kb")
        self.assertEqual(res.status_code, 401)

    def test_non_member_cannot_list(self):
        other_token = self._register("other@example.com", "Other")
        res = self.client.get(f"/api/tenants/{self.tenant_id}/kb", headers=self._auth(other_token))
        self.assertEqual(res.status_code, 403)

    def test_normal_member_can_list_but_cannot_create(self):
        normal_token = self._add_member("normal@example.com", "Normal", "NORMAL")
        list_res = self.client.get(f"/api/tenants/{self.tenant_id}/kb", headers=self._auth(normal_token))
        self.assertEqual(list_res.status_code, 200)

        create_res = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb", json={"name": "不该建成"}, headers=self._auth(normal_token)
        )
        self.assertEqual(create_res.status_code, 403)

    def test_dataset_operator_cannot_create_or_delete_kb(self):
        operator_token = self._add_member("operator@example.com", "Operator", "DATASET_OPERATOR")
        create_res = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb", json={"name": "不该建成"}, headers=self._auth(operator_token)
        )
        self.assertEqual(create_res.status_code, 403)

        existing = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb", json={"name": "已有知识库"}, headers=self._auth(self.owner_token)
        ).json()
        delete_res = self.client.delete(
            f"/api/tenants/{self.tenant_id}/kb/{existing['id']}", headers=self._auth(operator_token)
        )
        self.assertEqual(delete_res.status_code, 403)

    def test_editor_can_create_and_delete_kb(self):
        editor_token = self._add_member("editor@example.com", "Editor", "EDITOR")
        create_res = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb", json={"name": "编辑建的库"}, headers=self._auth(editor_token)
        )
        self.assertEqual(create_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_api_kb -v 2>&1 | tail -15
```

Expected: 大量 404（旧路由 `/api/kb` 还在，新路由 `/api/tenants/{tenant_id}/kb` 还不存在）。

- [ ] **Step 3: 改写 `ragify/api/routers/kb.py`**

```python
import logging

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import KB_LOCK, get_kb_manager, require_membership, require_role
from ..schemas import CreateKBRequest
from ...config import get_config
from ...core.kb_manager import KBManager
from ...core.tenant_manager import Membership
from ...core.vectorstores import VectorStoreManager

logger = logging.getLogger("ragify.api.routers.kb")

router = APIRouter()


@router.get("/api/tenants/{tenant_id}/kb")
def list_kbs(
    tenant_id: str,
    membership: Membership = Depends(require_membership),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    kbs = manager.list_all(tenant_id)
    kbs_out = []
    with KB_LOCK:
        for kb in kbs:
            doc_count = 0
            try:
                persist_dir = manager.get_persist_dir(tenant_id, kb.id)
                get_config().update("vectorstore.persist_directory", persist_dir)
                vm = VectorStoreManager()
                doc_count = vm.get_document_count()
            except Exception as e:
                logger.warning("获取知识库 '%s' (%s) 的文档数失败: %s", kb.name, kb.id, e)
            kbs_out.append({
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "created_at": kb.created_at,
                "doc_count": doc_count,
            })
    return {"knowledge_bases": kbs_out}


@router.post("/api/tenants/{tenant_id}/kb")
def create_kb(
    tenant_id: str,
    body: CreateKBRequest,
    membership: Membership = Depends(require_role("OWNER", "ADMIN", "EDITOR")),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="知识库名称不能为空")
    try:
        kb = manager.create(name, body.description, tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "created_at": kb.created_at,
    }


@router.delete("/api/tenants/{tenant_id}/kb/{kb_id}")
def delete_kb(
    tenant_id: str,
    kb_id: str,
    membership: Membership = Depends(require_role("OWNER", "ADMIN", "EDITOR")),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    ok = manager.delete(kb_id, tenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"知识库 '{kb_id}' 不存在")
    return {"success": True}
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_api_kb -v
```

Expected: 12 个测试全部 `ok`。

- [ ] **Step 5: Commit**

```bash
git add ragify/api/routers/kb.py tests/test_api_kb.py
git commit -m "feat: /api/kb 路由挪到 /api/tenants/{tenant_id}/kb，加登录门禁 + 建/删知识库权限门禁"
```

---

### Task 8: `query.py` 路由改形 + 权限门禁

**Files:**
- Modify: `ragify/api/routers/query.py`
- Modify: `tests/test_api_query.py`

- [ ] **Step 1: 改写 `tests/test_api_query.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/tenants/{tenant_id}/query 和 .../query/agentic 路由测试。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_kb_manager, get_tenant_manager, get_user_manager
from ragify.api.main import app
from ragify.core.kb_manager import KBManager
from ragify.core.tenant_manager import TenantManager
from ragify.core.user_manager import UserManager
from ragify.db.models import Base, TenantAccountJoinRow
from ragify.db.session import get_engine


class TestQueryRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = KBManager(
            database_url=self.db_url,
            vectorstore_dir=Path(self.tmp_dir) / "vectorstore",
        )
        self.user_manager = UserManager(database_url=self.db_url)
        self.tenant_manager = TenantManager(database_url=self.db_url)
        app.dependency_overrides[get_kb_manager] = lambda: self.manager
        app.dependency_overrides[get_user_manager] = lambda: self.user_manager
        app.dependency_overrides[get_tenant_manager] = lambda: self.tenant_manager
        self.client = TestClient(app)

        self.owner_token = self._register("owner@example.com", "Owner")
        tenant_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        self.tenant_id = tenant_res.json()["id"]
        self.manager.create("默认知识库", "", self.tenant_id)

    def tearDown(self):
        app.dependency_overrides.clear()
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _register(self, email: str, name: str) -> str:
        res = self.client.post("/api/auth/register", json={
            "email": email, "password": "password123", "name": name,
        })
        return res.json()["access_token"]

    def _auth(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def _add_member(self, email: str, name: str, role: str) -> str:
        token = self._register(email, name)
        user_id = self.user_manager.get_by_email(email).id
        import uuid
        from datetime import datetime, timezone
        with self.tenant_manager._session() as session:
            session.add(TenantAccountJoinRow(
                id=uuid.uuid4().hex[:12], tenant_id=self.tenant_id, user_id=user_id,
                role=role, created_at=datetime.now(timezone.utc).isoformat(),
            ))
            session.commit()
        return token

    def test_query_without_available_kb_returns_400(self):
        kbs = self.manager.list_all(self.tenant_id)
        self.manager.delete(kbs[0].id, self.tenant_id)

        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/query", json={"query": "test"}, headers=self._auth(self.owner_token)
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("没有可用知识库", res.json()["detail"])

    def test_agentic_query_without_available_kb_returns_400(self):
        kbs = self.manager.list_all(self.tenant_id)
        self.manager.delete(kbs[0].id, self.tenant_id)

        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/query/agentic", json={"query": "test"}, headers=self._auth(self.owner_token)
        )
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

        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/query",
            json={"query": "什么是RAG", "k": 3},
            headers=self._auth(self.owner_token),
        )

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

        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/query/agentic",
            json={"query": "问题"},
            headers=self._auth(self.owner_token),
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["response"], "答案")
        mock_agent.run.assert_called_once_with("问题", chat_history=None)

    def test_no_token_rejected(self):
        res = self.client.post(f"/api/tenants/{self.tenant_id}/query", json={"query": "test"})
        self.assertEqual(res.status_code, 401)

    def test_non_member_cannot_query(self):
        other_token = self._register("other@example.com", "Other")
        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/query", json={"query": "test"}, headers=self._auth(other_token)
        )
        self.assertEqual(res.status_code, 403)

    @patch("ragify.api.routers.query.QueryPipeline")
    def test_normal_member_can_query(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "response": "答案", "response_generated": True,
            "retrieved_documents": [], "query_summary": {},
        }
        mock_pipeline_cls.return_value = mock_pipeline
        normal_token = self._add_member("normal@example.com", "Normal", "NORMAL")

        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/query", json={"query": "test"}, headers=self._auth(normal_token)
        )
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_api_query -v 2>&1 | tail -15
```

Expected: 404（新路由还不存在）。

- [ ] **Step 3: 改写 `ragify/api/routers/query.py`**

```python
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import KB_LOCK, get_kb_manager, resolve_kb_path, require_membership
from ..schemas import AgenticQueryRequest, QueryRequest
from ...agentic import AgenticRAG
from ...core.kb_manager import KBManager
from ...core.tenant_manager import Membership
from ...mcp import QueryPipeline

router = APIRouter()


@router.post("/api/tenants/{tenant_id}/query")
def query(
    tenant_id: str,
    body: QueryRequest,
    membership: Membership = Depends(require_membership),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id, tenant_id)
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


@router.post("/api/tenants/{tenant_id}/query/agentic")
def agentic_query(
    tenant_id: str,
    body: AgenticQueryRequest,
    membership: Membership = Depends(require_membership),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    # 整个 AgenticRAG 构造 + .run() 都在锁内——它的 retrieve_docs 工具在
    # run() 执行期间（不是构造时）才现读一次全局 vectorstore 配置，所以
    # 不能像 query() 那样提前把锁放掉。见 dependencies.py 顶部的并发说明。
    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        agent = AgenticRAG(kb_id=body.kb_id, max_iterations=body.max_iterations)
        result = agent.run(body.query, chat_history=body.chat_history)
    return result
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_api_query -v
```

Expected: 8 个测试全部 `ok`。

- [ ] **Step 5: Commit**

```bash
git add ragify/api/routers/query.py tests/test_api_query.py
git commit -m "feat: /api/query 路由挪到 /api/tenants/{tenant_id}/query，加登录门禁（任意成员可查询）"
```

---

### Task 9: `documents.py` 路由改形 + 权限门禁

**Files:**
- Modify: `ragify/api/routers/documents.py`
- Modify: `tests/test_api_documents.py`

- [ ] **Step 1: 改写 `tests/test_api_documents.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/tenants/{tenant_id}/{index,stats,documents,chunks} 路由测试。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_kb_manager, get_tenant_manager, get_user_manager
from ragify.api.main import app
from ragify.core.kb_manager import KBManager
from ragify.core.tenant_manager import TenantManager
from ragify.core.user_manager import UserManager
from ragify.db.models import Base, TenantAccountJoinRow
from ragify.db.session import get_engine


class TestDocumentsRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = KBManager(
            database_url=self.db_url,
            vectorstore_dir=Path(self.tmp_dir) / "vectorstore",
        )
        self.user_manager = UserManager(database_url=self.db_url)
        self.tenant_manager = TenantManager(database_url=self.db_url)
        app.dependency_overrides[get_kb_manager] = lambda: self.manager
        app.dependency_overrides[get_user_manager] = lambda: self.user_manager
        app.dependency_overrides[get_tenant_manager] = lambda: self.tenant_manager
        self.client = TestClient(app)

        self.owner_token = self._register("owner@example.com", "Owner")
        tenant_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        self.tenant_id = tenant_res.json()["id"]
        self.kb = self.manager.create("默认知识库", "", self.tenant_id)

    def tearDown(self):
        app.dependency_overrides.clear()
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _register(self, email: str, name: str) -> str:
        res = self.client.post("/api/auth/register", json={
            "email": email, "password": "password123", "name": name,
        })
        return res.json()["access_token"]

    def _auth(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def _add_member(self, email: str, name: str, role: str) -> str:
        token = self._register(email, name)
        user_id = self.user_manager.get_by_email(email).id
        import uuid
        from datetime import datetime, timezone
        with self.tenant_manager._session() as session:
            session.add(TenantAccountJoinRow(
                id=uuid.uuid4().hex[:12], tenant_id=self.tenant_id, user_id=user_id,
                role=role, created_at=datetime.now(timezone.utc).isoformat(),
            ))
            session.commit()
        return token

    def _p(self, path: str) -> str:
        return f"/api/tenants/{self.tenant_id}{path}"

    @patch("ragify.api.routers.documents.IndexingPipeline")
    def test_index_with_directory_path(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"indexing_summary": {"total_documents_indexed": 2}}
        mock_pipeline_cls.return_value = mock_pipeline

        res = self.client.post(self._p("/index"), json={
            "directory_path": "/some/dir", "kb_id": self.kb.id,
        }, headers=self._auth(self.owner_token))

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["indexing_summary"]["total_documents_indexed"], 2)
        called_payload = mock_pipeline.run.call_args[0][0]
        self.assertEqual(called_payload["directory_path"], "/some/dir")
        self.assertTrue(called_payload["clear_vectorstore"])

    @patch("ragify.api.routers.documents.IndexingPipeline")
    def test_index_with_file_paths_does_not_default_clear_vectorstore(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"indexing_summary": {"total_documents_indexed": 1}}
        mock_pipeline_cls.return_value = mock_pipeline

        res = self.client.post(self._p("/index"), json={
            "file_paths": ["/some/file.txt"], "kb_id": self.kb.id,
        }, headers=self._auth(self.owner_token))

        self.assertEqual(res.status_code, 200)
        called_payload = mock_pipeline.run.call_args[0][0]
        self.assertEqual(called_payload["file_paths"], ["/some/file.txt"])
        self.assertNotIn("clear_vectorstore", called_payload)

    @patch("ragify.api.routers.documents.IndexingPipeline")
    def test_index_falls_back_to_kb_data_dir_when_no_path_or_files_given(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"indexing_summary": {"total_documents_indexed": 0}}
        mock_pipeline_cls.return_value = mock_pipeline

        kb_data_dir = Path(self.tmp_dir) / "project_root_stub" / "data" / self.kb.id

        with patch("ragify.api.routers.documents.PROJECT_ROOT", str(Path(self.tmp_dir) / "project_root_stub")):
            kb_data_dir.mkdir(parents=True, exist_ok=True)
            res = self.client.post(
                self._p("/index"), json={"kb_id": self.kb.id}, headers=self._auth(self.owner_token)
            )

        self.assertEqual(res.status_code, 200)
        called_payload = mock_pipeline.run.call_args[0][0]
        self.assertEqual(called_payload["directory_path"], str(kb_data_dir))
        self.assertTrue(called_payload["clear_vectorstore"])

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_clear_index(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.request(
            "DELETE", self._p("/index"), json={"kb_id": self.kb.id}, headers=self._auth(self.owner_token)
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"success": True})
        mock_vsm.clear.assert_called_once()

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_get_stats(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.get_document_count.return_value = 5
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.get(self._p(f"/stats?kb_id={self.kb.id}"), headers=self._auth(self.owner_token))

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["doc_count"], 5)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_list_documents(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.get_sources.return_value = [{"name": "a.txt", "source": "a.txt"}]
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.get(self._p(f"/documents?kb_id={self.kb.id}"), headers=self._auth(self.owner_token))

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["total"], 1)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_delete_document(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.delete_by_source.return_value = 3
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.request("DELETE", self._p("/documents"), json={
            "kb_id": self.kb.id, "source": "nonexistent.txt",
        }, headers=self._auth(self.owner_token))

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["chunks_removed"], 3)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_list_chunks_requires_source(self, mock_vsm_cls):
        res = self.client.get(self._p(f"/chunks?kb_id={self.kb.id}"), headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 422)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_update_chunk(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.update_chunk_content.return_value = True
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.put(self._p("/chunks"), json={
            "kb_id": self.kb.id, "chunk_id": "c1", "content": "新内容",
        }, headers=self._auth(self.owner_token))

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"success": True})

    def test_get_stats_without_kb_returns_400_when_no_kbs_exist(self):
        self.manager.delete(self.kb.id, self.tenant_id)
        res = self.client.get(self._p("/stats"), headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 400)
        self.assertIn("没有可用知识库", res.json()["detail"])

    def test_no_token_rejected(self):
        res = self.client.get(self._p(f"/documents?kb_id={self.kb.id}"))
        self.assertEqual(res.status_code, 401)

    def test_normal_member_can_read_but_cannot_upload_or_delete(self):
        normal_token = self._add_member("normal@example.com", "Normal", "NORMAL")

        read_res = self.client.get(self._p(f"/documents?kb_id={self.kb.id}"), headers=self._auth(normal_token))
        self.assertEqual(read_res.status_code, 200)

        upload_res = self.client.post(
            self._p("/index"), json={"kb_id": self.kb.id, "file_paths": ["/x.txt"]},
            headers=self._auth(normal_token),
        )
        self.assertEqual(upload_res.status_code, 403)

    @patch("ragify.api.routers.documents.IndexingPipeline")
    def test_dataset_operator_can_upload_documents(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"indexing_summary": {"total_documents_indexed": 1}}
        mock_pipeline_cls.return_value = mock_pipeline
        operator_token = self._add_member("operator@example.com", "Operator", "DATASET_OPERATOR")

        res = self.client.post(
            self._p("/index"), json={"kb_id": self.kb.id, "file_paths": ["/x.txt"]},
            headers=self._auth(operator_token),
        )
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_api_documents -v 2>&1 | tail -15
```

Expected: 404（新路由还不存在）。

- [ ] **Step 3: 改写 `ragify/api/routers/documents.py`**

```python
import os

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import KB_LOCK, PROJECT_ROOT, get_kb_manager, resolve_kb_path, require_membership, require_role
from ..schemas import ClearIndexRequest, DeleteDocRequest, IndexRequest, UpdateChunkRequest
from ...core.kb_manager import KBManager
from ...core.tenant_manager import Membership
from ...core.vectorstores import VectorStoreManager
from ...config import get_config
from ...mcp import IndexingPipeline

router = APIRouter()

_DATASET_ROLES = ("OWNER", "ADMIN", "EDITOR", "DATASET_OPERATOR")


@router.post("/api/tenants/{tenant_id}/index")
def index_documents(
    tenant_id: str,
    body: IndexRequest,
    membership: Membership = Depends(require_role(*_DATASET_ROLES)),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id, tenant_id)
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


@router.delete("/api/tenants/{tenant_id}/index")
def clear_index(
    tenant_id: str,
    body: ClearIndexRequest,
    membership: Membership = Depends(require_role(*_DATASET_ROLES)),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    vm.clear()
    return {"success": True}


@router.get("/api/tenants/{tenant_id}/stats")
def get_stats(
    tenant_id: str,
    kb_id: str | None = None,
    membership: Membership = Depends(require_membership),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, kb_id, tenant_id)
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


@router.get("/api/tenants/{tenant_id}/documents")
def list_documents(
    tenant_id: str,
    kb_id: str | None = None,
    membership: Membership = Depends(require_membership),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    sources = vm.get_sources()
    return {"documents": sources, "total": len(sources)}


@router.delete("/api/tenants/{tenant_id}/documents")
def delete_document(
    tenant_id: str,
    body: DeleteDocRequest,
    membership: Membership = Depends(require_role(*_DATASET_ROLES)),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    source = body.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="缺少 source 参数")

    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id, tenant_id)
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


@router.get("/api/tenants/{tenant_id}/chunks")
def list_chunks(
    tenant_id: str,
    source: str,
    kb_id: str | None = None,
    membership: Membership = Depends(require_membership),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    if not source.strip():
        raise HTTPException(status_code=400, detail="缺少 source 参数")

    with KB_LOCK:
        try:
            resolve_kb_path(manager, kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    chunks = vm.get_chunks_by_source(source)
    return {"chunks": chunks, "total": len(chunks)}


@router.put("/api/tenants/{tenant_id}/chunks")
def update_chunk(
    tenant_id: str,
    body: UpdateChunkRequest,
    membership: Membership = Depends(require_role(*_DATASET_ROLES)),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    if not body.chunk_id:
        raise HTTPException(status_code=400, detail="缺少 chunk_id 参数")

    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    ok = vm.update_chunk_content(body.chunk_id, body.content)
    return {"success": ok}
```

（`ClearIndexRequest`/`DeleteDocRequest` 原来 `kb_id` 字段是 body 里传的，改形后依然从 body 传——只是路由本身多了一段 `/api/tenants/{tenant_id}` 前缀，`tenant_id` 是路径参数，跟 body 里的 `kb_id` 是两个独立的东西，不要混淆着改。`DeleteDocRequest`/`UpdateChunkRequest` 的 `kb_id` 字段目前是必填的 `str`（不是 `str | None`），跟 `resolve_kb_path` 的 `kb_id: str | None` 参数类型不完全一致，但这是 Phase 1 就有的既有行为，不在这次改动范围内，直接传参即可，类型注解层面 FastAPI/Pydantic 不会因为传一个非 None 的 str 给 `str | None` 形参报错。）

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_api_documents -v
```

Expected: 14 个测试全部 `ok`。

- [ ] **Step 5: 跑全量测试套件确认无回归**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -10
```

Expected: 全部通过，无 FAILED/ERROR。

- [ ] **Step 6: Commit**

```bash
git add ragify/api/routers/documents.py tests/test_api_documents.py
git commit -m "feat: /api/index /api/stats /api/documents /api/chunks 挪到 /api/tenants/{tenant_id}/...，加登录门禁 + 文档管理权限门禁"
```

---

### Task 10: 前端代理路由改造

**Files:**
- Create: `frontend/src/lib/current-tenant.ts`
- Modify: `frontend/src/app/api/knowledge-bases/route.ts`
- Modify: `frontend/src/app/api/knowledge-bases/[id]/route.ts`
- Modify: `frontend/src/app/api/query/route.ts`
- Modify: `frontend/src/app/api/query/agentic/route.ts`
- Modify: `frontend/src/app/api/index/route.ts`
- Modify: `frontend/src/app/api/documents/route.ts`
- Modify: `frontend/src/app/api/stats/route.ts`
- Modify: `frontend/src/app/api/chunks/route.ts`

- [ ] **Step 1: 新增 `frontend/src/lib/current-tenant.ts`**

七个代理路由都需要同一段逻辑：读 cookie 拿 token，没有 token 就 401；查后端 `/api/tenants` 拿当前用户所属工作区列表，取第一个。抽成一个共享辅助函数，不在七个文件里各写一遍：

```typescript
import { AUTH_COOKIE_NAME } from "@/lib/auth-cookie";
import { callBackend } from "@/lib/backend";
import { NextRequest } from "next/server";

interface Tenant {
  id: string;
  name: string;
  created_at: string;
}

/**
 * 从请求的 cookie 里取 token，查这个用户所属的第一个工作区，返回
 * { token, tenantId }。目前一个用户通常只属于一个工作区（默认工作区），
 * 所以这里直接取列表第一个——真正的多工作区切换是 Phase 5 的事。
 *
 * 抛出的 Error message 就是要展示给前端调用方的错误信息："未登录"或
 * "还没有工作区"，调用方 catch 到之后统一包装成 401。
 */
export async function resolveCurrentTenant(
  req: NextRequest
): Promise<{ token: string; tenantId: string }> {
  const token = req.cookies.get(AUTH_COOKIE_NAME)?.value;
  if (!token) {
    throw new Error("未登录");
  }

  const tenants = await callBackend<Tenant[]>("/api/tenants", undefined, {
    method: "GET",
    timeout: 15_000,
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!tenants || tenants.length === 0) {
    throw new Error("还没有工作区");
  }

  return { token, tenantId: tenants[0].id };
}
```

- [ ] **Step 2: 改写 `frontend/src/app/api/knowledge-bases/route.ts`**

当前内容：
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

改成：
```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveCurrentTenant } from "@/lib/current-tenant";

export async function GET(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const result = await callBackend(`/api/tenants/${tenantId}/kb`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const body = await req.json();
    if (!body.name || typeof body.name !== "string" || !body.name.trim()) {
      return NextResponse.json(
        { error: "知识库名称不能为空" },
        { status: 400 }
      );
    }
    const result = await callBackend(`/api/tenants/${tenantId}/kb`, {
      name: body.name.trim(),
      description: typeof body.description === "string" ? body.description : "",
    }, { timeout: 15_000, headers: { Authorization: `Bearer ${token}` } });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

（原来 catch 到的错误统一包成 500，现在改成 401——因为现在 catch 块首先要处理的是 `resolveCurrentTenant` 抛出的"未登录"/"还没有工作区"这类认证性错误。如果是建知识库时 `callBackend` 本身抛出的业务错误（比如后端返回 400"知识库已存在"），这里全部统一成 401 会让"重名报错"看起来像"未登录"——这是一个已知的、这次不解决的简化：这个项目至今 `route.ts` 的错误处理一直是"catch 到什么都包成一个固定状态码"，跟 Phase 1 建立的既有约定一致，这次沿用同样的简化方式，只是把默认状态码从 500 改成 401，因为认证失败在这次改动里是更常见、更重要的失败模式。)

- [ ] **Step 3: 改写 `frontend/src/app/api/knowledge-bases/[id]/route.ts`**

当前内容：
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

改成：
```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveCurrentTenant } from "@/lib/current-tenant";

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const { id } = await params;
    const result = await callBackend(`/api/tenants/${tenantId}/kb/${id}`, undefined, {
      method: "DELETE", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 4: 改写 `frontend/src/app/api/query/route.ts`**

当前内容：
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

改成：
```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveCurrentTenant } from "@/lib/current-tenant";

export async function POST(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
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

    const result = await callBackend(`/api/tenants/${tenantId}/query`, payload, {
      timeout: 60_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 5: 改写 `frontend/src/app/api/query/agentic/route.ts`**

当前内容：
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

改成：
```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveCurrentTenant } from "@/lib/current-tenant";

export async function POST(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
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

    const result = await callBackend(`/api/tenants/${tenantId}/query/agentic`, payload, {
      timeout: 120_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 6: 改写 `frontend/src/app/api/index/route.ts`**

当前内容：
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

改成：
```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveCurrentTenant } from "@/lib/current-tenant";

export async function POST(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
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

    const result = await callBackend(`/api/tenants/${tenantId}/index`, payload, {
      timeout: 120_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const body = await req.json().catch(() => ({}));
    const result = await callBackend(`/api/tenants/${tenantId}/index`, { kb_id: body.kb_id }, {
      method: "DELETE", timeout: 30_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 7: 改写 `frontend/src/app/api/documents/route.ts`**

当前内容：
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

改成：
```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveCurrentTenant } from "@/lib/current-tenant";

export async function GET(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const kbId = req.nextUrl.searchParams.get("kb_id");
    const params = new URLSearchParams();
    if (kbId) params.set("kb_id", kbId);
    const query = params.toString() ? `?${params.toString()}` : "";
    const result = await callBackend(`/api/tenants/${tenantId}/documents${query}`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const { kb_id, source } = await req.json();
    if (!kb_id || !source) {
      return NextResponse.json({ error: "缺少 kb_id 或 source 参数" }, { status: 400 });
    }
    const result = await callBackend(`/api/tenants/${tenantId}/documents`, { kb_id, source }, {
      method: "DELETE", timeout: 60_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 8: 改写 `frontend/src/app/api/stats/route.ts`**

当前内容：
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

改成：
```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveCurrentTenant } from "@/lib/current-tenant";

export async function GET(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const kbId = req.nextUrl.searchParams.get("kb_id");
    const params = new URLSearchParams();
    if (kbId) params.set("kb_id", kbId);
    const query = params.toString() ? `?${params.toString()}` : "";
    const result = await callBackend(`/api/tenants/${tenantId}/stats${query}`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 9: 改写 `frontend/src/app/api/chunks/route.ts`**

当前内容：
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

改成：
```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveCurrentTenant } from "@/lib/current-tenant";

export async function GET(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const source = req.nextUrl.searchParams.get("source");
    const kbId = req.nextUrl.searchParams.get("kb_id");
    if (!source) {
      return NextResponse.json({ error: "缺少 source 参数" }, { status: 400 });
    }
    const params = new URLSearchParams({ source });
    if (kbId) params.set("kb_id", kbId);
    const result = await callBackend(`/api/tenants/${tenantId}/chunks?${params.toString()}`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}

export async function PUT(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const { kb_id, chunk_id, content } = await req.json();
    if (!chunk_id) {
      return NextResponse.json({ error: "缺少 chunk_id 参数" }, { status: 400 });
    }
    const result = await callBackend(`/api/tenants/${tenantId}/chunks`, { kb_id, chunk_id, content }, {
      method: "PUT", timeout: 30_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 10: 类型检查和 lint**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify/frontend && npx tsc --noEmit --pretty false && npx eslint src/lib/current-tenant.ts src/app/api/knowledge-bases src/app/api/query src/app/api/index src/app/api/documents src/app/api/stats src/app/api/chunks
```

Expected: 都无输出/无错误。

- [ ] **Step 11: Commit**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify
git add frontend/src/lib/current-tenant.ts frontend/src/app/api/knowledge-bases frontend/src/app/api/query frontend/src/app/api/index frontend/src/app/api/documents frontend/src/app/api/stats frontend/src/app/api/chunks
git commit -m "feat: 前端代理路由改成读 cookie + 查当前工作区 + 转发到 /api/tenants/{tenant_id}/... "
```

---

### Task 11: 前端登录页 + middleware

**Files:**
- Create: `frontend/src/app/login/page.tsx`
- Create: `frontend/src/middleware.ts`

- [ ] **Step 1: 新增 `frontend/src/app/login/page.tsx`**

最粗粝的登录/注册页——没有找回密码、没有邮箱验证、没有样式打磨，纯粹能用，两个表单切换：

```tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      if (mode === "login") {
        const res = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "登录失败");
      } else {
        const res = await fetch("/api/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password, name }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "注册失败");

        // 新注册的用户此时还没有任何工作区（Phase 3 的默认工作区迁移只
        // 拉了当时已存在的用户）——这里立刻建一个默认工作区，保证落地
        // 仪表盘时手上已经有工作区可用，不需要额外的"创建工作区"页面。
        const tenantRes = await fetch("/api/tenant-bootstrap", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: `${name}的工作区` }),
        });
        if (!tenantRes.ok) {
          const tenantData = await tenantRes.json().catch(() => ({}));
          throw new Error(tenantData.error || "创建默认工作区失败");
        }
      }
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "操作失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto mt-24 max-w-sm px-4">
      <h1 className="mb-6 text-2xl font-bold">
        {mode === "login" ? "登录" : "注册"} RAGify
      </h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        {mode === "register" && (
          <input
            type="text"
            placeholder="姓名"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            className="w-full rounded border px-3 py-2"
          />
        )}
        <input
          type="email"
          placeholder="邮箱"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          className="w-full rounded border px-3 py-2"
        />
        <input
          type="password"
          placeholder="密码（至少 8 位）"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          minLength={8}
          className="w-full rounded border px-3 py-2"
        />
        {error && <p className="text-sm text-red-600">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="w-full rounded bg-black px-3 py-2 text-white disabled:opacity-50"
        >
          {submitting ? "处理中..." : mode === "login" ? "登录" : "注册"}
        </button>
      </form>
      <button
        type="button"
        onClick={() => {
          setMode(mode === "login" ? "register" : "login");
          setError(null);
        }}
        className="mt-4 text-sm text-muted-foreground underline"
      >
        {mode === "login" ? "还没有账号？去注册" : "已经有账号？去登录"}
      </button>
    </div>
  );
}
```

- [ ] **Step 2: 新增 `frontend/src/app/api/tenant-bootstrap/route.ts`**

登录页注册成功后调用的代理路由——从 cookie 读 token（这时候 `/api/auth/register` 已经写好 cookie 了），转发到后端 `POST /api/tenants` 建一个工作区：

```typescript
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { AUTH_COOKIE_NAME } from "@/lib/auth-cookie";

export async function POST(req: NextRequest) {
  const token = req.cookies.get(AUTH_COOKIE_NAME)?.value;
  if (!token) {
    return NextResponse.json({ error: "未登录" }, { status: 401 });
  }
  try {
    const body = await req.json();
    const result = await callBackend(
      "/api/tenants",
      { name: typeof body.name === "string" && body.name.trim() ? body.name.trim() : "我的工作区" },
      { timeout: 15_000, headers: { Authorization: `Bearer ${token}` } }
    );
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
```

- [ ] **Step 3: 新增 `frontend/src/middleware.ts`**

```typescript
import { NextRequest, NextResponse } from "next/server";
import { AUTH_COOKIE_NAME } from "@/lib/auth-cookie";

export function middleware(req: NextRequest) {
  const token = req.cookies.get(AUTH_COOKIE_NAME)?.value;
  if (!token) {
    const loginUrl = new URL("/login", req.url);
    return NextResponse.redirect(loginUrl);
  }
  return NextResponse.next();
}

export const config = {
  // 只拦截真正的页面导航，不拦截任何 /api/* 路由——那些路由自己已经在
  // 各自的代码里判断 cookie 缺失时返回 401 JSON，如果被这个 middleware
  // 重定向到 /login（一个 HTML 页面），前端 fetch 期待的是 JSON 响应，
  // 会直接在解析阶段报错，而不是拿到一个清晰的"未登录"信号。
  matcher: [
    "/((?!api|login|_next/static|_next/image|favicon.ico).*)",
  ],
};
```

- [ ] **Step 4: 类型检查和 lint**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify/frontend && npx tsc --noEmit --pretty false && npx eslint src/app/login src/app/api/tenant-bootstrap src/middleware.ts
```

Expected: 都无输出/无错误。

- [ ] **Step 5: Commit**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify
git add frontend/src/app/login frontend/src/app/api/tenant-bootstrap frontend/src/middleware.ts
git commit -m "feat: 新增最粗粝的登录/注册页 + middleware，补上加门禁后浏览器端唯一缺失的登录入口"
```

---

### Task 12: MCP 服务的租户识别

**Files:**
- Modify: `ragify/mcp_server/server.py`
- Create: `tests/test_mcp_server.py`

设计背景见 `docs/superpowers/specs/2026-09-18-multi-tenant-phase4-data-isolation-design.md` 的"MCP 服务的租户识别"一节：`ragify/mcp_server/server.py` 是 Phase 1 建的独立 stdio 入口，不走 HTTP，没有 Authorization 头。这个任务让它在启动时读一个真实 JWT（跟浏览器登录用的是同一套 token），解出 `user_id`，再解出这个用户的第一个工作区 id，作为这次 MCP 会话全程使用的 `tenant_id`。

**这个任务依赖 Task 6 已经完成**（`resolve_kb_path` 加了 `tenant_id` 参数）——`server.py` 里 `ragify_query` 工具目前完全没调用 `resolve_kb_path`（这是 Phase 1 就存在的既有缺口，不是 Phase 4 引入的），这次顺便补上，让它跟 `query.py` 路由用同一套解析逻辑。

- [ ] **Step 1: 写失败的测试 `tests/test_mcp_server.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP server 的租户识别测试。
验证 RAGIFY_MCP_TOKEN 缺失/无效/用户不存在/用户无工作区几种启动失败场景，
以及正常场景下解出的 tenant_id 确实被传给 KBManager 的调用。
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.security import create_access_token
from ragify.core.tenant_manager import TenantManager
from ragify.core.user_manager import UserManager
from ragify.db.models import Base
from ragify.db.session import get_engine
from ragify.mcp_server.server import _call_tool, _list_resources, _resolve_mcp_tenant_id

TEST_SECRET = "test-secret-only-for-unit-tests"


class TestResolveMcpTenantId(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.user_manager = UserManager(database_url=self.db_url)
        self.tenant_manager = TenantManager(database_url=self.db_url)

    def tearDown(self):
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_token_raises(self):
        with self.assertRaises(RuntimeError):
            _resolve_mcp_tenant_id(self.user_manager, self.tenant_manager, secret=TEST_SECRET)

    def test_invalid_token_raises(self):
        with patch.dict(os.environ, {"RAGIFY_MCP_TOKEN": "not-a-real-token"}, clear=True):
            with self.assertRaises(RuntimeError):
                _resolve_mcp_tenant_id(self.user_manager, self.tenant_manager, secret=TEST_SECRET)

    def test_user_not_found_raises(self):
        token = create_access_token("ghost-user-id", "ghost@example.com", secret=TEST_SECRET)
        with patch.dict(os.environ, {"RAGIFY_MCP_TOKEN": token}, clear=True):
            with self.assertRaises(RuntimeError):
                _resolve_mcp_tenant_id(self.user_manager, self.tenant_manager, secret=TEST_SECRET)

    def test_user_with_no_tenants_raises(self):
        user = self.user_manager.create("solo@example.com", "password123", "Solo")
        token = create_access_token(user.id, user.email, secret=TEST_SECRET)
        with patch.dict(os.environ, {"RAGIFY_MCP_TOKEN": token}, clear=True):
            with self.assertRaises(RuntimeError):
                _resolve_mcp_tenant_id(self.user_manager, self.tenant_manager, secret=TEST_SECRET)

    def test_success_returns_first_tenant(self):
        user = self.user_manager.create("owner@example.com", "password123", "Owner")
        tenant = self.tenant_manager.create_tenant("工作区", user.id)
        token = create_access_token(user.id, user.email, secret=TEST_SECRET)
        with patch.dict(os.environ, {"RAGIFY_MCP_TOKEN": token}, clear=True):
            tenant_id = _resolve_mcp_tenant_id(self.user_manager, self.tenant_manager, secret=TEST_SECRET)
        self.assertEqual(tenant_id, tenant.id)


class TestMcpToolsTenantScoping(unittest.TestCase):
    @patch("ragify.mcp_server.server.KBManager")
    def test_list_resources_passes_tenant_id(self, mock_kb_manager_cls):
        mock_manager = mock_kb_manager_cls.return_value
        mock_manager.list_all.return_value = []
        _list_resources("tenant-x")
        mock_manager.list_all.assert_called_once_with("tenant-x")

    @patch("ragify.mcp_server.server.KBManager")
    def test_call_tool_list_kbs_passes_tenant_id(self, mock_kb_manager_cls):
        mock_manager = mock_kb_manager_cls.return_value
        mock_manager.list_all.return_value = []
        _call_tool("ragify_list_kbs", {}, "tenant-y")
        mock_manager.list_all.assert_called_once_with("tenant-y")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify && .venv/bin/python -m unittest tests.test_mcp_server -v 2>&1 | tail -20
```

Expected: `ImportError`（`_resolve_mcp_tenant_id` 还不存在），以及 `_list_resources`/`_call_tool` 的签名跟测试期望的参数个数不一致导致的 `TypeError`。

- [ ] **Step 3: 改写 `ragify/mcp_server/server.py`**

当前顶部 import：
```python
import json
import sys
from typing import Any

from ..agentic.skills import SkillRegistry
from ..core.kb_manager import KBManager
```

改成：
```python
import json
import os
import sys
from typing import Any

import jwt as pyjwt

from ..agentic.skills import SkillRegistry
from ..core.kb_manager import KBManager
from ..core.security import decode_access_token
from ..core.tenant_manager import TenantManager
from ..core.user_manager import UserManager
```

在 `_list_tools` 之前（文件靠前的位置）新增：

```python
def _resolve_mcp_tenant_id(
    user_manager: UserManager | None = None,
    tenant_manager: TenantManager | None = None,
    *,
    secret: str | None = None,
) -> str:
    """启动期解析当前 MCP 会话对应的 tenant_id。读 RAGIFY_MCP_TOKEN 环境变量
    （用户通过已有的 /api/auth/login 拿到的真实 JWT），解码拿 user_id，查用户
    是否存在，再取这个用户所属的第一个工作区。任何一步失败都直接抛异常让
    进程启动失败——不静默降级、不假装能继续跑。

    user_manager/tenant_manager/secret 三个参数只在测试里传（分别用临时
    数据库和固定密钥做确定性验证），生产代码路径永远不传，跟
    ragify/core/security.py 里 create_access_token/decode_access_token 的
    secret 参数是同一个"仅测试用"的设计思路。
    """
    user_manager = user_manager or UserManager()
    tenant_manager = tenant_manager or TenantManager()

    token = os.environ.get("RAGIFY_MCP_TOKEN")
    if not token:
        raise RuntimeError(
            "未设置 RAGIFY_MCP_TOKEN——MCP server 需要一个通过 /api/auth/login "
            "获取的有效登录凭证才能启动"
        )
    try:
        payload = decode_access_token(token, secret=secret)
        user_id = payload["sub"]
    except (pyjwt.PyJWTError, KeyError) as e:
        raise RuntimeError(f"RAGIFY_MCP_TOKEN 无效或已过期：{e}")

    user = user_manager.get_by_id(user_id)
    if user is None:
        raise RuntimeError("RAGIFY_MCP_TOKEN 对应的用户不存在")

    tenants = tenant_manager.list_tenants_for_user(user.id)
    if not tenants:
        raise RuntimeError(f"用户 {user.email} 目前不属于任何工作区，MCP server 无法启动")

    return tenants[0].id
```

`_list_resources` 改成：
```python
def _list_resources(tenant_id: str) -> list[dict]:
    manager = KBManager()
    manager.migrate_json_if_needed()
    kbs = manager.list_all(tenant_id)
    resources: list[dict] = []
    for kb in kbs:
        resources.append({
            "uri": f"ragify://kb/{kb.id}",
            "name": kb.name,
            "description": kb.description or "",
            "mimeType": "application/json",
        })
    return resources
```

`_handle_request` 改成：
```python
def _handle_request(request: dict, tenant_id: str) -> dict | None:
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "tools/list":
        result = _list_tools()
    elif method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = _call_tool(tool_name, arguments, tenant_id)
    elif method == "resources/list":
        result = _list_resources(tenant_id)
    elif method == "skills/list":
        result = _list_skills()
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    return {"jsonrpc": "2.0", "id": req_id, "result": result}
```

`_call_tool` 改成（新增 `tenant_id` 参数，`ragify_query` 分支补上之前完全没做的 `resolve_kb_path` 调用，`ragify_list_kbs` 分支的 `list_all()` 加上 `tenant_id`）：
```python
def _call_tool(name: str, arguments: dict, tenant_id: str) -> Any:
    if name == "ragify_query":
        query = arguments.get("query", "")
        kb_id = arguments.get("kb_id")
        try:
            from ..api.dependencies import KB_LOCK, resolve_kb_path
            manager = KBManager()
            with KB_LOCK:
                resolve_kb_path(manager, kb_id, tenant_id)
            from ..agentic.agent import AgenticRAG
            agent = AgenticRAG(kb_id=kb_id)
            result = agent.run(query)
            return result.get("response", "")
        except Exception as e:
            return f"Tool error: {e}"
    elif name == "ragify_list_kbs":
        manager = KBManager()
        manager.migrate_json_if_needed()
        return [{"id": kb.id, "name": kb.name} for kb in manager.list_all(tenant_id)]
    return {"error": f"Unknown tool: {name}"}
```

（`resolve_kb_path` 之前完全没有被 `ragify_query` 调用过——这是 Phase 1 就存在的既有缺口，MCP 工具调用一直靠"上一次请求残留的全局 vectorstore 配置"这种不可靠的方式工作。这次顺便补上，跟 `query.py` 路由用同一个函数、同一把锁，不是 Phase 4 引入的新问题，是趁这次加 tenant_id 的机会一并修掉。）

`run_mcp_server` 改成：
```python
def run_mcp_server() -> None:
    """Run the MCP server on stdio (JSON-RPC 2.0, one request per line)."""
    tenant_id = _resolve_mcp_tenant_id()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _handle_request(request, tenant_id)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
```

`_list_tools`/`_list_skills`/`_handle_request`（除了新增的 `tenant_id` 参数）/`if __name__ == "__main__":` 块，其余内容不变。

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_mcp_server -v
```

Expected: 7 个测试全部 `ok`。

- [ ] **Step 5: 跑全量测试套件确认无回归**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -10
```

Expected: 全部通过，无 FAILED/ERROR（这时 Task 6-9 已经把 `query.py`/`documents.py` 的路由改完，之前提到的跨任务中间态已经结束，全量测试套件应该重新回到全绿）。

- [ ] **Step 6: Commit**

```bash
git add ragify/mcp_server/server.py tests/test_mcp_server.py
git commit -m "feat: MCP server 启动时用 RAGIFY_MCP_TOKEN 解析 tenant_id，补上 ragify_query 缺失的 resolve_kb_path 调用"
```

---

### Task 13: 端到端验证 + 全量测试 + 收尾

**Files:** 无新文件，只做验证

- [ ] **Step 1: 启动完整服务栈**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify
./start.sh start
sleep 3
```

- [ ] **Step 2: 浏览器手动验证登录 + 门禁的完整闭环**

由于这次改动直接影响浏览器端能不能正常使用产品，用真实浏览器走一遍（不是 curl）：

1. 访问 `http://localhost:3000/`，确认被 `middleware.ts` 重定向到 `http://localhost:3000/login`。
2. 在登录页切到"注册"，注册一个新账号（比如 `e2e-browser@example.com` / `password123` / 姓名 `E2E Browser`）。
3. 确认注册成功后跳转回首页 `/`，仪表盘正常显示（知识库列表为空、统计数字为 0，不报错）。
4. 在知识库页面建一个新知识库，确认能成功建、能看到、能删除。
5. 打开浏览器开发者工具，确认请求 `/api/knowledge-bases` 等代理路由时，Cookie 里带着 `ragify_token`，且请求成功（不是 401）。
6. 手动清掉浏览器里的 `ragify_token` cookie（开发者工具 Application/Storage 面板），刷新页面，确认又被重定向回 `/login`。

Expected: 以上 6 步全部符合预期，任何一步失败都要停下来报告具体现象（不要自行修改应用代码去"让它通过"）。

- [ ] **Step 3: curl 验证跨租户隔离（这是本阶段最核心的不变量）**

```bash
# 注册两个独立账号，各自建一个工作区
OWNER_A_TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"e2e-tenant-a@example.com","password":"password123","name":"Tenant A"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
TENANT_A=$(curl -s -X POST http://localhost:8000/api/tenants \
  -H "Content-Type: application/json" -H "Authorization: Bearer $OWNER_A_TOKEN" \
  -d '{"name":"E2E 工作区 A"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

OWNER_B_TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"e2e-tenant-b@example.com","password":"password123","name":"Tenant B"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
TENANT_B=$(curl -s -X POST http://localhost:8000/api/tenants \
  -H "Content-Type: application/json" -H "Authorization: Bearer $OWNER_B_TOKEN" \
  -d '{"name":"E2E 工作区 B"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

# A 建一个知识库
KB_A=$(curl -s -X POST http://localhost:8000/api/tenants/$TENANT_A/kb \
  -H "Content-Type: application/json" -H "Authorization: Bearer $OWNER_A_TOKEN" \
  -d '{"name":"A的知识库"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

# B 用自己的 token，但拼上偷看到的 A 的 kb_id 和 tenant_id，尝试访问
curl -s -o /dev/null -w "B 访问 A 的工作区列表: %{http_code}\n" \
  http://localhost:8000/api/tenants/$TENANT_A/kb -H "Authorization: Bearer $OWNER_B_TOKEN"

curl -s -o /dev/null -w "B 用自己工作区的 tenant_id + A 的 kb_id 查文档: %{http_code}\n" \
  "http://localhost:8000/api/tenants/$TENANT_B/documents?kb_id=$KB_A" -H "Authorization: Bearer $OWNER_B_TOKEN"
```

Expected: 第一个 `403`（B 根本不是 A 工作区的成员）；第二个 `400`（B 在自己的工作区下用 A 的 kb_id 查，`resolve_kb_path` 里 `manager.get(kb_id, tenant_id)` 校验不通过，返回"知识库不存在"，不会读到 A 的任何数据）。

- [ ] **Step 4: curl 验证现有功能完全不受无关影响**

```bash
curl -s -o /dev/null -w "health: %{http_code}\n" http://localhost:3000/api/health
curl -s -o /dev/null -w "auth me (no token): %{http_code}\n" http://localhost:3000/api/auth/me
```

Expected: `200`、`401`——跟 Phase 3 结束时完全一致（`/api/health` 本身从来没有加门禁，这次也不在改动范围内）。

- [ ] **Step 5: 清理测试数据**

```bash
rm -f /tmp/ragify-e2e-*.txt 2>/dev/null || true
```

（这次 E2E 测试用注册接口建的账号/工作区/知识库留在数据库里也无妨，跟 Phase 2/3 的收尾约定一致。）

- [ ] **Step 6: 跑全量 Python 测试套件**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -15
```

Expected: 全部通过，无 FAILED/ERROR。

- [ ] **Step 7: 前端类型检查和 lint（全量）**

```bash
cd frontend && npx tsc --noEmit --pretty false && npx eslint src/app/api src/app/login src/middleware.ts src/lib/current-tenant.ts
```

Expected: 都无输出/无错误。

- [ ] **Step 8: 停止服务，确认工作区干净**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify
./start.sh stop
git status --short
```

Expected: 两个服务都已停止；`git status --short` 只剩下已知的、跟本次任务无关的历史遗留改动（`ragify.egg-info/*`、`__pycache__/*.pyc`、未跟踪的 `CLAUDE.md`、`test_vectorstore/`）——如果发现其他改动，报告出来，不要自行提交或丢弃。

- [ ] **Step 9: 最终确认所有提交都在**

```bash
git log --oneline 3e6f4bb..HEAD
```

（`3e6f4bb` 是 "docs: Phase 4 数据隔离迁移 设计文档" 那个 commit，即 Task 1 开始之前的状态。）

Expected: 能看到本计划 Task 1-13 对应的全部 commit。

---

## 已知问题，记录为后续任务（本计划不处理）

**`KB_LOCK` 锁粒度问题**（Task 7 code review 发现，用户确认先记录、不在 Phase 4 里改）：`KB_LOCK`（`ragify/api/dependencies.py`）是一把跨整个进程的全局锁，Phase 4 之前所有知识库全局共享时这个设计没有问题。租户隔离之后，一个工作区在 `list_kbs` 里列出很多知识库、逐个查文档数时会一直占着这把锁，连带卡住其他工作区完全无关的查询/文档请求——实测 30 个知识库场景下，无关工作区的请求会被卡住约 1.7 秒。这不是 Task 7 引入的新 bug（这个"边循环边占锁"的写法从 Phase 1 第一个 commit 就有），但 Phase 4 的租户隔离让这个问题第一次变得对"无关的人"不公平。

修复方向（不在本计划范围内，留给以后）：把 `KB_LOCK` 改成跟 `TenantManager.create_tenant`/`KBManager.create` 已经用过的按 `tenant_id` 分锁的模式，或者把逐个知识库查文档数这部分挪到锁外面。这个改动会牵动 `kb.py`/`query.py`/`documents.py` 三个文件里所有用到 `KB_LOCK` 的地方，需要等这三个文件的 Phase 4 改造（Task 7-9）全部完成、看清楚锁的完整使用方式之后再统一处理，不适合在某一个路由文件的任务里单独改一半。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-18-phase4-data-isolation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
