# Phase 3: 租户/工作区 + 角色 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 RAGify 加 Tenant（工作区）/ TenantAccountJoin（成员+角色）/ 邮件邀请三块基础设施，不改动任何现有接口，不做前端 UI。

**Architecture:** 沿用 Phase 1/2 确立的分层模式：`ragify/core/tenant_manager.py`（`TenantManager`，DB 驱动，跟 `UserManager` 同构）+ `ragify/core/invitation_manager.py`（`InvitationManager`，同样模式，单独一个文件因为邀请是独立的生命周期）+ `ragify/core/mailer.py`（纯函数邮件发送）→ `ragify/api/routers/{tenants,invitations}.py`（FastAPI 路由）。角色权限判断分两层：`require_membership`/`require_role` 依赖负责"能不能进这个接口"，`TenantManager`/`InvitationManager` 内部方法负责"ADMIN 能不能操作这个具体目标"（用 Python 内置的 `PermissionError` 表达，路由层转成 403；`ValueError` 表达业务数据错误，转成 400）。

**Tech Stack:** 标准库 `smtplib`（邮件）+ `secrets`（邀请 token），其余复用 Phase 1/2 已经装好的 FastAPI/SQLAlchemy/Alembic/PyJWT 技术栈，不新增任何 pyproject.toml 依赖。

---

### Task 1: 数据模型 + Alembic 迁移

**Files:**
- Modify: `ragify/db/models.py`
- Create: `alembic/versions/xxxx_create_tenant_tables.py`

- [ ] **Step 1: 在 `ragify/db/models.py` 顶部 import 加 `UniqueConstraint`**

当前顶部：
```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
```

改成：
```python
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
```

- [ ] **Step 2: 在文件末尾（`UserRow` 类定义之后）追加三张新表**

```python


class TenantRow(Base):
    """Phase 3：工作区表。跟 UserRow/KnowledgeBaseRow 平级，互不关联——知识库
    的归属是 Phase 4（数据隔离迁移）的职责。
    """
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[str] = mapped_column(nullable=False)


class TenantAccountJoinRow(Base):
    """用户-工作区多对多关联表，带 role 字段。一个 (tenant_id, user_id) 组合
    唯一——同一个人在同一个工作区里只能有一条成员记录。
    """
    __tablename__ = "tenant_account_joins"
    __table_args__ = (UniqueConstraint("tenant_id", "user_id"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(nullable=False)
    user_id: Mapped[str] = mapped_column(nullable=False)
    role: Mapped[str] = mapped_column(nullable=False)  # OWNER/ADMIN/EDITOR/NORMAL/DATASET_OPERATOR
    created_at: Mapped[str] = mapped_column(nullable=False)


class TenantInvitationRow(Base):
    """邀请表，跟 TenantAccountJoinRow 分开——邀请是"还没成为成员"的中间状态，
    生命周期跟正式成员关系不一样，用 status 字符串状态机保留历史。
    """
    __tablename__ = "tenant_invitations"

    id: Mapped[str] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(nullable=False)
    email: Mapped[str] = mapped_column(nullable=False)
    role: Mapped[str] = mapped_column(nullable=False)
    token: Mapped[str] = mapped_column(nullable=False, unique=True)
    invited_by: Mapped[str] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(nullable=False)  # pending/accepted/revoked
    expires_at: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[str] = mapped_column(nullable=False)
```

- [ ] **Step 3: 生成迁移**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify && .venv/bin/alembic revision --autogenerate -m "create tenant tables"
```

Expected: 在 `alembic/versions/` 下生成一个新文件，`down_revision` 自动指向 `b75d2f3bf9f5`（当前最新的 revision，即 users 表迁移），只包含 `tenants`/`tenant_account_joins`/`tenant_invitations` 三张表的 `upgrade()`/`downgrade()`——不应该出现任何对 `users`/`knowledge_bases` 表的改动（如果出现了，说明当前数据库状态跟已有迁移不一致，先跑 `alembic upgrade head` 让数据库追上现有迁移再重新生成）。

- [ ] **Step 4: 检查生成的迁移内容**

打开生成的文件，确认 `upgrade()`/`downgrade()` 内容等价于：

```python
def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "tenant_account_joins",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "user_id"),
    )
    op.create_table(
        "tenant_invitations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("token", sa.String(), nullable=False),
        sa.Column("invited_by", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("expires_at", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token"),
    )


def downgrade() -> None:
    op.drop_table("tenant_invitations")
    op.drop_table("tenant_account_joins")
    op.drop_table("tenants")
```

如果 autogenerate 生成的版本格式不完全一致（比如带了自动生成的注释、表顺序不同），把 `upgrade()`/`downgrade()` 替换成上面这段（`downgrade()` 的顺序刻意反过来，虽然这几张表之间没有真正的数据库外键约束，但保持"后建的表先删"的惯例）。

- [ ] **Step 5: 跑迁移，验证建表成功**

```bash
.venv/bin/alembic upgrade head
.venv/bin/python -c "
import sqlite3
conn = sqlite3.connect('vectorstore/ragify.db')
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
print(tables)
print(conn.execute('PRAGMA table_info(tenant_account_joins)').fetchall())
"
```

Expected: 第一行输出包含 `('tenants',)`、`('tenant_account_joins',)`、`('tenant_invitations',)`（还有 `users`、`knowledge_bases`、`alembic_version`）；第二行输出 `tenant_account_joins` 的 5 列：id/tenant_id/user_id/role/created_at。

- [ ] **Step 6: 清理这次手动验证生成的数据库文件**

```bash
rm -f vectorstore/ragify.db
```

- [ ] **Step 7: Commit**

```bash
git add ragify/db/models.py alembic/versions/
git commit -m "feat: 新增 TenantRow/TenantAccountJoinRow/TenantInvitationRow 模型和迁移"
```

---

### Task 2: `TenantManager` — 创建、列表、成员查询

**Files:**
- Create: `ragify/core/tenant_manager.py`
- Test: `tests/test_tenant_manager.py`

- [ ] **Step 1: 写失败的测试 `tests/test_tenant_manager.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TenantManager 测试
验证工作区的建/查、成员关系查询、角色变更权限矩阵、移除/退出的唯一
OWNER 保护、默认工作区迁移的幂等性。
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.tenant_manager import TenantManager
from ragify.db.models import Base
from ragify.db.session import get_engine


class TestTenantManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = TenantManager(database_url=self.db_url)

    def tearDown(self):
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_tenant_creates_owner_membership(self):
        tenant = self.manager.create_tenant("测试工作区", "user-1")
        self.assertTrue(tenant.id)
        self.assertEqual(tenant.name, "测试工作区")

        membership = self.manager.get_membership(tenant.id, "user-1")
        self.assertIsNotNone(membership)
        self.assertEqual(membership.role, "OWNER")

    def test_create_tenant_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.manager.create_tenant("   ", "user-1")

    def test_get_tenant(self):
        tenant = self.manager.create_tenant("工作区A", "user-1")
        fetched = self.manager.get_tenant(tenant.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "工作区A")

    def test_get_tenant_missing_returns_none(self):
        self.assertIsNone(self.manager.get_tenant("does-not-exist"))

    def test_list_tenants_for_user(self):
        t1 = self.manager.create_tenant("工作区1", "user-1")
        t2 = self.manager.create_tenant("工作区2", "user-1")
        self.manager.create_tenant("工作区3", "user-2")

        tenants = self.manager.list_tenants_for_user("user-1")
        ids = {t.id for t in tenants}
        self.assertEqual(ids, {t1.id, t2.id})

    def test_list_tenants_for_user_empty(self):
        self.assertEqual(self.manager.list_tenants_for_user("nobody"), [])

    def test_get_membership_missing_returns_none(self):
        tenant = self.manager.create_tenant("工作区", "user-1")
        self.assertIsNone(self.manager.get_membership(tenant.id, "user-2"))

    def test_list_members(self):
        tenant = self.manager.create_tenant("工作区", "user-1")
        members = self.manager.list_members(tenant.id)
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0].user_id, "user-1")
        self.assertEqual(members[0].role, "OWNER")

    def test_list_members_empty_tenant(self):
        self.assertEqual(self.manager.list_members("does-not-exist"), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_tenant_manager -v 2>&1 | tail -15
```

Expected: `ModuleNotFoundError: No module named 'ragify.core.tenant_manager'`。

- [ ] **Step 3: 写 `ragify/core/tenant_manager.py`**

```python
"""工作区（Tenant）管理。跟 ragify/core/user_manager.py 的 UserManager 同一个
模式：DB 驱动、session-per-call、构造函数接受可选的 database_url 用于测试
隔离。角色变更/移除/退出相关的方法在 Task 3 里补充。
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..db.models import TenantAccountJoinRow, TenantRow, UserRow
from ..db.session import get_session

VALID_ROLES = {"OWNER", "ADMIN", "EDITOR", "NORMAL", "DATASET_OPERATOR"}
# ADMIN 不能创造或修改跟自己平级或更高的角色——这两档只有 OWNER 能触碰。
ADMIN_RESTRICTED_ROLES = {"OWNER", "ADMIN"}


@dataclass
class Tenant:
    id: str
    name: str
    created_at: str


@dataclass
class Membership:
    tenant_id: str
    user_id: str
    role: str
    created_at: str


class TenantManager:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url

    def _session(self) -> Session:
        return get_session(self.database_url)

    def create_tenant(self, name: str, owner_user_id: str) -> Tenant:
        name = name.strip()
        if not name:
            raise ValueError("工作区名称不能为空")

        with self._session() as session:
            tenant_id = uuid.uuid4().hex[:12]
            created_at = datetime.now(timezone.utc).isoformat()
            session.add(TenantRow(id=tenant_id, name=name, created_at=created_at))
            session.add(TenantAccountJoinRow(
                id=uuid.uuid4().hex[:12], tenant_id=tenant_id, user_id=owner_user_id,
                role="OWNER", created_at=created_at,
            ))
            session.commit()

        return Tenant(id=tenant_id, name=name, created_at=created_at)

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        with self._session() as session:
            row = session.get(TenantRow, tenant_id)
            if row is None:
                return None
            return Tenant(id=row.id, name=row.name, created_at=row.created_at)

    def list_tenants_for_user(self, user_id: str) -> list[Tenant]:
        with self._session() as session:
            rows = (
                session.query(TenantRow)
                .join(TenantAccountJoinRow, TenantAccountJoinRow.tenant_id == TenantRow.id)
                .filter(TenantAccountJoinRow.user_id == user_id)
                .all()
            )
            return [Tenant(id=r.id, name=r.name, created_at=r.created_at) for r in rows]

    def get_membership(self, tenant_id: str, user_id: str) -> Membership | None:
        with self._session() as session:
            row = (
                session.query(TenantAccountJoinRow)
                .filter(
                    TenantAccountJoinRow.tenant_id == tenant_id,
                    TenantAccountJoinRow.user_id == user_id,
                )
                .first()
            )
            if row is None:
                return None
            return Membership(tenant_id=row.tenant_id, user_id=row.user_id, role=row.role, created_at=row.created_at)

    def list_members(self, tenant_id: str) -> list[Membership]:
        with self._session() as session:
            rows = session.query(TenantAccountJoinRow).filter(TenantAccountJoinRow.tenant_id == tenant_id).all()
            return [
                Membership(tenant_id=r.tenant_id, user_id=r.user_id, role=r.role, created_at=r.created_at)
                for r in rows
            ]
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_tenant_manager -v
```

Expected: 9 个测试全部 `ok`。

- [ ] **Step 5: Commit**

```bash
git add ragify/core/tenant_manager.py tests/test_tenant_manager.py
git commit -m "feat: 新增 TenantManager（建工作区、查工作区、查成员关系）"
```

---

### Task 3: `TenantManager` — 角色变更、移除成员、退出、删除工作区

**Files:**
- Modify: `ragify/core/tenant_manager.py`
- Test: `tests/test_tenant_manager.py`

- [ ] **Step 1: 在 `tests/test_tenant_manager.py` 里补充失败的测试**

在 `TestTenantManager` 类末尾（`test_list_members_empty_tenant` 之后）追加：

```python

    def test_update_member_role_by_owner(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "user-2", "NORMAL")
        updated = self.manager.update_member_role(tenant.id, "user-2", "EDITOR", acting_role="OWNER")
        self.assertEqual(updated.role, "EDITOR")

    def test_update_member_role_invalid_role_raises_value_error(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        with self.assertRaises(ValueError):
            self.manager.update_member_role(tenant.id, "owner-1", "SUPERUSER", acting_role="OWNER")

    def test_update_member_role_missing_member_raises_value_error(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        with self.assertRaises(ValueError):
            self.manager.update_member_role(tenant.id, "ghost", "NORMAL", acting_role="OWNER")

    def test_admin_cannot_promote_to_admin(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "user-2", "NORMAL")
        with self.assertRaises(PermissionError):
            self.manager.update_member_role(tenant.id, "user-2", "ADMIN", acting_role="ADMIN")

    def test_admin_cannot_modify_another_admin(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "admin-2", "ADMIN")
        with self.assertRaises(PermissionError):
            self.manager.update_member_role(tenant.id, "admin-2", "NORMAL", acting_role="ADMIN")

    def test_admin_can_promote_normal_to_editor(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "user-2", "NORMAL")
        updated = self.manager.update_member_role(tenant.id, "user-2", "EDITOR", acting_role="ADMIN")
        self.assertEqual(updated.role, "EDITOR")

    def test_cannot_demote_sole_owner(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        with self.assertRaises(ValueError):
            self.manager.update_member_role(tenant.id, "owner-1", "ADMIN", acting_role="OWNER")

    def test_can_demote_owner_when_another_owner_exists(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "owner-2", "OWNER")
        updated = self.manager.update_member_role(tenant.id, "owner-1", "ADMIN", acting_role="OWNER")
        self.assertEqual(updated.role, "ADMIN")

    def test_remove_member_by_owner(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "user-2", "NORMAL")
        self.manager.remove_member(tenant.id, "user-2", acting_role="OWNER")
        self.assertIsNone(self.manager.get_membership(tenant.id, "user-2"))

    def test_admin_cannot_remove_owner(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "admin-2", "ADMIN")
        with self.assertRaises(PermissionError):
            self.manager.remove_member(tenant.id, "owner-1", acting_role="ADMIN")

    def test_cannot_remove_sole_owner(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        with self.assertRaises(ValueError):
            self.manager.remove_member(tenant.id, "owner-1", acting_role="OWNER")

    def test_leave_tenant(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "user-2", "NORMAL")
        self.manager.leave_tenant(tenant.id, "user-2")
        self.assertIsNone(self.manager.get_membership(tenant.id, "user-2"))

    def test_sole_owner_cannot_leave(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        with self.assertRaises(ValueError):
            self.manager.leave_tenant(tenant.id, "owner-1")

    def test_owner_can_leave_when_another_owner_exists(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "owner-2", "OWNER")
        self.manager.leave_tenant(tenant.id, "owner-1")
        self.assertIsNone(self.manager.get_membership(tenant.id, "owner-1"))

    def test_delete_tenant(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self.manager.delete_tenant(tenant.id)
        self.assertIsNone(self.manager.get_tenant(tenant.id))
        self.assertEqual(self.manager.list_members(tenant.id), [])

    def test_delete_missing_tenant_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.manager.delete_tenant("does-not-exist")

    def _add_member(self, tenant_id: str, user_id: str, role: str) -> None:
        import uuid as uuid_module
        from datetime import datetime, timezone
        from ragify.db.models import TenantAccountJoinRow
        with self.manager._session() as session:
            session.add(TenantAccountJoinRow(
                id=uuid_module.uuid4().hex[:12], tenant_id=tenant_id, user_id=user_id,
                role=role, created_at=datetime.now(timezone.utc).isoformat(),
            ))
            session.commit()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_tenant_manager -v 2>&1 | tail -20
```

Expected: `AttributeError: 'TenantManager' object has no attribute 'update_member_role'`（以及后续几个方法同理）。

- [ ] **Step 3: 在 `ragify/core/tenant_manager.py` 末尾追加四个方法**

```python

    def update_member_role(self, tenant_id: str, target_user_id: str, new_role: str, acting_role: str) -> Membership:
        if new_role not in VALID_ROLES:
            raise ValueError(f"无效角色 '{new_role}'")
        if acting_role != "OWNER" and new_role in ADMIN_RESTRICTED_ROLES:
            raise PermissionError("ADMIN 不能把成员角色改成 ADMIN 或 OWNER")

        with self._session() as session:
            row = (
                session.query(TenantAccountJoinRow)
                .filter(
                    TenantAccountJoinRow.tenant_id == tenant_id,
                    TenantAccountJoinRow.user_id == target_user_id,
                )
                .first()
            )
            if row is None:
                raise ValueError("该用户不是这个工作区的成员")
            if acting_role != "OWNER" and row.role in ADMIN_RESTRICTED_ROLES:
                raise PermissionError("ADMIN 不能修改 ADMIN 或 OWNER 成员的角色")

            if row.role == "OWNER" and new_role != "OWNER":
                owner_count = (
                    session.query(TenantAccountJoinRow)
                    .filter(TenantAccountJoinRow.tenant_id == tenant_id, TenantAccountJoinRow.role == "OWNER")
                    .count()
                )
                if owner_count <= 1:
                    raise ValueError("工作区至少需要一个 OWNER，请先把 OWNER 转让给别人")

            row.role = new_role
            session.commit()
            return Membership(tenant_id=row.tenant_id, user_id=row.user_id, role=row.role, created_at=row.created_at)

    def remove_member(self, tenant_id: str, target_user_id: str, acting_role: str) -> None:
        with self._session() as session:
            row = (
                session.query(TenantAccountJoinRow)
                .filter(
                    TenantAccountJoinRow.tenant_id == tenant_id,
                    TenantAccountJoinRow.user_id == target_user_id,
                )
                .first()
            )
            if row is None:
                raise ValueError("该用户不是这个工作区的成员")
            if acting_role != "OWNER" and row.role in ADMIN_RESTRICTED_ROLES:
                raise PermissionError("ADMIN 不能移除 ADMIN 或 OWNER 成员")
            if row.role == "OWNER":
                owner_count = (
                    session.query(TenantAccountJoinRow)
                    .filter(TenantAccountJoinRow.tenant_id == tenant_id, TenantAccountJoinRow.role == "OWNER")
                    .count()
                )
                if owner_count <= 1:
                    raise ValueError("工作区至少需要一个 OWNER，不能移除唯一的 OWNER")

            session.delete(row)
            session.commit()

    def leave_tenant(self, tenant_id: str, user_id: str) -> None:
        with self._session() as session:
            row = (
                session.query(TenantAccountJoinRow)
                .filter(
                    TenantAccountJoinRow.tenant_id == tenant_id,
                    TenantAccountJoinRow.user_id == user_id,
                )
                .first()
            )
            if row is None:
                raise ValueError("该用户不是这个工作区的成员")
            if row.role == "OWNER":
                # 尽力而为的检查——不是数据库约束强制。SQLite 在这里不方便表达
                # "每个租户至少一个 OWNER" 这种跨行业务规则的 CHECK 约束，两个
                # OWNER 同时点"退出"这种极窄时间窗口的并发场景理论上仍可能让
                # 工作区失去 OWNER。内部小团队工具场景下这个限制可以接受
                # （YAGNI）；面向高并发/对抗性场景需要再加显式行锁或触发器。
                owner_count = (
                    session.query(TenantAccountJoinRow)
                    .filter(TenantAccountJoinRow.tenant_id == tenant_id, TenantAccountJoinRow.role == "OWNER")
                    .count()
                )
                if owner_count <= 1:
                    raise ValueError("你是这个工作区唯一的 OWNER，请先把 OWNER 转让给别人，或者直接删除工作区")

            session.delete(row)
            session.commit()

    def delete_tenant(self, tenant_id: str) -> None:
        with self._session() as session:
            tenant = session.get(TenantRow, tenant_id)
            if tenant is None:
                raise ValueError("工作区不存在")
            session.query(TenantAccountJoinRow).filter(TenantAccountJoinRow.tenant_id == tenant_id).delete()
            session.delete(tenant)
            session.commit()
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_tenant_manager -v
```

Expected: 25 个测试全部 `ok`。

- [ ] **Step 5: Commit**

```bash
git add ragify/core/tenant_manager.py tests/test_tenant_manager.py
git commit -m "feat: TenantManager 新增角色变更/移除成员/退出/删除工作区（含唯一 OWNER 保护）"
```

---

### Task 4: `TenantManager` — 默认工作区迁移

**Files:**
- Modify: `ragify/core/tenant_manager.py`
- Test: `tests/test_tenant_manager.py`

- [ ] **Step 1: 在 `tests/test_tenant_manager.py` 末尾（`_add_member` 辅助方法之前，`test_delete_missing_tenant_raises_value_error` 之后）追加**

```python

    def test_migrate_default_tenant_creates_default_workspace(self):
        self._insert_user("user-1", "2024-01-01T00:00:00+00:00")
        self._insert_user("user-2", "2024-01-02T00:00:00+00:00")

        migrated = self.manager.migrate_default_tenant_if_needed()

        self.assertTrue(migrated)
        tenants = self.manager.list_tenants_for_user("user-1")
        self.assertEqual(len(tenants), 1)
        self.assertEqual(tenants[0].name, "默认工作区")

        owner_membership = self.manager.get_membership(tenants[0].id, "user-1")
        self.assertEqual(owner_membership.role, "OWNER")
        admin_membership = self.manager.get_membership(tenants[0].id, "user-2")
        self.assertEqual(admin_membership.role, "ADMIN")

    def test_migrate_default_tenant_noop_when_tenant_exists(self):
        self._insert_user("user-1", "2024-01-01T00:00:00+00:00")
        self.manager.create_tenant("已有工作区", "user-1")

        migrated = self.manager.migrate_default_tenant_if_needed()

        self.assertFalse(migrated)
        self.assertEqual(len(self.manager.list_tenants_for_user("user-1")), 1)

    def test_migrate_default_tenant_noop_when_no_users(self):
        self.assertFalse(self.manager.migrate_default_tenant_if_needed())

    def _insert_user(self, user_id: str, created_at: str) -> None:
        from ragify.db.models import UserRow
        with self.manager._session() as session:
            session.add(UserRow(
                id=user_id, email=f"{user_id}@example.com", password_hash="x",
                name=user_id, created_at=created_at,
            ))
            session.commit()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_tenant_manager -v 2>&1 | tail -10
```

Expected: `AttributeError: 'TenantManager' object has no attribute 'migrate_default_tenant_if_needed'`。

- [ ] **Step 3: 在 `ragify/core/tenant_manager.py` 末尾追加**

```python

    def migrate_default_tenant_if_needed(self) -> bool:
        with self._session() as session:
            if session.query(TenantRow).first() is not None:
                return False
            users = session.query(UserRow).order_by(UserRow.created_at.asc()).all()
            if not users:
                return False

            tenant_id = uuid.uuid4().hex[:12]
            created_at = datetime.now(timezone.utc).isoformat()
            session.add(TenantRow(id=tenant_id, name="默认工作区", created_at=created_at))
            for index, user in enumerate(users):
                role = "OWNER" if index == 0 else "ADMIN"
                session.add(TenantAccountJoinRow(
                    id=uuid.uuid4().hex[:12], tenant_id=tenant_id, user_id=user.id,
                    role=role, created_at=created_at,
                ))
            session.commit()
        return True
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_tenant_manager -v
```

Expected: 28 个测试全部 `ok`。

- [ ] **Step 5: 跑现有完整测试套件确认无回归**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -10
```

Expected: 全部通过（现有 129 个 + Task 2-4 累计新增的 28 个 = 157 个），无 FAILED/ERROR。

- [ ] **Step 6: Commit**

```bash
git add ragify/core/tenant_manager.py tests/test_tenant_manager.py
git commit -m "feat: TenantManager 新增默认工作区迁移（幂等，跟 KBManager.migrate_json_if_needed 同模式）"
```

---

### Task 5: `ragify/core/mailer.py`（邀请邮件发送）

**Files:**
- Create: `ragify/core/mailer.py`
- Test: `tests/test_mailer.py`

- [ ] **Step 1: 写失败的测试 `tests/test_mailer.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""send_invitation_email 的纯函数测试，mock smtplib.SMTP，不真的发邮件。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.mailer import send_invitation_email

SMTP_ENV = {
    "SMTP_HOST": "smtp.example.com",
    "SMTP_PORT": "587",
    "SMTP_USER": "bot@example.com",
    "SMTP_PASSWORD": "secret",
    "SMTP_FROM": "noreply@example.com",
}


class TestMailer(unittest.TestCase):
    @patch.dict("os.environ", SMTP_ENV, clear=True)
    @patch("ragify.core.mailer.smtplib.SMTP")
    def test_sends_via_starttls_with_correct_args(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = mock_server

        send_invitation_email(
            "invitee@example.com", "测试工作区", "邀请人", "http://localhost:3000/invitations/abc123"
        )

        mock_smtp_cls.assert_called_once_with("smtp.example.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("bot@example.com", "secret")
        self.assertEqual(mock_server.sendmail.call_count, 1)
        args, _ = mock_server.sendmail.call_args
        self.assertEqual(args[0], "noreply@example.com")
        self.assertEqual(args[1], ["invitee@example.com"])
        self.assertIn("邀请人", args[2])
        self.assertIn("http://localhost:3000/invitations/abc123", args[2])

    @patch.dict("os.environ", {}, clear=True)
    def test_raises_when_smtp_not_configured(self):
        with self.assertRaises(RuntimeError):
            send_invitation_email("x@example.com", "T", "I", "http://x/y")

    @patch.dict("os.environ", {**SMTP_ENV, "SMTP_PASSWORD": ""}, clear=True)
    def test_raises_when_partially_configured(self):
        with self.assertRaises(RuntimeError):
            send_invitation_email("x@example.com", "T", "I", "http://x/y")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_mailer -v 2>&1 | tail -15
```

Expected: `ModuleNotFoundError: No module named 'ragify.core.mailer'`。

- [ ] **Step 3: 写 `ragify/core/mailer.py`**

```python
"""邀请邮件发送。纯函数，用标准库 smtplib + STARTTLS。配置从环境变量读，
未配置时直接抛异常——不像 JWT secret 那样有可以继续跑的临时兜底方案，因为
"假装发送成功但其实没发"会让邀请人以为对方收到了邮件，比明确报错更糟。
"""

import os
import smtplib
from email.mime.text import MIMEText


def send_invitation_email(to_email: str, tenant_name: str, inviter_name: str, invite_url: str) -> None:
    host = os.environ.get("SMTP_HOST")
    port = os.environ.get("SMTP_PORT")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    sender = os.environ.get("SMTP_FROM")
    if not all([host, port, user, password, sender]):
        raise RuntimeError("SMTP 未配置（需要 SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASSWORD/SMTP_FROM）")

    subject = f"{inviter_name} 邀请你加入工作区 \"{tenant_name}\""
    body = f"{inviter_name} 邀请你加入 RAGify 工作区 \"{tenant_name}\"。\n\n点击链接加入：{invite_url}\n\n此链接 7 天内有效。"
    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = to_email

    with smtplib.SMTP(host, int(port)) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(sender, [to_email], message.as_string())
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_mailer -v
```

Expected: 3 个测试全部 `ok`。

- [ ] **Step 5: Commit**

```bash
git add ragify/core/mailer.py tests/test_mailer.py
git commit -m "feat: 新增邀请邮件发送（smtplib + STARTTLS）"
```

---

### Task 6: `InvitationManager`（邀请生命周期）

**Files:**
- Create: `ragify/core/invitation_manager.py`
- Test: `tests/test_invitation_manager.py`

- [ ] **Step 1: 写失败的测试 `tests/test_invitation_manager.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InvitationManager 测试
验证邀请的建/查/撤销/接受，接受时的邮箱匹配、过期判断，以及并发接受时
的唯一约束竞态处理。
"""

import shutil
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.invitation_manager import InvitationManager
from ragify.core.tenant_manager import TenantManager
from ragify.db.models import Base, TenantAccountJoinRow
from ragify.db.session import get_engine, get_session


class TestInvitationManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = InvitationManager(database_url=self.db_url)
        self.tenant_manager = TenantManager(database_url=self.db_url)
        self.tenant = self.tenant_manager.create_tenant("工作区", "owner-1")

    def tearDown(self):
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_invitation(self):
        invitation = self.manager.create_invitation(self.tenant.id, "New@Example.com", "NORMAL", "owner-1")
        self.assertTrue(invitation.id)
        self.assertEqual(invitation.email, "new@example.com")
        self.assertEqual(invitation.status, "pending")
        self.assertTrue(invitation.token)

    def test_create_invitation_empty_email_raises(self):
        with self.assertRaises(ValueError):
            self.manager.create_invitation(self.tenant.id, "   ", "NORMAL", "owner-1")

    def test_list_invitations(self):
        self.manager.create_invitation(self.tenant.id, "a@example.com", "NORMAL", "owner-1")
        self.manager.create_invitation(self.tenant.id, "b@example.com", "EDITOR", "owner-1")
        invitations = self.manager.list_invitations(self.tenant.id)
        self.assertEqual(len(invitations), 2)

    def test_get_by_token(self):
        invitation = self.manager.create_invitation(self.tenant.id, "a@example.com", "NORMAL", "owner-1")
        fetched = self.manager.get_by_token(invitation.token)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.email, "a@example.com")

    def test_get_by_token_missing_returns_none(self):
        self.assertIsNone(self.manager.get_by_token("not-a-real-token"))

    def test_revoke_invitation(self):
        invitation = self.manager.create_invitation(self.tenant.id, "a@example.com", "NORMAL", "owner-1")
        self.manager.revoke_invitation(self.tenant.id, invitation.id)
        fetched = self.manager.get_by_token(invitation.token)
        self.assertEqual(fetched.status, "revoked")

    def test_revoke_missing_invitation_raises(self):
        with self.assertRaises(ValueError):
            self.manager.revoke_invitation(self.tenant.id, "does-not-exist")

    def test_accept_invitation_creates_membership(self):
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "EDITOR", "owner-1")
        self.manager.accept_invitation(invitation.token, "user-2", "invitee@example.com")

        membership = self.tenant_manager.get_membership(self.tenant.id, "user-2")
        self.assertIsNotNone(membership)
        self.assertEqual(membership.role, "EDITOR")

        fetched = self.manager.get_by_token(invitation.token)
        self.assertEqual(fetched.status, "accepted")

    def test_accept_invitation_wrong_email_raises_permission_error(self):
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "NORMAL", "owner-1")
        with self.assertRaises(PermissionError):
            self.manager.accept_invitation(invitation.token, "user-2", "someone-else@example.com")

    def test_accept_invitation_missing_token_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.manager.accept_invitation("not-a-real-token", "user-2", "invitee@example.com")

    def test_accept_invitation_already_accepted_raises(self):
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "NORMAL", "owner-1")
        self.manager.accept_invitation(invitation.token, "user-2", "invitee@example.com")
        with self.assertRaises(ValueError):
            self.manager.accept_invitation(invitation.token, "user-3", "invitee@example.com")

    def test_accept_invitation_revoked_raises(self):
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "NORMAL", "owner-1")
        self.manager.revoke_invitation(self.tenant.id, invitation.id)
        with self.assertRaises(ValueError):
            self.manager.accept_invitation(invitation.token, "user-2", "invitee@example.com")

    def test_accept_invitation_expired_raises(self):
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "NORMAL", "owner-1")
        with self._session() as session:
            from ragify.db.models import TenantInvitationRow
            row = session.get(TenantInvitationRow, invitation.id)
            row.expires_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            session.commit()

        with self.assertRaises(ValueError):
            self.manager.accept_invitation(invitation.token, "user-2", "invitee@example.com")

    def test_accept_invitation_already_a_member_raises(self):
        invitation = self.manager.create_invitation(self.tenant.id, "owner-1@example.com", "NORMAL", "owner-1")
        # owner-1 已经是这个工作区的 OWNER 了（create_tenant 时建的）
        with self.assertRaises(ValueError):
            self.manager.accept_invitation(invitation.token, "owner-1", "owner-1@example.com")

    def test_accept_invitation_race_condition_raises_value_error(self):
        """跟 test_kb_manager.py/test_user_manager.py 的竞态测试同一个模式：
        两次并发 accept 同一个邀请（或者两个不同邀请但目标是同一个
        (tenant_id, user_id)）会撞上 TenantAccountJoinRow 的唯一约束，
        必须走 except IntegrityError -> ValueError 翻译，而不是让
        sqlalchemy.exc.IntegrityError 未处理地抛出去。"""
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "NORMAL", "owner-1")
        tenant_id = self.tenant.id
        original_commit = Session.commit
        state = {"injected": False}

        def racing_commit(session_self, *args, **kwargs):
            if not state["injected"]:
                state["injected"] = True
                other_session = get_session(self.db_url)
                try:
                    other_session.add(TenantAccountJoinRow(
                        id=uuid.uuid4().hex[:12], tenant_id=tenant_id, user_id="user-2",
                        role="NORMAL", created_at=datetime.now(timezone.utc).isoformat(),
                    ))
                    original_commit(other_session)
                finally:
                    other_session.close()
            return original_commit(session_self, *args, **kwargs)

        with patch.object(Session, "commit", racing_commit):
            with self.assertRaises(ValueError):
                self.manager.accept_invitation(invitation.token, "user-2", "invitee@example.com")

    def _session(self):
        return self.manager._session()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_invitation_manager -v 2>&1 | tail -15
```

Expected: `ModuleNotFoundError: No module named 'ragify.core.invitation_manager'`。

- [ ] **Step 3: 写 `ragify/core/invitation_manager.py`**

```python
"""邀请管理。跟 ragify/core/tenant_manager.py 的 TenantManager 同一个模式：
DB 驱动、session-per-call、构造函数接受可选的 database_url 用于测试隔离。
单独一个文件（而不是塞进 TenantManager）是因为邀请是独立的生命周期
（pending/accepted/revoked 状态机），跟"正式成员关系"的 CRUD 职责不同。
"""

import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db.models import TenantAccountJoinRow, TenantInvitationRow
from ..db.session import get_session

INVITATION_EXPIRES_DAYS = 7


@dataclass
class Invitation:
    id: str
    tenant_id: str
    email: str
    role: str
    token: str
    invited_by: str
    status: str
    expires_at: str
    created_at: str


class InvitationManager:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url

    def _session(self) -> Session:
        return get_session(self.database_url)

    def create_invitation(self, tenant_id: str, email: str, role: str, invited_by: str) -> Invitation:
        email = email.strip().lower()
        if not email:
            raise ValueError("邮箱不能为空")

        with self._session() as session:
            invitation_id = uuid.uuid4().hex[:12]
            token = secrets.token_urlsafe(32)
            now = datetime.now(timezone.utc)
            created_at = now.isoformat()
            expires_at = (now + timedelta(days=INVITATION_EXPIRES_DAYS)).isoformat()
            session.add(TenantInvitationRow(
                id=invitation_id, tenant_id=tenant_id, email=email, role=role,
                token=token, invited_by=invited_by, status="pending",
                expires_at=expires_at, created_at=created_at,
            ))
            session.commit()
            return Invitation(
                id=invitation_id, tenant_id=tenant_id, email=email, role=role,
                token=token, invited_by=invited_by, status="pending",
                expires_at=expires_at, created_at=created_at,
            )

    def list_invitations(self, tenant_id: str) -> list[Invitation]:
        with self._session() as session:
            rows = session.query(TenantInvitationRow).filter(TenantInvitationRow.tenant_id == tenant_id).all()
            return [self._to_dataclass(r) for r in rows]

    def get_by_token(self, token: str) -> Invitation | None:
        # token 是 secrets.token_urlsafe(32) 生成的随机字符串，这里通过数据库
        # 唯一索引做等值查询——没有在应用层手写字符串比较，查询耗时由索引查找
        # 主导、不依赖 token 内容逐字节匹配的过程，不存在时序侧信道，不需要
        # hmac.compare_digest 这类常量时间比较。
        with self._session() as session:
            row = session.query(TenantInvitationRow).filter(TenantInvitationRow.token == token).first()
            if row is None:
                return None
            return self._to_dataclass(row)

    def revoke_invitation(self, tenant_id: str, invitation_id: str) -> None:
        with self._session() as session:
            row = session.get(TenantInvitationRow, invitation_id)
            if row is None or row.tenant_id != tenant_id:
                raise ValueError("邀请不存在")
            row.status = "revoked"
            session.commit()

    def accept_invitation(self, token: str, user_id: str, user_email: str) -> None:
        with self._session() as session:
            row = session.query(TenantInvitationRow).filter(TenantInvitationRow.token == token).first()
            if row is None:
                raise ValueError("邀请不存在")
            if row.status != "pending":
                raise ValueError("这个邀请已经被处理过了")
            if row.email.strip().lower() != user_email.strip().lower():
                raise PermissionError("这个邀请不是发给当前登录账号的")
            if datetime.now(timezone.utc) > datetime.fromisoformat(row.expires_at):
                raise ValueError("邀请已过期")

            existing = (
                session.query(TenantAccountJoinRow)
                .filter(
                    TenantAccountJoinRow.tenant_id == row.tenant_id,
                    TenantAccountJoinRow.user_id == user_id,
                )
                .first()
            )
            if existing is not None:
                raise ValueError("你已经是这个工作区的成员了")

            session.add(TenantAccountJoinRow(
                id=uuid.uuid4().hex[:12], tenant_id=row.tenant_id, user_id=user_id,
                role=row.role, created_at=datetime.now(timezone.utc).isoformat(),
            ))
            row.status = "accepted"
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                raise ValueError("你已经是这个工作区的成员了")

    @staticmethod
    def _to_dataclass(row: TenantInvitationRow) -> Invitation:
        return Invitation(
            id=row.id, tenant_id=row.tenant_id, email=row.email, role=row.role,
            token=row.token, invited_by=row.invited_by, status=row.status,
            expires_at=row.expires_at, created_at=row.created_at,
        )
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_invitation_manager -v
```

Expected: 15 个测试全部 `ok`。

- [ ] **Step 5: 跑现有完整测试套件确认无回归**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -10
```

Expected: 全部通过（累计 157 + 3 + 15 = 175 个），无 FAILED/ERROR。

- [ ] **Step 6: Commit**

```bash
git add ragify/core/invitation_manager.py tests/test_invitation_manager.py
git commit -m "feat: 新增 InvitationManager（邀请建/查/撤销/接受，含并发接受的竞态处理）"
```

---

### Task 7: 依赖注入扩展

**Files:**
- Modify: `ragify/api/dependencies.py`

- [ ] **Step 1: 修改文件顶部 import**

当前顶部：
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

改成：
```python
import os
import threading
from pathlib import Path

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import get_config
from ..core.invitation_manager import InvitationManager
from ..core.kb_manager import KBManager
from ..core.security import decode_access_token
from ..core.tenant_manager import Membership, TenantManager
from ..core.user_manager import User, UserManager
```

- [ ] **Step 2: 在文件末尾（`get_current_user` 函数之后）追加**

```python


def get_tenant_manager() -> TenantManager:
    return TenantManager()


def get_invitation_manager() -> InvitationManager:
    return InvitationManager()


def require_membership(
    tenant_id: str,
    current_user: User = Depends(get_current_user),
    manager: TenantManager = Depends(get_tenant_manager),
) -> Membership:
    membership = manager.get_membership(tenant_id, current_user.id)
    if membership is None:
        raise HTTPException(status_code=403, detail="你不是这个工作区的成员")
    return membership


def require_role(*allowed_roles: str):
    def _dependency(membership: Membership = Depends(require_membership)) -> Membership:
        if membership.role not in allowed_roles:
            raise HTTPException(status_code=403, detail="没有权限执行这个操作")
        return membership
    return _dependency
```

- [ ] **Step 3: 验证能正常导入**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify && .venv/bin/python -c "
from ragify.api.dependencies import get_tenant_manager, get_invitation_manager, require_membership, require_role
print('ok')
"
```

Expected: 打印 `ok`。

- [ ] **Step 4: 跑现有完整测试套件确认没有破坏其他路由（它们也在这个文件里）**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -10
```

Expected: 全部通过，无 FAILED/ERROR。

- [ ] **Step 5: Commit**

```bash
git add ragify/api/dependencies.py
git commit -m "feat: 新增 get_tenant_manager/get_invitation_manager/require_membership/require_role 依赖"
```

---

### Task 8: Schema 扩展

**Files:**
- Modify: `ragify/api/schemas.py`

- [ ] **Step 1: 在文件末尾（`LoginRequest` 类之后）追加**

```python


class CreateTenantRequest(BaseModel):
    name: str


class UpdateMemberRoleRequest(BaseModel):
    role: str


class CreateInvitationRequest(BaseModel):
    email: EmailStr
    role: str
```

- [ ] **Step 2: 验证能正常导入**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify && .venv/bin/python -c "
from ragify.api.schemas import CreateTenantRequest, UpdateMemberRoleRequest, CreateInvitationRequest
print('ok')
"
```

Expected: 打印 `ok`。

- [ ] **Step 3: Commit**

```bash
git add ragify/api/schemas.py
git commit -m "feat: 新增工作区/角色/邀请相关的请求 schema"
```

---

### Task 9: `/api/tenants/*` 路由

**Files:**
- Create: `ragify/api/routers/tenants.py`
- Modify: `ragify/api/main.py`
- Test: `tests/test_api_tenants.py`

- [ ] **Step 1: 写失败的测试 `tests/test_api_tenants.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/tenants/* 路由测试（工作区建/查、成员管理、邀请管理）。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_invitation_manager, get_tenant_manager, get_user_manager
from ragify.api.main import app
from ragify.core.invitation_manager import InvitationManager
from ragify.core.tenant_manager import TenantManager
from ragify.core.user_manager import UserManager
from ragify.db.models import Base
from ragify.db.session import get_engine


class TestTenantRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.user_manager = UserManager(database_url=self.db_url)
        self.tenant_manager = TenantManager(database_url=self.db_url)
        self.invitation_manager = InvitationManager(database_url=self.db_url)
        app.dependency_overrides[get_user_manager] = lambda: self.user_manager
        app.dependency_overrides[get_tenant_manager] = lambda: self.tenant_manager
        app.dependency_overrides[get_invitation_manager] = lambda: self.invitation_manager
        self.client = TestClient(app)

        self.owner_token = self._register("owner@example.com", "Owner")
        self.other_token = self._register("other@example.com", "Other")

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

    def test_create_tenant_makes_creator_owner(self):
        res = self.client.post("/api/tenants", json={"name": "测试工作区"}, headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 200)
        tenant_id = res.json()["id"]

        members_res = self.client.get(f"/api/tenants/{tenant_id}/members", headers=self._auth(self.owner_token))
        self.assertEqual(members_res.status_code, 200)
        roles = {m["role"] for m in members_res.json()}
        self.assertEqual(roles, {"OWNER"})

    def test_list_my_tenants(self):
        self.client.post("/api/tenants", json={"name": "工作区A"}, headers=self._auth(self.owner_token))
        res = self.client.get("/api/tenants", headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 1)

    def test_non_member_cannot_list_members(self):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]

        res = self.client.get(f"/api/tenants/{tenant_id}/members", headers=self._auth(self.other_token))
        self.assertEqual(res.status_code, 403)

    def test_update_member_role_requires_owner_or_admin(self):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]

        # other 不是成员，改角色应该 403
        other_user_id = self.user_manager.get_by_email("other@example.com").id
        res = self.client.patch(
            f"/api/tenants/{tenant_id}/members/{other_user_id}",
            json={"role": "NORMAL"},
            headers=self._auth(self.other_token),
        )
        self.assertEqual(res.status_code, 403)

    def test_delete_tenant_requires_owner(self):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]

        res = self.client.delete(f"/api/tenants/{tenant_id}", headers=self._auth(self.other_token))
        self.assertEqual(res.status_code, 403)

        res = self.client.delete(f"/api/tenants/{tenant_id}", headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 200)

    def test_sole_owner_cannot_leave(self):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]

        res = self.client.post(f"/api/tenants/{tenant_id}/leave", headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 400)

    @patch("ragify.api.routers.tenants.send_invitation_email")
    def test_create_invitation_sends_email(self, mock_send):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]

        res = self.client.post(
            f"/api/tenants/{tenant_id}/invitations",
            json={"email": "invitee@example.com", "role": "NORMAL"},
            headers=self._auth(self.owner_token),
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["email"], "invitee@example.com")
        mock_send.assert_called_once()

    @patch("ragify.api.routers.tenants.send_invitation_email")
    def test_admin_cannot_invite_as_admin(self, mock_send):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]
        other_user_id = self.user_manager.get_by_email("other@example.com").id
        # 手工把 other 加成 ADMIN，验证 ADMIN 邀请 ADMIN 会被拒绝
        from ragify.db.models import TenantAccountJoinRow
        import uuid
        from datetime import datetime, timezone
        with self.tenant_manager._session() as session:
            session.add(TenantAccountJoinRow(
                id=uuid.uuid4().hex[:12], tenant_id=tenant_id, user_id=other_user_id,
                role="ADMIN", created_at=datetime.now(timezone.utc).isoformat(),
            ))
            session.commit()

        res = self.client.post(
            f"/api/tenants/{tenant_id}/invitations",
            json={"email": "x@example.com", "role": "ADMIN"},
            headers=self._auth(self.other_token),
        )
        self.assertEqual(res.status_code, 403)
        mock_send.assert_not_called()

    @patch("ragify.api.routers.tenants.send_invitation_email")
    def test_list_and_revoke_invitations(self, mock_send):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]
        invite_res = self.client.post(
            f"/api/tenants/{tenant_id}/invitations",
            json={"email": "invitee@example.com", "role": "NORMAL"},
            headers=self._auth(self.owner_token),
        )
        invitation_id = invite_res.json()["id"]

        list_res = self.client.get(f"/api/tenants/{tenant_id}/invitations", headers=self._auth(self.owner_token))
        self.assertEqual(len(list_res.json()), 1)

        revoke_res = self.client.delete(
            f"/api/tenants/{tenant_id}/invitations/{invitation_id}", headers=self._auth(self.owner_token)
        )
        self.assertEqual(revoke_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_api_tenants -v 2>&1 | tail -15
```

Expected: 报错（`ModuleNotFoundError` 或一堆 404），因为 `tenants.py` 路由还不存在。

- [ ] **Step 3: 写 `ragify/api/routers/tenants.py`**

```python
import os

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import (
    get_current_user,
    get_invitation_manager,
    get_tenant_manager,
    require_membership,
    require_role,
)
from ..schemas import CreateInvitationRequest, CreateTenantRequest, UpdateMemberRoleRequest
from ...core.invitation_manager import InvitationManager
from ...core.mailer import send_invitation_email
from ...core.tenant_manager import Membership, Tenant, VALID_ROLES, TenantManager
from ...core.user_manager import User

router = APIRouter()


def _tenant_out(tenant: Tenant) -> dict:
    return {"id": tenant.id, "name": tenant.name, "created_at": tenant.created_at}


def _membership_out(membership: Membership) -> dict:
    return {
        "tenant_id": membership.tenant_id, "user_id": membership.user_id,
        "role": membership.role, "created_at": membership.created_at,
    }


@router.post("/api/tenants")
def create_tenant(
    body: CreateTenantRequest,
    current_user: User = Depends(get_current_user),
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    try:
        tenant = manager.create_tenant(body.name, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _tenant_out(tenant)


@router.get("/api/tenants")
def list_my_tenants(
    current_user: User = Depends(get_current_user),
    manager: TenantManager = Depends(get_tenant_manager),
) -> list[dict]:
    return [_tenant_out(t) for t in manager.list_tenants_for_user(current_user.id)]


@router.get("/api/tenants/{tenant_id}/members")
def list_members(
    tenant_id: str,
    membership: Membership = Depends(require_membership),
    manager: TenantManager = Depends(get_tenant_manager),
) -> list[dict]:
    return [_membership_out(m) for m in manager.list_members(tenant_id)]


@router.patch("/api/tenants/{tenant_id}/members/{user_id}")
def update_member_role(
    tenant_id: str,
    user_id: str,
    body: UpdateMemberRoleRequest,
    membership: Membership = Depends(require_role("OWNER", "ADMIN")),
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    try:
        updated = manager.update_member_role(tenant_id, user_id, body.role, membership.role)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _membership_out(updated)


@router.delete("/api/tenants/{tenant_id}/members/{user_id}")
def remove_member(
    tenant_id: str,
    user_id: str,
    membership: Membership = Depends(require_role("OWNER", "ADMIN")),
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    try:
        manager.remove_member(tenant_id, user_id, membership.role)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True}


@router.post("/api/tenants/{tenant_id}/leave")
def leave_tenant(
    tenant_id: str,
    current_user: User = Depends(get_current_user),
    membership: Membership = Depends(require_membership),
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    try:
        manager.leave_tenant(tenant_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True}


@router.delete("/api/tenants/{tenant_id}")
def delete_tenant(
    tenant_id: str,
    membership: Membership = Depends(require_role("OWNER")),
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    try:
        manager.delete_tenant(tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True}


@router.post("/api/tenants/{tenant_id}/invitations")
def create_invitation(
    tenant_id: str,
    body: CreateInvitationRequest,
    current_user: User = Depends(get_current_user),
    membership: Membership = Depends(require_role("OWNER", "ADMIN")),
    tenant_manager: TenantManager = Depends(get_tenant_manager),
    invitation_manager: InvitationManager = Depends(get_invitation_manager),
) -> dict:
    if body.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"无效角色 '{body.role}'")
    if membership.role != "OWNER" and body.role in {"OWNER", "ADMIN"}:
        raise HTTPException(status_code=403, detail="ADMIN 不能邀请成员为 ADMIN 或 OWNER")

    tenant = tenant_manager.get_tenant(tenant_id)
    invitation = invitation_manager.create_invitation(tenant_id, body.email, body.role, current_user.id)

    frontend_url = os.environ.get("RAGIFY_FRONTEND_URL", "http://localhost:3000")
    invite_url = f"{frontend_url}/invitations/{invitation.token}"
    try:
        send_invitation_email(invitation.email, tenant.name, current_user.name, invite_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"邮件服务未配置或发送失败：{e}")

    return {
        "id": invitation.id, "tenant_id": invitation.tenant_id, "email": invitation.email,
        "role": invitation.role, "status": invitation.status, "expires_at": invitation.expires_at,
        "created_at": invitation.created_at,
    }


@router.get("/api/tenants/{tenant_id}/invitations")
def list_invitations(
    tenant_id: str,
    membership: Membership = Depends(require_role("OWNER", "ADMIN")),
    invitation_manager: InvitationManager = Depends(get_invitation_manager),
) -> list[dict]:
    return [
        {
            "id": inv.id, "tenant_id": inv.tenant_id, "email": inv.email, "role": inv.role,
            "status": inv.status, "expires_at": inv.expires_at, "created_at": inv.created_at,
        }
        for inv in invitation_manager.list_invitations(tenant_id)
    ]


@router.delete("/api/tenants/{tenant_id}/invitations/{invitation_id}")
def revoke_invitation(
    tenant_id: str,
    invitation_id: str,
    membership: Membership = Depends(require_role("OWNER", "ADMIN")),
    invitation_manager: InvitationManager = Depends(get_invitation_manager),
) -> dict:
    try:
        invitation_manager.revoke_invitation(tenant_id, invitation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True}
```

- [ ] **Step 4: 在 `ragify/api/main.py` 里挂载这个 router，并在启动事件里追加默认工作区迁移**

当前 `main.py` 完整内容：

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

改成：

```python
from fastapi import FastAPI

from .routers import auth, documents, health, kb, query, tenants
from ..core.kb_manager import KBManager
from ..core.tenant_manager import TenantManager

app = FastAPI(title="RAGify API")

app.include_router(kb.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(auth.router)
app.include_router(tenants.router)
app.include_router(health.router)


@app.on_event("startup")
def _migrate_legacy_json_on_startup() -> None:
    KBManager().migrate_json_if_needed()
    TenantManager().migrate_default_tenant_if_needed()
```

（`invitations.router` 的挂载放在 Task 10，那时候 `ragify/api/routers/invitations.py` 才会被创建。）

- [ ] **Step 5: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_api_tenants -v
```

Expected: 9 个测试全部 `ok`。

- [ ] **Step 6: 跑全量测试套件确认无回归**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -10
```

Expected: 全部通过（累计 175 + 9 = 184 个）。

- [ ] **Step 7: Commit**

```bash
git add ragify/api/routers/tenants.py ragify/api/main.py tests/test_api_tenants.py
git commit -m "feat: /api/tenants/* 路由（工作区/成员/邀请管理）"
```

---

### Task 10: `/api/invitations/*` 公开路由（查看邀请、接受邀请）

**Files:**
- Create: `ragify/api/routers/invitations.py`
- Modify: `ragify/api/main.py`
- Test: `tests/test_api_invitations.py`

- [ ] **Step 1: 写失败的测试 `tests/test_api_invitations.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/invitations/{token}、/api/invitations/{token}/accept 路由测试。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_invitation_manager, get_tenant_manager, get_user_manager
from ragify.api.main import app
from ragify.core.invitation_manager import InvitationManager
from ragify.core.tenant_manager import TenantManager
from ragify.core.user_manager import UserManager
from ragify.db.models import Base
from ragify.db.session import get_engine


class TestInvitationRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.user_manager = UserManager(database_url=self.db_url)
        self.tenant_manager = TenantManager(database_url=self.db_url)
        self.invitation_manager = InvitationManager(database_url=self.db_url)
        app.dependency_overrides[get_user_manager] = lambda: self.user_manager
        app.dependency_overrides[get_tenant_manager] = lambda: self.tenant_manager
        app.dependency_overrides[get_invitation_manager] = lambda: self.invitation_manager
        self.client = TestClient(app)

        owner_res = self.client.post("/api/auth/register", json={
            "email": "owner@example.com", "password": "password123", "name": "Owner",
        })
        self.owner_token = owner_res.json()["access_token"]
        tenant_res = self.client.post(
            "/api/tenants", json={"name": "工作区"}, headers={"Authorization": f"Bearer {self.owner_token}"}
        )
        self.tenant_id = tenant_res.json()["id"]

    def tearDown(self):
        app.dependency_overrides.clear()
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch("ragify.api.routers.tenants.send_invitation_email")
    def _create_invitation(self, mock_send, email: str = "invitee@example.com", role: str = "NORMAL") -> str:
        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/invitations",
            json={"email": email, "role": role},
            headers={"Authorization": f"Bearer {self.owner_token}"},
        )
        invitation = self.invitation_manager.list_invitations(self.tenant_id)[-1]
        return invitation.token

    def test_get_invitation_by_token(self):
        token = self._create_invitation()
        res = self.client.get(f"/api/invitations/{token}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["email"], "invitee@example.com")
        self.assertEqual(res.json()["tenant_name"], "工作区")

    def test_get_invitation_missing_token_404(self):
        res = self.client.get("/api/invitations/not-a-real-token")
        self.assertEqual(res.status_code, 404)

    def test_accept_invitation_success(self):
        token = self._create_invitation(email="invitee@example.com")
        invitee_res = self.client.post("/api/auth/register", json={
            "email": "invitee@example.com", "password": "password123", "name": "Invitee",
        })
        invitee_token = invitee_res.json()["access_token"]

        res = self.client.post(
            f"/api/invitations/{token}/accept", headers={"Authorization": f"Bearer {invitee_token}"}
        )
        self.assertEqual(res.status_code, 200)

        invitee_id = self.user_manager.get_by_email("invitee@example.com").id
        membership = self.tenant_manager.get_membership(self.tenant_id, invitee_id)
        self.assertIsNotNone(membership)
        self.assertEqual(membership.role, "NORMAL")

    def test_accept_invitation_wrong_account_rejected(self):
        token = self._create_invitation(email="invitee@example.com")
        wrong_res = self.client.post("/api/auth/register", json={
            "email": "someone-else@example.com", "password": "password123", "name": "Someone",
        })
        wrong_token = wrong_res.json()["access_token"]

        res = self.client.post(
            f"/api/invitations/{token}/accept", headers={"Authorization": f"Bearer {wrong_token}"}
        )
        self.assertEqual(res.status_code, 403)

    def test_accept_invitation_without_login_rejected(self):
        token = self._create_invitation()
        res = self.client.post(f"/api/invitations/{token}/accept")
        self.assertEqual(res.status_code, 401)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m unittest tests.test_api_invitations -v 2>&1 | tail -15
```

Expected: 报错（`ModuleNotFoundError` 或 404），因为 `invitations.py` 路由还不存在。

- [ ] **Step 3: 写 `ragify/api/routers/invitations.py`**

```python
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_current_user, get_invitation_manager, get_tenant_manager
from ...core.invitation_manager import InvitationManager
from ...core.tenant_manager import TenantManager
from ...core.user_manager import User

router = APIRouter()


@router.get("/api/invitations/{token}")
def get_invitation(
    token: str,
    invitation_manager: InvitationManager = Depends(get_invitation_manager),
    tenant_manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    invitation = invitation_manager.get_by_token(token)
    if invitation is None:
        raise HTTPException(status_code=404, detail="邀请不存在")
    tenant = tenant_manager.get_tenant(invitation.tenant_id)
    return {
        "tenant_name": tenant.name if tenant else None,
        "email": invitation.email,
        "role": invitation.role,
        "status": invitation.status,
        "expires_at": invitation.expires_at,
    }


@router.post("/api/invitations/{token}/accept")
def accept_invitation(
    token: str,
    current_user: User = Depends(get_current_user),
    invitation_manager: InvitationManager = Depends(get_invitation_manager),
) -> dict:
    try:
        invitation_manager.accept_invitation(token, current_user.id, current_user.email)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True}
```

- [ ] **Step 4: 在 `ragify/api/main.py` 里挂载这个 router**

当前内容（Task 9 之后）：

```python
from fastapi import FastAPI

from .routers import auth, documents, health, kb, query, tenants
from ..core.kb_manager import KBManager
from ..core.tenant_manager import TenantManager

app = FastAPI(title="RAGify API")

app.include_router(kb.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(auth.router)
app.include_router(tenants.router)
app.include_router(health.router)


@app.on_event("startup")
def _migrate_legacy_json_on_startup() -> None:
    KBManager().migrate_json_if_needed()
    TenantManager().migrate_default_tenant_if_needed()
```

改成：

```python
from fastapi import FastAPI

from .routers import auth, documents, health, invitations, kb, query, tenants
from ..core.kb_manager import KBManager
from ..core.tenant_manager import TenantManager

app = FastAPI(title="RAGify API")

app.include_router(kb.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(auth.router)
app.include_router(tenants.router)
app.include_router(invitations.router)
app.include_router(health.router)


@app.on_event("startup")
def _migrate_legacy_json_on_startup() -> None:
    KBManager().migrate_json_if_needed()
    TenantManager().migrate_default_tenant_if_needed()
```

- [ ] **Step 5: 跑测试，确认通过**

```bash
.venv/bin/python -m unittest tests.test_api_invitations -v
```

Expected: 5 个测试全部 `ok`。

- [ ] **Step 6: 跑全量测试套件确认无回归**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -10
```

Expected: 全部通过（累计 184 + 5 = 189 个）。

- [ ] **Step 7: Commit**

```bash
git add ragify/api/routers/invitations.py ragify/api/main.py tests/test_api_invitations.py
git commit -m "feat: /api/invitations/{token} 和 /api/invitations/{token}/accept 公开路由"
```

---

### Task 11: 端到端验证 + 全量测试 + 收尾

**Files:** 无新文件，只做验证

- [ ] **Step 1: 启动完整服务栈**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify
export SMTP_HOST=localhost
export SMTP_PORT=1025
export SMTP_USER=test
export SMTP_PASSWORD=test
export SMTP_FROM=noreply@ragify.local
./start.sh start
sleep 3
```

（`SMTP_*` 只是为了让"发起邀请"这一步不因为"邮件服务未配置"直接 400——本地没有真的跑 SMTP 服务器，所以下面第 4 步的"发起邀请"预期会因为连不上 `localhost:1025` 而报错，这是预期行为，不是 bug，验证的是"配置存在但连不上时会正确报错"而不是"假装发送成功"。）

- [ ] **Step 2: curl 验证工作区建/查、成员管理**

```bash
# 注册两个账号
curl -s -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"e2e-owner@example.com","password":"password123","name":"E2E Owner"}' \
  -c /tmp/ragify-e2e-owner.txt
curl -s -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"e2e-other@example.com","password":"password123","name":"E2E Other"}' \
  -c /tmp/ragify-e2e-other.txt
```

Expected: 两次都 200。

由于目前的 Next.js 代理层只转发 `/api/auth/*`（Phase 2 交付的范围），Phase 3 新增的 `/api/tenants/*`、`/api/invitations/*` 还没有对应的前端代理路由（这是刻意的——Phase 3 是纯后端阶段，前端代理路由要等 Phase 5 有实际 UI 消费这些接口时再加）。所以下面直接对 FastAPI 后端（8000 端口）发请求验证，而不是通过 3000 端口的 Next.js 代理：

```bash
# 从注册响应里取 access_token（注意：这是直接调后端，Set-Cookie 机制这里不适用，
# 用响应体里的 access_token 直接当 Bearer token）
OWNER_TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"e2e-owner@example.com","password":"password123"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s -X POST http://localhost:8000/api/tenants \
  -H "Content-Type: application/json" -H "Authorization: Bearer $OWNER_TOKEN" \
  -d '{"name":"E2E 工作区"}'
```

Expected: 200，返回 `{"id":"...","name":"E2E 工作区","created_at":"..."}`。记下这个 `id` 为 `TENANT_ID`。

```bash
TENANT_ID="<上一步返回的 id>"
curl -s http://localhost:8000/api/tenants/$TENANT_ID/members \
  -H "Authorization: Bearer $OWNER_TOKEN"
```

Expected: `[{"tenant_id":"...","user_id":"...","role":"OWNER","created_at":"..."}]`。

```bash
# 唯一 OWNER 不能退出
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/api/tenants/$TENANT_ID/leave \
  -H "Authorization: Bearer $OWNER_TOKEN"
```

Expected: `400`。

```bash
# 非成员不能查看成员列表
OTHER_TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"e2e-other@example.com","password":"password123"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/api/tenants/$TENANT_ID/members \
  -H "Authorization: Bearer $OTHER_TOKEN"
```

Expected: `403`。

- [ ] **Step 3: curl 验证邀请流程（预期在发信这一步因为连不上本地不存在的 SMTP 服务器而报错，这正是"未配置/不可用时明确报错而不是假装成功"这条设计要求的验证）**

```bash
curl -s -X POST http://localhost:8000/api/tenants/$TENANT_ID/invitations \
  -H "Content-Type: application/json" -H "Authorization: Bearer $OWNER_TOKEN" \
  -d '{"email":"invitee@example.com","role":"NORMAL"}'
```

Expected: 400，`detail` 里包含"邮件服务未配置或发送失败"（因为 `localhost:1025` 上没有真的跑 SMTP 服务器，连接会被拒绝，`send_invitation_email` 抛出的异常被路由层捕获转成 400——这证明了设计里"发信失败必须让调用方明确知道，而不是假装成功"这条要求生效了）。

- [ ] **Step 4: 确认现有功能完全不受影响**

```bash
curl -s -o /dev/null -w "kb list: %{http_code}\n" http://localhost:3000/api/knowledge-bases
curl -s -o /dev/null -w "health: %{http_code}\n" http://localhost:3000/api/health
curl -s -o /dev/null -w "auth me (no token): %{http_code}\n" http://localhost:3000/api/auth/me
```

Expected: 依次 `200`、`200`、`401`——跟 Phase 3 开始之前完全一致。

- [ ] **Step 5: 清理测试数据**

```bash
rm -f /tmp/ragify-e2e-owner.txt /tmp/ragify-e2e-other.txt
```

（测试过程中建的 `e2e-owner@example.com`/`e2e-other@example.com` 用户和"E2E 工作区"留在数据库里也无妨——这套系统里没有任何"物理删除用户"的接口，工作区已经在 Step 3 之前的流程里保留即可，不需要额外清理。）

- [ ] **Step 6: 跑全量 Python 测试套件**

```bash
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -15
```

Expected: 全部通过（189 个测试左右，具体以实际累计数为准），无 FAILED/ERROR。

- [ ] **Step 7: 前端类型检查和 lint（本阶段没有改动任何前端文件，这一步是确认没有意外触碰）**

```bash
cd frontend && npx tsc --noEmit --pretty false && npx eslint src/app/api
```

Expected: 都无输出/无错误。

- [ ] **Step 8: 停止服务，确认工作区干净**

```bash
cd /Users/arron/Desktop/ArronAI/RAGify
./start.sh stop
unset SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASSWORD SMTP_FROM
git status --short
```

Expected: 两个服务都已停止；`git status --short` 只剩下已知的、跟本次任务无关的历史遗留改动（`ragify.egg-info/*`、`__pycache__/*.pyc`、未跟踪的 `CLAUDE.md`、`test_vectorstore/`）——如果发现其他改动，报告出来，不要自行提交或丢弃。

- [ ] **Step 9: 最终确认所有提交都在**

```bash
git log --oneline 2ff7eef..HEAD
```

（`2ff7eef` 是 "docs: Phase 3 租户/工作区 + 角色 设计文档" 那个 commit，即 Task 1 开始之前的状态。）

Expected: 能看到本计划 Task 1-10 对应的全部 commit。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-18-phase3-tenant-roles.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
