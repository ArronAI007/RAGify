"""工作区（Tenant）管理。跟 ragify/core/user_manager.py 的 UserManager 同一个
模式：DB 驱动、session-per-call、构造函数接受可选的 database_url 用于测试
隔离。角色变更/移除/退出相关的方法在 Task 3 里补充。
"""

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..db.models import TenantAccountJoinRow, TenantRow, UserRow
from ..db.session import get_session

# update_member_role/remove_member/leave_tenant 都要做"数一下还有几个 OWNER，
# 再决定能不能改/删"这个 check-then-act。两个并发请求都在对方提交前读到旧的
# OWNER 数量、都通过校验、都提交成功，就可能让工作区同时失去所有 OWNER——
# 这不是理论上的边界情况：用两个线程同时调用 remove_member 各自移除一个
# OWNER，在没有任何人工延迟的情况下就能稳定复现（多次实测命中率在 40% 左右）。
# 这个竞态在 SQLite 和 Postgres 默认隔离级别下都存在（不是 SQLite 特有问题），
# 用进程内锁按 tenant_id 序列化这三个方法，对目前"单进程部署的内部小团队
# 工具"这个定位来说是足够且成本最低的修复；如果未来改成多进程/多 worker
# 部署，这把锁会失效，需要换成数据库级方案（Postgres 下用
# SELECT ... FOR UPDATE 锁 OWNER 行，SQLite 下用显式 BEGIN IMMEDIATE）。
_tenant_locks: dict[str, threading.Lock] = {}
_tenant_locks_guard = threading.Lock()


def _get_tenant_lock(tenant_id: str) -> threading.Lock:
    with _tenant_locks_guard:
        if tenant_id not in _tenant_locks:
            _tenant_locks[tenant_id] = threading.Lock()
        return _tenant_locks[tenant_id]


# migrate_default_tenant_if_needed() 在建默认工作区之前，还没有 tenant_id 可用
# 于加锁——用一个固定的哨兵 key 复用同一套锁机制，防止两次并发调用都读到"还
# 没有 TenantRow"、都各自建一个"默认工作区"，产生两份重复的默认工作区。
_DEFAULT_TENANT_MIGRATION_LOCK_KEY = "__default_tenant_migration__"


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

    def _count_owners(self, session: Session, tenant_id: str) -> int:
        return (
            session.query(TenantAccountJoinRow)
            .filter(TenantAccountJoinRow.tenant_id == tenant_id, TenantAccountJoinRow.role == "OWNER")
            .count()
        )

    def update_member_role(self, tenant_id: str, target_user_id: str, new_role: str, acting_role: str) -> Membership:
        if new_role not in VALID_ROLES:
            raise ValueError(f"无效角色 '{new_role}'")
        if acting_role != "OWNER" and new_role in ADMIN_RESTRICTED_ROLES:
            raise PermissionError("ADMIN 不能把成员角色改成 ADMIN 或 OWNER")

        with _get_tenant_lock(tenant_id):
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
                    if self._count_owners(session, tenant_id) <= 1:
                        raise ValueError("工作区至少需要一个 OWNER，请先把 OWNER 转让给别人")

                row.role = new_role
                session.commit()
                return Membership(tenant_id=row.tenant_id, user_id=row.user_id, role=row.role, created_at=row.created_at)

    def remove_member(self, tenant_id: str, target_user_id: str, acting_role: str) -> None:
        with _get_tenant_lock(tenant_id):
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
                    if self._count_owners(session, tenant_id) <= 1:
                        raise ValueError("工作区至少需要一个 OWNER，不能移除唯一的 OWNER")

                session.delete(row)
                session.commit()

    def leave_tenant(self, tenant_id: str, user_id: str) -> None:
        with _get_tenant_lock(tenant_id):
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
                    # 加了 _get_tenant_lock 之后，这个检查在同一个 tenant_id 下是
                    # 真正互斥的——不会再出现两个并发请求都读到旧计数、都通过
                    # 校验的情况。锁的粒度是进程内的，多进程/多 worker 部署时需要
                    # 换成数据库级方案（见上面 _get_tenant_lock 的说明）。
                    if self._count_owners(session, tenant_id) <= 1:
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

    def migrate_default_tenant_if_needed(self) -> bool:
        with _get_tenant_lock(_DEFAULT_TENANT_MIGRATION_LOCK_KEY):
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
