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
