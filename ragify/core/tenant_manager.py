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
