"""邀请管理。跟 ragify/core/tenant_manager.py 的 TenantManager 同一个模式：
DB 驱动、session-per-call、构造函数接受可选的 database_url 用于测试隔离。
单独一个文件（而不是塞进 TenantManager）是因为邀请是独立的生命周期
（pending/accepted/revoked 状态机），跟"正式成员关系"的 CRUD 职责不同。
"""

import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import update
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
            result = session.execute(
                update(TenantInvitationRow)
                .where(TenantInvitationRow.id == invitation_id, TenantInvitationRow.status == "pending")
                .values(status="revoked")
            )
            if result.rowcount == 0:
                raise ValueError("这个邀请已经被处理过了，无法撤销")
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

            # 上面这些检查和下面的写入之间没有原子性——两个并发请求可能都读到
            # status=="pending"、都通过校验。用条件 UPDATE（WHERE status =
            # 'pending'）代替"读 status 再直接赋值"，让状态迁移本身是一次
            # 原子的比较并交换：谁的 UPDATE 先提交，rowcount 就是 1；后到的
            # 请求这时候再读到的 status 已经不是 pending，UPDATE 会匹配 0 行。
            # 这不只是同一个邀请被"accept 和 revoke 同时调用"的问题——同一个
            # token 被两个不同的 user_id 并发 accept 时，(tenant_id, user_id)
            # 唯一约束根本不会冲突，下面的 IntegrityError 兜底完全不会触发，
            # 靠它是不够的（这两种场景都用真实并发线程实测过，之前的写法
            # 100% 复现"两次都成功"）。
            result = session.execute(
                update(TenantInvitationRow)
                .where(TenantInvitationRow.id == row.id, TenantInvitationRow.status == "pending")
                .values(status="accepted")
            )
            if result.rowcount == 0:
                raise ValueError("这个邀请已经被处理过了")

            session.add(TenantAccountJoinRow(
                id=uuid.uuid4().hex[:12], tenant_id=row.tenant_id, user_id=user_id,
                role=row.role, created_at=datetime.now(timezone.utc).isoformat(),
            ))
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
