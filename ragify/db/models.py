"""SQLAlchemy ORM models. Phase 1 只有 KnowledgeBaseRow 一张表——
不加 tenant_id/owner_id 之类的列，那是 Phase 4（数据隔离迁移）的职责，
账户/租户模型设计出来之后再加对应的外键和迁移脚本。
"""

from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class KnowledgeBaseRow(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False, unique=True)
    description: Mapped[str] = mapped_column(nullable=False, default="")
    created_at: Mapped[str] = mapped_column(nullable=False)


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
