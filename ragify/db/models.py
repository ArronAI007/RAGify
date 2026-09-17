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
