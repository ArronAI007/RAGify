"""add tenant_id to knowledge_bases

Revision ID: 5748de7e01a6
Revises: b50496f6382f
Create Date: 2026-09-18 14:33:06.846001

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5748de7e01a6'
down_revision: Union[str, Sequence[str], None] = 'b50496f6382f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # knowledge_bases.name 原来的唯一约束是内联的 UNIQUE(name)，SQLite
    # 反射后它是匿名约束（constraint.name 为 None）。batch_alter_table
    # 的 drop_constraint() 按名字查找 named_constraints 字典，匿名约束
    # 不在其中，传任何名字都会报 "No such constraint" / "Constraint
    # must have a name"（已用真实 SQLite 库文件验证）。autogenerate 也
    # 因为同样的原因，没能识别出这个旧约束需要被删除。
    # 用 copy_from 显式给出迁移前的表结构（不包含这个匿名约束）作为
    # batch 重建的起点，让它在重建后的新表里自然消失。
    old_table = sa.Table(
        "knowledge_bases",
        sa.MetaData(),
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
    )
    with op.batch_alter_table(
        "knowledge_bases", schema=None, copy_from=old_table
    ) as batch_op:
        batch_op.add_column(sa.Column("tenant_id", sa.String(), nullable=True))
        batch_op.create_unique_constraint(
            "uq_knowledge_bases_tenant_id_name", ["tenant_id", "name"]
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("knowledge_bases", schema=None) as batch_op:
        batch_op.drop_constraint("uq_knowledge_bases_tenant_id_name", type_="unique")
        batch_op.create_unique_constraint("knowledge_bases_name_key", ["name"])
        batch_op.drop_column("tenant_id")
