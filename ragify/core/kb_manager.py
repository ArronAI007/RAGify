import json
import logging
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db.models import KnowledgeBaseRow
from ..db.session import get_session
from .tenant_manager import TenantManager

logger = logging.getLogger("ragify.core.kb_manager")

DEFAULT_VECTORSTORE_DIR = Path("vectorstore")

# create() 先做一次 Python 层的大小写不敏感重名检查，再 insert。数据库的
# UniqueConstraint("tenant_id", "name") 是大小写敏感的（SQLite 默认
# collation），所以两个并发的 create() 调用如果用的是大小写不同但
# lower() 后相同的名字（比如 "KB" 和 "kb"），DB 约束完全不会拦截——两行
# 都能插入成功，Python 层的重名检查在这种并发窗口下形同虚设（已用
# monkeypatch Session.commit 实测复现）。按 tenant_id 加一把进程内锁，把
# "查重复 + insert" 这段逻辑序列化，跟 ragify/core/tenant_manager.py 里
# _get_tenant_lock 解决类似问题用的是同一个模式——但这里用独立的锁注册表，
# 不跟 TenantManager 共享，因为两者保护的是完全不相关的资源，共享一把锁
# 会让"邀请成员"这种操作被"建知识库"无谓地阻塞。
_kb_tenant_locks: dict[str, threading.Lock] = {}
_kb_tenant_locks_guard = threading.Lock()


def _get_kb_tenant_lock(tenant_id: str) -> threading.Lock:
    with _kb_tenant_locks_guard:
        if tenant_id not in _kb_tenant_locks:
            _kb_tenant_locks[tenant_id] = threading.Lock()
        return _kb_tenant_locks[tenant_id]


@dataclass
class KnowledgeBase:
    id: str
    tenant_id: str | None
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

    def create(self, name: str, description: str, tenant_id: str) -> KnowledgeBase:
        name = name.strip()
        if not name:
            raise ValueError("知识库名称不能为空")

        with _get_kb_tenant_lock(tenant_id):
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

    def get_persist_dir(self, tenant_id: str, kb_id: str) -> str:
        return str(self.vectorstore_dir / tenant_id / kb_id)

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

    def migrate_vectorstore_layout_if_needed(self) -> bool:
        with self._session() as session:
            rows = session.query(KnowledgeBaseRow).filter(KnowledgeBaseRow.tenant_id.isnot(None)).all()
            kb_infos = [(row.id, row.tenant_id) for row in rows]

        migrated_any = False
        for kb_id, tenant_id in kb_infos:
            flat_dir = self.vectorstore_dir / kb_id
            nested_dir = self.vectorstore_dir / tenant_id / kb_id
            if nested_dir.exists():
                if flat_dir.exists():
                    # shutil.move 只有走 os.rename 那条路径才是真正原子的；
                    # 遇到跨文件系统/权限问题等 OSError 时它会静默退化成
                    # copytree+rmtree，这条路径不是原子的，中途被打断（磁盘
                    # 满、进程被杀）会留下一个不完整的 nested_dir。此时
                    # flat_dir 和 nested_dir 同时存在，是这次迁移曾经被打断
                    # 过的信号——不能当成"已经迁移完成"直接跳过，否则应用会
                    # 永久加载一个残缺的向量库，且没有任何提示。
                    raise RuntimeError(
                        f"检测到知识库 {kb_id} 的迁移残留：{flat_dir} 和 {nested_dir} "
                        "同时存在（上一次迁移可能被中途打断），需要人工确认后再清理，"
                        "不能自动判断该保留哪一份。"
                    )
                continue
            if not flat_dir.exists():
                continue
            nested_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(flat_dir), str(nested_dir))
            logger.info("已将知识库 %s 的向量库文件从扁平路径迁移到 %s", kb_id, nested_dir)
            migrated_any = True

        return migrated_any
