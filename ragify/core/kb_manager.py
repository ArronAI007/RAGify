import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db.models import KnowledgeBaseRow
from ..db.session import get_session

logger = logging.getLogger("ragify.core.kb_manager")

DEFAULT_VECTORSTORE_DIR = Path("vectorstore")


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

    def get_persist_dir(self, kb_id: str) -> str:
        return str(self.vectorstore_dir / kb_id)
