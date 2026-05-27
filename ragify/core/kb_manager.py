import json
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("ragify.core.kb_manager")

VECTORSTORE_DIR = Path("vectorstore")
KBS_FILE = VECTORSTORE_DIR / "kbs.json"


@dataclass
class KnowledgeBase:
    id: str
    name: str
    description: str
    created_at: str


class KBManager:
    def __init__(self):
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    def migrate_if_needed(self) -> bool:
        """One-time migration: move old flat index files into a KB subdirectory."""
        if KBS_FILE.exists():
            return False

        old_index = VECTORSTORE_DIR / "index.faiss"
        old_pkl = VECTORSTORE_DIR / "index.pkl"

        if old_index.exists() and old_pkl.exists():
            kb_id = "default-" + uuid.uuid4().hex[:8]
            kb_dir = VECTORSTORE_DIR / kb_id
            kb_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_index), str(kb_dir / "index.faiss"))
            shutil.move(str(old_pkl), str(kb_dir / "index.pkl"))

            kb = KnowledgeBase(
                id=kb_id,
                name="默认知识库",
                description="迁移自旧版本数据",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._save([kb])
            logger.info("已迁移旧索引到 KB '%s' (%s)", kb.name, kb_id)
            return True

        self._save([])
        return False

    def create(self, name: str, description: str = "") -> KnowledgeBase:
        name = name.strip()
        if not name:
            raise ValueError("知识库名称不能为空")

        kbs = self._load()
        existing = {kb.name.lower() for kb in kbs}
        if name.lower() in existing:
            raise ValueError(f"知识库 '{name}' 已存在")

        kb_id = uuid.uuid4().hex[:12]
        kb_dir = VECTORSTORE_DIR / kb_id
        kb_dir.mkdir(parents=True, exist_ok=True)

        kb = KnowledgeBase(
            id=kb_id,
            name=name,
            description=description.strip(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        kbs.append(kb)
        self._save(kbs)
        logger.info("创建知识库 '%s' (%s)", name, kb_id)
        return kb

    def delete(self, kb_id: str) -> bool:
        kbs = self._load()
        target = next((kb for kb in kbs if kb.id == kb_id), None)
        if target is None:
            return False

        kb_dir = VECTORSTORE_DIR / kb_id
        if kb_dir.exists():
            shutil.rmtree(str(kb_dir))

        kbs = [kb for kb in kbs if kb.id != kb_id]
        self._save(kbs)
        logger.info("删除知识库 '%s' (%s)", target.name, kb_id)
        return True

    def list_all(self) -> list[KnowledgeBase]:
        return self._load()

    def get(self, kb_id: str) -> KnowledgeBase | None:
        return next((kb for kb in self._load() if kb.id == kb_id), None)

    def get_persist_dir(self, kb_id: str) -> str:
        return str(VECTORSTORE_DIR / kb_id)

    def update_doc_count(self, kb_id: str, count: int) -> None:
        """Called after indexing to update doc_count in metadata (best-effort)."""
        # doc_count is derived from get_document_count() at query time,
        # so this is a no-op — kept for future use if we want cached counts.
        pass

    # ---- private ----

    def _load(self) -> list[KnowledgeBase]:
        if not KBS_FILE.exists():
            return []
        try:
            data = json.loads(KBS_FILE.read_text(encoding="utf-8"))
            return [
                KnowledgeBase(
                    id=item["id"],
                    name=item["name"],
                    description=item.get("description", ""),
                    created_at=item.get("created_at", ""),
                )
                for item in data.get("kbs", [])
            ]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("解析 kbs.json 失败: %s", e)
            return []

    def _save(self, kbs: list[KnowledgeBase]) -> None:
        data = {
            "kbs": [
                {
                    "id": kb.id,
                    "name": kb.name,
                    "description": kb.description,
                    "created_at": kb.created_at,
                }
                for kb in kbs
            ]
        }
        # Atomic write: temp file + rename
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=VECTORSTORE_DIR,
            prefix=".kbs_",
            suffix=".tmp",
            delete=False,
        )
        try:
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp.name, str(KBS_FILE))
        except Exception:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)
            raise
