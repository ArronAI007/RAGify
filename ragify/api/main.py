from fastapi import FastAPI

from .routers import auth, documents, health, kb, query
from ..core.kb_manager import KBManager

app = FastAPI(title="RAGify API")

app.include_router(kb.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(auth.router)
app.include_router(health.router)


@app.on_event("startup")
def _migrate_legacy_json_on_startup() -> None:
    KBManager().migrate_json_if_needed()
