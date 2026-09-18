from fastapi import FastAPI

from .routers import auth, documents, health, invitations, kb, query, tenants
from ..core.kb_manager import KBManager
from ..core.tenant_manager import TenantManager

app = FastAPI(title="RAGify API")

app.include_router(kb.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(auth.router)
app.include_router(tenants.router)
app.include_router(invitations.router)
app.include_router(health.router)


@app.on_event("startup")
def _migrate_legacy_json_on_startup() -> None:
    KBManager().migrate_json_if_needed()
    TenantManager().migrate_default_tenant_if_needed()
