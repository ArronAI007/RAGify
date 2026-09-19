from dotenv import load_dotenv

# 必须在导入任何读取环境变量的模块之前调用——uvicorn 直接跑这个 app 时
# 不会像 CLI（ragify/cli/cli.py 的 initialize_config）那样自动加载 .env，
# 之前 RAGIFY_JWT_SECRET/DASHSCOPE_API_KEY 等配置一直被静默忽略，JWT 签名
# 用的是进程内随机生成的临时密钥，每次重启服务都会让所有人的登录状态失效。
load_dotenv()

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
    tenant_manager = TenantManager()
    tenant_manager.migrate_default_tenant_if_needed()
    KBManager().migrate_tenant_id_if_needed(tenant_manager)
    KBManager().migrate_vectorstore_layout_if_needed()
