from fastapi import APIRouter

from ...config import get_config

router = APIRouter()


@router.get("/api/health")
def get_health() -> dict:
    cfg = get_config()
    return {
        "status": "healthy",
        "version": cfg.get("base.version", "0.1.0"),
        "llm_provider": cfg.get("llm.provider", "unknown"),
        "vectorstore_type": cfg.get("vectorstore.type", "unknown"),
    }
