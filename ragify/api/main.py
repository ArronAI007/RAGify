from fastapi import FastAPI

from .routers import health, kb

app = FastAPI(title="RAGify API")

app.include_router(kb.router)
app.include_router(health.router)
