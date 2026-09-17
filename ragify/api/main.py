from fastapi import FastAPI

from .routers import health, kb, query

app = FastAPI(title="RAGify API")

app.include_router(kb.router)
app.include_router(query.router)
app.include_router(health.router)
