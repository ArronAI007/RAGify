from fastapi import FastAPI

from .routers import health

app = FastAPI(title="RAGify API")

app.include_router(health.router)
