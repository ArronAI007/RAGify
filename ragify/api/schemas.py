"""Pydantic 请求体模型，一一对应 frontend route.ts 发过来的 JSON 形状。"""

from pydantic import BaseModel, EmailStr


class CreateKBRequest(BaseModel):
    name: str
    description: str = ""


class QueryRequest(BaseModel):
    query: str
    k: int = 3
    score_threshold: float | None = None
    kb_id: str | None = None


class AgenticQueryRequest(BaseModel):
    query: str
    kb_id: str | None = None
    chat_history: list[dict] | None = None
    max_iterations: int | None = None


class IndexRequest(BaseModel):
    directory_path: str | None = None
    file_paths: list[str] | None = None
    clear_vectorstore: bool | None = None
    kb_id: str | None = None


class ClearIndexRequest(BaseModel):
    kb_id: str | None = None


class DeleteDocRequest(BaseModel):
    kb_id: str
    source: str


class UpdateChunkRequest(BaseModel):
    kb_id: str | None = None
    chunk_id: str
    content: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
