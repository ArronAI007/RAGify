from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_current_user, get_user_manager
from ..schemas import LoginRequest, RegisterRequest
from ...core.security import create_access_token
from ...core.user_manager import User, UserManager

router = APIRouter()


def _user_out(user: User) -> dict:
    return {"id": user.id, "email": user.email, "name": user.name, "created_at": user.created_at}


@router.post("/api/auth/register")
def register(body: RegisterRequest, manager: UserManager = Depends(get_user_manager)) -> dict:
    try:
        user = manager.create(body.email, body.password, body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = create_access_token(user.id, user.email)
    return {"access_token": token, "token_type": "bearer", "user": _user_out(user)}


@router.post("/api/auth/login")
def login(body: LoginRequest, manager: UserManager = Depends(get_user_manager)) -> dict:
    user = manager.verify_credentials(body.email, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    token = create_access_token(user.id, user.email)
    return {"access_token": token, "token_type": "bearer", "user": _user_out(user)}


@router.get("/api/auth/me")
def me(current_user: User = Depends(get_current_user)) -> dict:
    return _user_out(current_user)
