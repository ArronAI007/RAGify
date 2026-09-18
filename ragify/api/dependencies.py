"""FastAPI 依赖注入 + 并发安全辅助函数。

见本计划文档开头的并发说明：KB_LOCK 用来保护"改全局 vectorstore 配置 +
构造读这个配置的对象"这一小段临界区，调用方式见各 router 文件。
"""

import os
import threading
from pathlib import Path

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import get_config
from ..core.invitation_manager import InvitationManager
from ..core.kb_manager import KBManager
from ..core.security import decode_access_token
from ..core.tenant_manager import Membership, TenantManager
from ..core.user_manager import User, UserManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

KB_LOCK = threading.Lock()


def get_kb_manager() -> KBManager:
    return KBManager()


def resolve_kb_path(manager: KBManager, kb_id: str | None) -> str:
    """解析 kb_id 对应的 persist_directory，并把它写进全局 vectorstore 配置。

    调用方必须已经持有 KB_LOCK。如果 kb_id 为 None，回退到第一个可用知识库。
    """
    if kb_id:
        kb = manager.get(kb_id)
        if kb is None:
            raise ValueError(f"知识库 '{kb_id}' 不存在")
    else:
        all_kbs = manager.list_all()
        if not all_kbs:
            raise ValueError("没有可用知识库，请先创建知识库")
        kb_id = all_kbs[0].id

    persist_dir = manager.get_persist_dir(kb_id)
    get_config().update("vectorstore.persist_directory", persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir


_bearer_scheme = HTTPBearer(auto_error=False)


def get_user_manager() -> UserManager:
    return UserManager()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    manager: UserManager = Depends(get_user_manager),
) -> User:
    if credentials is None:
        raise HTTPException(status_code=401, detail="缺少登录凭证")
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload["sub"]
    except (jwt.PyJWTError, KeyError):
        raise HTTPException(status_code=401, detail="登录凭证无效或已过期")
    user = manager.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


def get_tenant_manager() -> TenantManager:
    return TenantManager()


def get_invitation_manager() -> InvitationManager:
    return InvitationManager()


def require_membership(
    tenant_id: str,
    current_user: User = Depends(get_current_user),
    manager: TenantManager = Depends(get_tenant_manager),
) -> Membership:
    membership = manager.get_membership(tenant_id, current_user.id)
    if membership is None:
        raise HTTPException(status_code=403, detail="你不是这个工作区的成员")
    return membership


def require_role(*allowed_roles: str):
    def _dependency(membership: Membership = Depends(require_membership)) -> Membership:
        if membership.role not in allowed_roles:
            raise HTTPException(status_code=403, detail="没有权限执行这个操作")
        return membership
    return _dependency
