"""密码哈希/校验 + JWT 编码/解码。纯函数，不依赖数据库。

JWT 密钥从环境变量 RAGIFY_JWT_SECRET 读取；如果没配置，进程启动后第一次
用到时生成一个随机密钥并打警告日志——这意味着不配置的话每次重启服务
都会让所有人的登录状态失效，这是刻意的安全默认行为（好过留一个硬编码
的、任何人都能伪造 token 的默认密钥）。生产部署应该在 .env 里配置
RAGIFY_JWT_SECRET，让重启服务不影响已登录用户。

create_access_token/decode_access_token 都接受一个可选的 secret 参数，
仅用于测试场景下传入一个已知的固定密钥，跳过上面这套"读环境变量/生成
随机密钥"的逻辑，让测试可以确定性地验证编码-解码往返。生产代码路径
永远不传这个参数，两处调用（签发 token 的 auth.py 和验证 token 的
get_current_user）都会一致地读到同一个进程内缓存的密钥。
"""

import logging
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

logger = logging.getLogger("ragify.core.security")

JWT_ALGORITHM = "HS256"
JWT_EXPIRES_DAYS = 7
BCRYPT_ROUNDS = 12
# bcrypt 只使用密码的前 72 字节，超出部分会被静默忽略——如果不做长度限制，
# 两个仅在第 72 字节之后不同的密码会被判定为"相同密码"，这是一个真实的
# 密码碰撞漏洞，不是理论上的边界情况。
MAX_PASSWORD_BYTES = 72

_fallback_secret: str | None = None
_fallback_secret_lock = threading.Lock()


def _get_jwt_secret() -> str:
    global _fallback_secret
    secret = os.environ.get("RAGIFY_JWT_SECRET")
    if secret:
        return secret
    with _fallback_secret_lock:
        if _fallback_secret is None:
            _fallback_secret = secrets.token_hex(32)
            logger.warning(
                "未设置 RAGIFY_JWT_SECRET，已生成临时密钥——重启服务后所有登录状态"
                "会失效。生产使用请在 .env 里配置 RAGIFY_JWT_SECRET。"
            )
        return _fallback_secret


def _check_password_length(password: str) -> bytes:
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(f"密码过长（最多 {MAX_PASSWORD_BYTES} 字节）")
    return encoded


def hash_password(password: str) -> str:
    encoded = _check_password_length(password)
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    encoded = _check_password_length(password)
    return bcrypt.checkpw(encoded, password_hash.encode("utf-8"))


def create_access_token(user_id: str, email: str, secret: str | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(days=JWT_EXPIRES_DAYS),
    }
    return jwt.encode(payload, secret or _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, secret: str | None = None) -> dict:
    """token 无效/过期/签名不匹配都会抛 jwt.PyJWTError（或其子类）。"""
    return jwt.decode(token, secret or _get_jwt_secret(), algorithms=[JWT_ALGORITHM])
