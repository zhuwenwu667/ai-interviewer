"""训练模式认证：scrypt 密码校验 + HMAC 无状态短期令牌（全部标准库实现）。

设计约束（与面试模式隔离）：
- 密码只存 scrypt 哈希（``.env`` 配置 ``TRAIN_PASSWORD_HASH``），格式
  ``scrypt$N$r$p$salt_hex$hash_hex``；校验用 ``hmac.compare_digest`` 防时序侧信道；
- 令牌为 HMAC-SHA256 签名的无状态票据：``trn1.<b64url(payload)>.<b64url(sig)>``，
  payload 含 ``scope="train"`` 与过期时间——``trn1.`` 前缀 + scope 双重标记，
  与面试 session_id（uuid hex）零命名空间重叠，不可互换；
- 单账号（``TRAIN_USERNAME``），不做注册体系；公网当前为明文 HTTP，
  训练密码必须独立设置、严禁与其他场景复用（README 有显式提醒）。

生成密码哈希：``uv run python -m app.train.auth <密码>``。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sys
import time

# scrypt 工作参数：N=2^14（约 16MB 内存、单机数十毫秒级），可随硬件上调
_SCRYPT_N = 1 << 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16

TOKEN_PREFIX = "trn1."
TOKEN_SCOPE = "train"


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    """生成 ``scrypt$N$r$p$salt_hex$hash_hex`` 格式的密码哈希（随机盐）。"""
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """按存储参数重算并常量时间比对；格式非法/参数不符一律 False。"""
    try:
        algo, n, r, p, salt_hex, hash_hex = stored.split("$", 5)
        if algo != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode(),
            salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), dklen=_SCRYPT_DKLEN,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), hash_hex)


def _sign(payload_b64: str, secret: str) -> str:
    sig = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).digest()
    return _b64e(sig)


def issue_token(username: str, secret: str, ttl_seconds: int, *, now: float | None = None) -> str:
    """签发 ``trn1.<payload>.<sig>`` 令牌；payload 为 JSON（scope/username/exp）。"""
    exp = int((time.time() if now is None else now) + ttl_seconds)
    payload = _b64e(
        json.dumps(
            {"scope": TOKEN_SCOPE, "u": username, "exp": exp},
            separators=(",", ":"),
        ).encode()
    )
    return f"{TOKEN_PREFIX}{payload}.{_sign(payload, secret)}"


def verify_token(token: str, secret: str, *, now: float | None = None) -> str | None:
    """校验令牌：前缀/签名/scope/过期四项全过才返回用户名，否则 None。"""
    if not token.startswith(TOKEN_PREFIX):
        return None
    body = token[len(TOKEN_PREFIX):]
    try:
        payload_b64, sig = body.rsplit(".", 1)
    except ValueError:
        return None
    # 按字节做常量时间比较：伪造令牌可能携带非 ASCII 字符（str 比较会直接 TypeError）
    if not hmac.compare_digest(_sign(payload_b64, secret).encode(), sig.encode()):
        return None
    try:
        payload = json.loads(_b64d(payload_b64))
    except (ValueError, json.JSONDecodeError):
        return None
    if payload.get("scope") != TOKEN_SCOPE:
        return None
    exp = payload.get("exp")
    if not isinstance(exp, int) or (time.time() if now is None else now) >= exp:
        return None
    username = payload.get("u")
    return str(username) if username else None


if __name__ == "__main__":
    # 生成训练密码哈希：uv run python -m app.train.auth <密码>
    if len(sys.argv) != 2:
        sys.exit("用法：uv run python -m app.train.auth <密码>")
    print(hash_password(sys.argv[1]))
