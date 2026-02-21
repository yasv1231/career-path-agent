from __future__ import annotations

import hashlib
import hmac
import os
import secrets

PBKDF2_ITERATIONS = 120_000


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, password_hash_hex: str) -> bool:
    _, candidate = hash_password(password, salt_hex=salt_hex)
    return hmac.compare_digest(candidate, password_hash_hex)


def generate_token() -> str:
    return secrets.token_urlsafe(48)
