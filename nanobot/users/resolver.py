"""Resolve internal user identities across channels."""

from __future__ import annotations

import asyncio
import json
import secrets
import string
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _account_key(channel: str, sender_id: str) -> str:
    return f"{channel}:{sender_id}"


@dataclass(slots=True)
class LinkConsumeResult:
    """Outcome of consuming a one-time link code."""

    ok: bool
    user_id: str | None = None
    error: str | None = None


class UserResolver:
    """Thread-safe account resolver + link code storage."""

    def __init__(
        self,
        storage_dir: Path,
        *,
        code_ttl_seconds: int = 600,
        code_attempt_limit: int = 5,
    ) -> None:
        self._storage_dir = ensure_dir(storage_dir)
        self._path = self._storage_dir / "user_links.json"
        self._code_ttl_seconds = max(60, int(code_ttl_seconds))
        self._code_attempt_limit = max(1, int(code_attempt_limit))
        self._lock = asyncio.Lock()

    def user_workspace(self, users_root: Path, user_id: str) -> Path:
        """Return the workspace path for a resolved user id."""
        return ensure_dir(users_root / user_id)

    async def lookup(self, channel: str, sender_id: str) -> str | None:
        """Lookup mapped user id for a channel account."""
        account = _account_key(channel, sender_id)
        async with self._lock:
            db = self._load()
            return db.get("accounts", {}).get(account)

    async def resolve_or_create(self, channel: str, sender_id: str) -> str:
        """Resolve existing user id, or create a new one."""
        account = _account_key(channel, sender_id)
        async with self._lock:
            db = self._load()
            accounts = db.setdefault("accounts", {})
            if account in accounts:
                return str(accounts[account])
            user_id = uuid.uuid4().hex
            accounts[account] = user_id
            self._save(db)
            logger.info("Created user mapping {} -> {}", account, user_id)
            return user_id

    async def link_account(self, user_id: str, channel: str, sender_id: str) -> None:
        """Force-link a channel account to the specified internal user."""
        account = _account_key(channel, sender_id)
        async with self._lock:
            db = self._load()
            db.setdefault("accounts", {})[account] = user_id
            self._save(db)
            logger.info("Linked account {} to user {}", account, user_id)

    async def create_link_code(self, user_id: str) -> str:
        """Create a one-time code that can link another account to *user_id*."""
        alphabet = string.ascii_uppercase + string.digits
        for _ in range(20):
            code = "".join(secrets.choice(alphabet) for _ in range(8))
            async with self._lock:
                db = self._load()
                links = db.setdefault("pending_links", {})
                if code in links:
                    continue
                links[code] = {
                    "user_id": user_id,
                    "expires_at": (_now() + timedelta(seconds=self._code_ttl_seconds)).isoformat(),
                    "remaining_attempts": self._code_attempt_limit,
                }
                self._save(db)
                return code
        raise RuntimeError("Failed to generate unique link code")

    async def consume_link_code(self, code: str, channel: str, sender_id: str) -> LinkConsumeResult:
        """Consume and apply a link code for the current channel account."""
        normalized = (code or "").strip().upper()
        if not normalized:
            return LinkConsumeResult(ok=False, error="empty_code")
        account = _account_key(channel, sender_id)
        async with self._lock:
            db = self._load()
            links = db.setdefault("pending_links", {})
            payload = links.get(normalized)
            if not payload:
                return LinkConsumeResult(ok=False, error="invalid_code")

            expires_at = self._parse_ts(payload.get("expires_at"))
            if expires_at is None or expires_at <= _now():
                links.pop(normalized, None)
                self._save(db)
                return LinkConsumeResult(ok=False, error="expired_code")

            attempts = int(payload.get("remaining_attempts", 0))
            if attempts <= 0:
                links.pop(normalized, None)
                self._save(db)
                return LinkConsumeResult(ok=False, error="attempts_exhausted")

            user_id = str(payload.get("user_id") or "")
            if not user_id:
                links.pop(normalized, None)
                self._save(db)
                return LinkConsumeResult(ok=False, error="invalid_payload")

            db.setdefault("accounts", {})[account] = user_id
            links.pop(normalized, None)
            self._save(db)
            return LinkConsumeResult(ok=True, user_id=user_id)

    async def register_failed_link_attempt(self, code: str) -> None:
        """Decrease remaining attempts for a code when validation fails."""
        normalized = (code or "").strip().upper()
        if not normalized:
            return
        async with self._lock:
            db = self._load()
            links = db.setdefault("pending_links", {})
            payload = links.get(normalized)
            if not payload:
                return
            attempts = int(payload.get("remaining_attempts", 0))
            attempts -= 1
            if attempts <= 0:
                links.pop(normalized, None)
            else:
                payload["remaining_attempts"] = attempts
            self._save(db)

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"accounts": {}, "pending_links": {}}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("accounts", {})
                data.setdefault("pending_links", {})
                return data
        except Exception:
            logger.exception("Failed to load user links from {}", self._path)
        return {"accounts": {}, "pending_links": {}}

    def _save(self, payload: dict[str, Any]) -> None:
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _parse_ts(raw: Any) -> datetime | None:
        if not isinstance(raw, str) or not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
