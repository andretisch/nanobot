"""Qwen OAuth Provider — free tier via qwen.ai Device Code OAuth."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

import httpx
import json_repair
from loguru import logger
from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

QWEN_OAUTH_BASE = "https://chat.qwen.ai"
QWEN_DEVICE_CODE_URL = f"{QWEN_OAUTH_BASE}/api/v1/oauth2/device/code"
QWEN_TOKEN_URL = f"{QWEN_OAUTH_BASE}/api/v1/oauth2/token"
QWEN_API_BASE = "https://portal.qwen.ai/v1"
QWEN_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
QWEN_SCOPES = "openid profile email model.completion"
# Browser-like headers to avoid Alibaba WAF blocking automated requests
QWEN_OAUTH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://chat.qwen.ai",
    "Referer": "https://chat.qwen.ai/",
}
CREDENTIALS_PATH = Path.home() / ".nanobot" / "qwen_oauth.json"
# Compatibility with Qwen Code CLI
QWEN_CODE_CREDENTIALS = Path.home() / ".qwen" / "oauth_creds.json"


def _pkce_pair() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _load_creds(path: Path | None = None) -> dict[str, Any] | None:
    """Load credentials from file. Tries nanobot path first, then Qwen Code path."""
    for p in (path or CREDENTIALS_PATH, QWEN_CODE_CREDENTIALS):
        if p and p.exists():
            try:
                data = json.loads(p.read_text())
                if data.get("access_token") or data.get("access"):
                    return {
                        "access_token": data.get("access_token") or data.get("access"),
                        "refresh_token": data.get("refresh_token") or data.get("refresh"),
                        "expires_at": data.get("expires_at") or 0,
                    }
            except Exception as e:
                logger.warning("Failed to load Qwen OAuth credentials from {}: {}", p, e)
    return None


def _parse_json(response: httpx.Response) -> dict[str, Any]:
    """Parse JSON response, raising a helpful error if the body is not JSON."""
    try:
        return response.json()
    except json.JSONDecodeError as e:
        body = response.text[:300] if response.text else "(empty)"
        ct = response.headers.get("content-type", "")
        hint = ""
        if "aliyun_waf" in body.lower() or "waf" in body.lower():
            hint = (
                " Alibaba WAF is blocking automated requests. Try: (1) without proxy "
                "(unset https_proxy); (2) run `npx @qwen-code/qwen-code` to login, "
                "nanobot will use ~/.qwen/oauth_creds.json; (3) use Dashscope API key instead."
            )
        raise RuntimeError(
            f"Server returned non-JSON (status={response.status_code}, content-type={ct}). "
            f"Body preview: {body!r}.{hint}"
        ) from e


def _save_creds(access: str, refresh: str, expires_in: int, path: Path | None = None) -> None:
    """Save credentials to file."""
    p = path or CREDENTIALS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({
            "access_token": access,
            "refresh_token": refresh,
            "expires_at": int(time.time()) + expires_in,
        }, indent=2),
        encoding="utf-8",
    )


def _http_client() -> httpx.Client:
    """Create HTTP client with optional proxy from env."""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    return httpx.Client(timeout=30.0, proxy=proxy, trust_env=True)


def _device_code_flow() -> tuple[str, str, int]:
    """Run Device Code + PKCE OAuth flow. Returns (access_token, refresh_token, expires_in)."""
    verifier, challenge = _pkce_pair()

    with _http_client() as client:
        resp = client.post(
            QWEN_DEVICE_CODE_URL,
            data={
                "client_id": QWEN_CLIENT_ID,
                "scope": QWEN_SCOPES,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                **QWEN_OAUTH_HEADERS,
            },
        )
        resp.raise_for_status()
        device = _parse_json(resp)

    user_code = device.get("user_code", "")
    verification_uri = device.get("verification_uri", "https://chat.qwen.ai")
    verification_uri_complete = device.get("verification_uri_complete") or f"{verification_uri}?user_code={user_code}"
    device_code = device["device_code"]
    interval = device.get("interval", 5)
    expires_in = device.get("expires_in", 300)

    print("\n[Qwen OAuth] Open this URL in your browser and authorize:")
    print(f"  {verification_uri_complete}\n")
    print(f"  User code: {user_code}")
    print(f"  (Expires in {expires_in}s, polling every {interval}s)\n")

    deadline = time.time() + expires_in
    while time.time() < deadline:
        time.sleep(interval)
        with _http_client() as client:
            r = client.post(
                QWEN_TOKEN_URL,
                data={
                    "client_id": QWEN_CLIENT_ID,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "code_verifier": verifier,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    **QWEN_OAUTH_HEADERS,
                },
            )
        if r.status_code != 200:
            try:
                err = r.json() if "application/json" in r.headers.get("content-type", "") else {}
            except json.JSONDecodeError:
                err = {"error_description": r.text[:200] or f"HTTP {r.status_code}"}
            code = err.get("error", "")
            if code == "authorization_pending":
                continue
            if code == "slow_down":
                interval += 5
                continue
            raise RuntimeError(err.get("error_description", r.text) or f"OAuth error: {r.status_code}")
        data = _parse_json(r)
        access = data.get("access_token")
        refresh = data.get("refresh_token")
        exp = data.get("expires_in", 3600)
        if not access:
            raise RuntimeError("OAuth response missing access_token")
        if not refresh:
            raise RuntimeError("OAuth response missing refresh_token")
        return access, refresh, exp

    raise RuntimeError("Device code expired. Please run `nanobot provider login qwen-oauth` again.")


def _refresh_token(refresh_token: str) -> tuple[str, str, int]:
    """Refresh access token. Returns (access_token, refresh_token, expires_in)."""
    with _http_client() as client:
        r = client.post(
            QWEN_TOKEN_URL,
            data={
                "client_id": QWEN_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                **QWEN_OAUTH_HEADERS,
            },
        )
        r.raise_for_status()
        data = _parse_json(r)
    access = data.get("access_token")
    new_refresh = data.get("refresh_token") or refresh_token
    exp = data.get("expires_in", 3600)
    if not access:
        raise RuntimeError("Refresh response missing access_token")
    return access, new_refresh, exp


def get_token() -> str:
    """Get current access token, refreshing if needed. For sync callers."""
    creds = _load_creds()
    if not creds:
        raise RuntimeError(
            "Not authenticated. Run: nanobot provider login qwen-oauth"
        )
    access = creds["access_token"]
    refresh = creds.get("refresh_token")
    expires_at = creds.get("expires_at", 0)
    # Refresh if expires within 5 minutes
    if expires_at and time.time() >= expires_at - 300:
        if not refresh:
            raise RuntimeError("Token expired and no refresh token. Run: nanobot provider login qwen-oauth")
        access, new_refresh, exp = _refresh_token(refresh)
        _save_creds(access, new_refresh, exp)
    return access


async def get_token_async() -> str:
    """Get current access token, refreshing if needed. For async callers."""
    return await asyncio.to_thread(get_token)


# portal.qwen.ai expects "coder-model" and "vision-model" (OpenClaw naming)
QWEN_MODEL_ALIASES = {
    "qwen3-coder-plus": "coder-model",
    "qwen3-vl-plus": "vision-model",
}


class QwenOAuthProvider(LLMProvider):
    """Qwen models via qwen.ai OAuth (free tier: ~1000-2000 req/day).
    Supported: coder-model (text), vision-model (text + images).
    Aliases: qwen3-coder-plus → coder-model, qwen3-vl-plus → vision-model.
    """

    def __init__(self, default_model: str = "qwen_oauth/coder-model"):
        super().__init__(api_key=None, api_base=QWEN_API_BASE)
        self.default_model = _resolve_model_name(_strip_model_prefix(default_model))

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        raw = _strip_model_prefix(model or self.default_model)
        model = _resolve_model_name(raw)
        token = await get_token_async()

        client = AsyncOpenAI(api_key=token, base_url=QWEN_API_BASE)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._sanitize_empty_content(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        try:
            resp = await client.chat.completions.create(**kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if "401" in err_str or "invalid" in err_str or "expired" in err_str:
                logger.warning("Qwen OAuth token may be expired. Run: nanobot provider login qwen-oauth")
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

        if not resp.choices:
            return LLMResponse(
                content="Error: API returned empty choices.",
                finish_reason="error",
            )
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = []
        for tc in msg.tool_calls or []:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json_repair.loads(args)
                except Exception:
                    args = {}
            tool_calls.append(
                ToolCallRequest(id=tc.id, name=tc.function.name, arguments=args or {})
            )
        u = resp.usage
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            } if u else {},
        )

    def get_default_model(self) -> str:
        return f"qwen_oauth/{self.default_model}"


def _strip_model_prefix(model: str) -> str:
    if model.startswith("qwen_oauth/") or model.startswith("qwen-oauth/"):
        return model.split("/", 1)[1]
    return model


def _resolve_model_name(model: str) -> str:
    """Map user-friendly names to portal.qwen.ai model IDs."""
    return QWEN_MODEL_ALIASES.get(model, model)


def login_qwen_oauth() -> None:
    """Run interactive Qwen OAuth login and save credentials."""
    access, refresh, expires_in = _device_code_flow()
    _save_creds(access, refresh, expires_in)
    print("✓ Authenticated with Qwen OAuth. Credentials saved.")
