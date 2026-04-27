from pathlib import Path

import pytest

from nanobot.users import UserResolver


@pytest.mark.asyncio
async def test_resolve_or_create_is_stable(tmp_path: Path):
    resolver = UserResolver(tmp_path)
    first = await resolver.resolve_or_create("telegram", "42")
    second = await resolver.resolve_or_create("telegram", "42")
    assert first == second
    assert await resolver.lookup("telegram", "42") == first


@pytest.mark.asyncio
async def test_link_code_links_second_account(tmp_path: Path):
    resolver = UserResolver(tmp_path, code_ttl_seconds=600, code_attempt_limit=3)
    user = await resolver.resolve_or_create("telegram", "42")
    code = await resolver.create_link_code(user)

    result = await resolver.consume_link_code(code, "email", "alice@example.com")
    assert result.ok is True
    assert result.user_id == user
    assert await resolver.lookup("email", "alice@example.com") == user


@pytest.mark.asyncio
async def test_invalid_code_records_attempt(tmp_path: Path):
    resolver = UserResolver(tmp_path, code_ttl_seconds=600, code_attempt_limit=1)
    user = await resolver.resolve_or_create("telegram", "42")
    _ = await resolver.create_link_code(user)
    bad = await resolver.consume_link_code("INVALID", "telegram", "100")
    assert bad.ok is False
