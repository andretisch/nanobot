from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ToolsConfig
from nanobot.providers.base import GenerationSettings, LLMResponse


def _make_loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=512)
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))
    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        restrict_to_workspace=True,
        multi_user_config=ToolsConfig.MultiUserConfig(enabled=True),
    )


@pytest.mark.asyncio
async def test_link_code_shares_user_between_channels(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)

    create = await loop.process_direct(
        "/link",
        session_key="telegram:chat1",
        channel="telegram",
        chat_id="chat1",
        sender_id="alice_tg",
    )
    assert create is not None
    assert "Link code created." in create.content
    code = create.content.split("/link ", 1)[1].split()[0]

    consume = await loop.process_direct(
        f"/link {code}",
        session_key="email:inbox",
        channel="email",
        chat_id="inbox",
        sender_id="alice@mail.example",
    )
    assert consume is not None
    assert "linked successfully" in consume.content.lower()

    user_a = await loop.user_resolver.lookup("telegram", "alice_tg")
    user_b = await loop.user_resolver.lookup("email", "alice@mail.example")
    assert user_a is not None and user_b is not None
    assert user_a == user_b

    user_sessions = tmp_path / "users" / user_a / "sessions"
    assert user_sessions.exists()


@pytest.mark.asyncio
async def test_different_accounts_get_isolated_workspaces_and_tool_roots(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)

    msg_a = InboundMessage(channel="telegram", sender_id="user_a", chat_id="a", content="hi")
    msg_b = InboundMessage(channel="telegram", sender_id="user_b", chat_id="b", content="hi")
    runtime_a = await loop._runtime_for_message(msg_a)
    runtime_b = await loop._runtime_for_message(msg_b)

    assert runtime_a.user_id is not None and runtime_b.user_id is not None
    assert runtime_a.user_id != runtime_b.user_id
    assert runtime_a.workspace != runtime_b.workspace

    read_a = runtime_a.tools.get("read_file")
    read_b = runtime_b.tools.get("read_file")
    assert read_a is not None and read_b is not None
    assert getattr(read_a, "_allowed_dir", None) == runtime_a.workspace
    assert getattr(read_b, "_allowed_dir", None) == runtime_b.workspace

    await loop.process_direct(
        "hello from A",
        session_key="telegram:a",
        channel="telegram",
        chat_id="a",
        sender_id="user_a",
    )
    await loop.process_direct(
        "hello from B",
        session_key="telegram:b",
        channel="telegram",
        chat_id="b",
        sender_id="user_b",
    )

    a_sessions = runtime_a.workspace / "sessions"
    b_sessions = runtime_b.workspace / "sessions"
    assert any(a_sessions.glob("*.jsonl"))
    assert any(b_sessions.glob("*.jsonl"))


@pytest.mark.asyncio
async def test_user_tools_cannot_read_another_users_workspace(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)

    msg_a = InboundMessage(channel="telegram", sender_id="user_a", chat_id="a", content="hi")
    msg_b = InboundMessage(channel="telegram", sender_id="user_b", chat_id="b", content="hi")
    runtime_a = await loop._runtime_for_message(msg_a)
    runtime_b = await loop._runtime_for_message(msg_b)

    secret = runtime_a.workspace / "secret.txt"
    secret.write_text("top-secret-a", encoding="utf-8")

    result = await runtime_b.tools.execute("read_file", {"path": str(secret)})
    assert isinstance(result, str)
    assert "Error" in result
    assert "outside allowed directory" in result
