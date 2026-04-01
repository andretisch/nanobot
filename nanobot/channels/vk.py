"""VK (VKontakte) channel implementation using vkbottle."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
from typing import Any

import httpx
from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import Base

VKBOTTLE_AVAILABLE = importlib.util.find_spec("vkbottle") is not None
if VKBOTTLE_AVAILABLE:
    from vkbottle.bot import Bot, Message


class VKConfig(Base):
    """VK channel configuration."""

    enabled: bool = False
    token: str = ""
    allow_from: list[str] = Field(default_factory=list, alias="allowFrom")
    reaction_id: int = Field(default=10, alias="reactionId")
    access_denied_message: str = Field(
        default="Ваш ID: {id}. Этот пользователь не в доверенных. Обратитесь к администратору бота.",
        alias="accessDeniedMessage",
    )


class VKChannel(BaseChannel):
    """VK long-poll channel."""

    name = "vk"
    display_name = "VK"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return VKConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = VKConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: VKConfig = config
        self.bot: Bot | None = None
        self._task: asyncio.Task | None = None

    async def _download_media(self, url: str, ext: str = ".bin") -> str | None:
        """Download media and store it under media/vk."""
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                media_dir = get_media_dir("vk")
                fd, path = tempfile.mkstemp(suffix=ext, prefix="vk_media_", dir=str(media_dir))
                with os.fdopen(fd, "wb") as f:
                    f.write(resp.content)
                return path
        except Exception as e:
            logger.warning("VK media download failed: {}", e)
            return None

    async def _extract_attachments(self, message: Message) -> list[str]:
        """Extract known VK attachments and return local file paths."""
        media: list[str] = []
        for att in getattr(message, "attachments", []) or []:
            photo = getattr(att, "photo", None)
            if photo and getattr(photo, "sizes", None):
                sizes = sorted(photo.sizes, key=lambda s: (getattr(s, "width", 0) * getattr(s, "height", 0)))
                if sizes:
                    path = await self._download_media(getattr(sizes[-1], "url", ""), ext=".jpg")
                    if path:
                        media.append(path)
                continue

            doc = getattr(att, "doc", None)
            doc_url = getattr(doc, "url", None) if doc else None
            if doc_url:
                title = getattr(doc, "title", "") or ""
                ext = os.path.splitext(title)[1] or ".bin"
                path = await self._download_media(doc_url, ext=ext)
                if path:
                    media.append(path)
                continue

            # VK voice messages are delivered as attachment type "audio_message".
            audio_message = getattr(att, "audio_message", None)
            if audio_message:
                ogg_url = getattr(audio_message, "link_ogg", None)
                mp3_url = getattr(audio_message, "link_mp3", None)
                audio_url = ogg_url or mp3_url
                ext = ".ogg" if ogg_url else ".mp3"
                if audio_url:
                    path = await self._download_media(audio_url, ext=ext)
                    if path:
                        media.append(path)
        return media

    async def start(self) -> None:
        if not VKBOTTLE_AVAILABLE:
            logger.error("vkbottle not installed. Run: pip install vkbottle")
            return
        if not self.config.token:
            logger.error("VK token not configured")
            return

        self._running = True
        self.bot = Bot(token=self.config.token)

        @self.bot.on.message()
        async def _on_message(message: Message) -> None:
            if not self._running:
                return

            sender_id = str(getattr(message, "from_id", ""))
            chat_id = str(getattr(message, "peer_id", ""))

            if not self.is_allowed(sender_id):
                try:
                    deny_text = self.config.access_denied_message
                    if "{id}" in deny_text:
                        deny_text = deny_text.replace("{id}", sender_id)
                    await self.bot.api.messages.send(
                        peer_id=int(chat_id),
                        message=deny_text,
                        random_id=0,
                    )
                except Exception:
                    pass
                return

            content = getattr(message, "text", "") or ""
            media = await self._extract_attachments(message)
            if media:
                audio_paths = [p for p in media if p.lower().endswith((".ogg", ".mp3", ".wav", ".m4a"))]
                if audio_paths:
                    transcription = await self.transcribe_audio(audio_paths[0])
                    if transcription:
                        content = (f"{content}\n" if content else "") + f"[transcription: {transcription}]"
                    else:
                        content = (f"{content}\n" if content else "") + f"[voice: {audio_paths[0]}]"

            reply = getattr(message, "reply_message", None)
            reply_text = (getattr(reply, "text", "") or "").strip() if reply else ""
            if reply_text:
                short = reply_text[:100] + ("..." if len(reply_text) > 100 else "")
                content = f"[Reply to: {short}]\n{content}" if content else f"[Reply to: {short}]"

            if not content and not media:
                content = "[empty message]"

            async def _typing_and_reaction() -> None:
                try:
                    if self.config.reaction_id > 0 and getattr(message, "conversation_message_id", None):
                        await self.bot.api.request(
                            "messages.sendReaction",
                            {
                                "peer_id": int(chat_id),
                                "cmid": getattr(message, "conversation_message_id"),
                                "reaction_id": self.config.reaction_id,
                            },
                        )
                except Exception:
                    pass
                try:
                    await self.bot.api.messages.set_activity(peer_id=int(chat_id), type="typing")
                except Exception:
                    pass

            asyncio.create_task(_typing_and_reaction())

            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=content,
                media=media,
                metadata={
                    "message_id": getattr(message, "id", None),
                    "conversation_message_id": getattr(message, "conversation_message_id", None),
                },
            )

        self._task = asyncio.create_task(self.bot.run_polling())
        while self._running and self._task and not self._task.done():
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def send(self, msg: OutboundMessage) -> None:
        if not self._running or not self.bot:
            return
        await self.bot.api.messages.send(peer_id=int(msg.chat_id), message=msg.content, random_id=0)
