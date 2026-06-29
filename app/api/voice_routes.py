"""
voice_routes.py — speech → structured items for the family-os app.

  POST /voice/grocery   ← multipart audio upload (field name: "audio").

Transcribes Hebrew speech with Whisper, parses it into grocery items by reusing
the existing grocery intent parser, and RETURNS {transcript, items} WITHOUT
writing anything. The family-os app shows the items for review and adds them
itself (through its own optimistic grocery CRUD), so the user can edit/remove
before they land on the list.

Mounted at ROOT (no /api prefix) — the family-os frontend calls
${ASSISTANT_URL}/voice/grocery, the same convention as /telegram/*.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.intent_parser import GroceryIntent, _get_client, parse_intent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

# OpenAI transcription model — reliable, cheap, good Hebrew support.
_TRANSCRIBE_MODEL = "whisper-1"


class VoiceGroceryItem(BaseModel):
    title: str
    qty: str | None = None
    shopping_category: str = "grocery"


class VoiceGroceryResponse(BaseModel):
    transcript: str
    items: list[VoiceGroceryItem]


@router.post("/grocery", response_model=VoiceGroceryResponse)
async def voice_grocery(audio: UploadFile = File(...)) -> VoiceGroceryResponse:
    """Transcribe a Hebrew voice clip and return parsed grocery items (no write)."""
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio")

    client = _get_client()
    try:
        # The OpenAI SDK accepts a (filename, bytes) tuple as the file.
        result = await client.audio.transcriptions.create(
            model=_TRANSCRIBE_MODEL,
            file=(audio.filename or "voice.m4a", data),
            language="he",
        )
    except Exception:  # noqa: BLE001 — surface a clean 502 to the caller
        logger.exception("voice_grocery: transcription failed")
        raise HTTPException(status_code=502, detail="transcription failed")

    transcript = (getattr(result, "text", "") or "").strip()
    if not transcript:
        return VoiceGroceryResponse(transcript="", items=[])

    parsed = await parse_intent(transcript)
    items: list[VoiceGroceryItem] = []
    if isinstance(parsed, GroceryIntent):
        items = [
            VoiceGroceryItem(
                title=it.title,
                qty=it.qty,
                shopping_category=it.shopping_category,
            )
            for it in parsed.items
        ]

    logger.info("voice_grocery: transcript=%r -> %d item(s)", transcript, len(items))
    return VoiceGroceryResponse(transcript=transcript, items=items)
