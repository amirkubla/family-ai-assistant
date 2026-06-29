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

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.grocery_categorizer import categorize_grocery
from app.services.intent_parser import GroceryIntent, _get_client, parse_intent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

# OpenAI transcription model. gpt-4o-transcribe is markedly better at Hebrew
# than whisper-1. The prompt biases it toward grocery vocabulary.
_TRANSCRIBE_MODEL = "gpt-4o-transcribe"
_TRANSCRIBE_PROMPT = (
    "רשימת קניות לבית בעברית. דוגמאות: חלב, ביצים, לחם, עגבניות, גבינה, "
    "יוגורט, נייר טואלט, סבון, שמן, אורז, פסטה, בננות."
)


class VoiceGroceryItem(BaseModel):
    title: str
    qty: str | None = None
    shopping_category: str = "grocery"
    subcategory: str | None = None


class VoiceGroceryResponse(BaseModel):
    transcript: str
    items: list[VoiceGroceryItem]


@router.post("/grocery", response_model=VoiceGroceryResponse)
async def voice_grocery(
    audio: UploadFile = File(...),
    subcategories: str | None = Form(None),
) -> VoiceGroceryResponse:
    """Transcribe a Hebrew clip → parsed + categorized grocery items (no write).

    `subcategories` (optional) is a JSON taxonomy
    {"grocery":[...],"home":[...],"health":[...]} of the family's sub-category
    names; when present, each item is assigned a main category + sub-category
    from it. The app reviews and adds the items.
    """
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
            prompt=_TRANSCRIBE_PROMPT,
        )
    except Exception:  # noqa: BLE001 — surface a clean 502 to the caller
        logger.exception("voice_grocery: transcription failed")
        raise HTTPException(status_code=502, detail="transcription failed")

    transcript = (getattr(result, "text", "") or "").strip()
    if not transcript:
        return VoiceGroceryResponse(transcript="", items=[])

    parsed = await parse_intent(transcript)
    parsed_items = list(parsed.items) if isinstance(parsed, GroceryIntent) else []
    if not parsed_items:
        return VoiceGroceryResponse(transcript=transcript, items=[])

    # Optional family taxonomy → LLM category + sub-category per item.
    taxonomy: dict[str, list[str]] = {}
    if subcategories:
        try:
            loaded = json.loads(subcategories)
            if isinstance(loaded, dict):
                taxonomy = {k: list(v) for k, v in loaded.items() if isinstance(v, list)}
        except (ValueError, TypeError):
            taxonomy = {}

    titles = [it.title for it in parsed_items]
    cats = await categorize_grocery(titles, taxonomy) if taxonomy else []

    items: list[VoiceGroceryItem] = []
    for i, it in enumerate(parsed_items):
        c = cats[i] if i < len(cats) and isinstance(cats[i], dict) else {}
        shop = c.get("shopping_category")
        if shop not in ("grocery", "home", "health"):
            shop = it.shopping_category
        sub = c.get("subcategory")
        # Trust only sub-categories that exist in the family's taxonomy for that
        # category; otherwise fall back to "אחר" if present, else leave unset.
        allowed = taxonomy.get(shop, [])
        if sub not in allowed:
            sub = "אחר" if "אחר" in allowed else None
        items.append(
            VoiceGroceryItem(
                title=it.title, qty=it.qty, shopping_category=shop, subcategory=sub
            )
        )

    logger.info("voice_grocery: transcript=%r -> %d item(s)", transcript, len(items))
    return VoiceGroceryResponse(transcript=transcript, items=items)
