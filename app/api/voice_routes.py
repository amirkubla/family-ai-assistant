"""
voice_routes.py — speech → structured items for the family-os app.

  POST /voice/grocery   ← multipart audio → parsed + categorized grocery items.
  POST /voice/note      ← multipart audio → {transcript, title, body} for a note.

Transcribes Hebrew speech with gpt-4o-transcribe and RETURNS structured data
WITHOUT writing anything — the family-os app reviews it and writes via its own
optimistic CRUD, so the user can edit/remove before it lands.

Mounted at ROOT (no /api prefix) — the family-os frontend calls
${ASSISTANT_URL}/voice/grocery, the same convention as /telegram/*.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import get_settings
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

# Concise Hebrew note title from free-form content.
_NOTE_TITLE_SYSTEM = (
    "אתה יוצר כותרת קצרה וברורה בעברית לפתק על סמך תוכנו — עד 5 מילים, "
    'ללא מירכאות וללא נקודה בסוף. החזר JSON בלבד: {"title": "..."}'
)

# Split free-form Hebrew speech into discrete to-do tasks.
_CHORE_SYSTEM = (
    "אתה מחלץ רשימת מטלות (to-do) מהקלטה בעברית. פצל את הדברים שנאמרו למטלות "
    "נפרדות, כל אחת כניסוח קצר וברור (2–6 מילים). התעלם ממילות קישור. "
    'החזר JSON בלבד: {"tasks": ["...", "..."]}'
)

# Concise Hebrew project name from a free-form description.
_PROJECT_TITLE_SYSTEM = (
    "אתה יוצר שם קצר וברור בעברית לפרויקט על סמך תיאורו — עד 5 מילים, "
    'ללא מירכאות וללא נקודה בסוף. החזר JSON בלבד: {"title": "..."}'
)


async def _transcribe(audio: UploadFile, prompt: str | None = None) -> str:
    """Transcribe a Hebrew clip to text. 400 on empty audio, 502 on failure."""
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio")
    kwargs: dict = {
        "model": _TRANSCRIBE_MODEL,
        "file": (audio.filename or "voice.m4a", data),
        "language": "he",
    }
    if prompt:
        kwargs["prompt"] = prompt
    try:
        result = await _get_client().audio.transcriptions.create(**kwargs)
    except Exception:  # noqa: BLE001 — surface a clean 502 to the caller
        logger.exception("transcription failed")
        raise HTTPException(status_code=502, detail="transcription failed")
    return (getattr(result, "text", "") or "").strip()


async def _generate_title(transcript: str, system: str) -> str:
    """LLM-generated short Hebrew title for the given system prompt. "" on error."""
    try:
        resp = await _get_client().chat.completions.create(
            model=get_settings().openai_model,
            response_format={"type": "json_object"},
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": transcript},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return (data.get("title") or "").strip()[:80]
    except Exception:  # noqa: BLE001 — title is optional, never fail the request
        logger.exception("title generation failed")
        return ""


async def _extract_tasks(transcript: str) -> list[str]:
    """Split a Hebrew transcript into discrete to-do titles. [] on failure."""
    try:
        resp = await _get_client().chat.completions.create(
            model=get_settings().openai_model,
            response_format={"type": "json_object"},
            temperature=0.1,
            messages=[
                {"role": "system", "content": _CHORE_SYSTEM},
                {"role": "user", "content": transcript},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return [str(x).strip() for x in data.get("tasks", []) if str(x).strip()]
    except Exception:  # noqa: BLE001 — never fail the request on a parse error
        logger.exception("chore extraction failed")
        return []


class VoiceGroceryItem(BaseModel):
    title: str
    qty: str | None = None
    shopping_category: str = "grocery"
    subcategory: str | None = None


class VoiceGroceryResponse(BaseModel):
    transcript: str
    items: list[VoiceGroceryItem]


class VoiceNoteResponse(BaseModel):
    transcript: str
    title: str
    body: str


class VoiceChoreItem(BaseModel):
    title: str


class VoiceChoreResponse(BaseModel):
    transcript: str
    items: list[VoiceChoreItem]


class VoiceProjectResponse(BaseModel):
    transcript: str
    title: str
    description: str


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
    transcript = await _transcribe(audio, _TRANSCRIBE_PROMPT)
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


@router.post("/note", response_model=VoiceNoteResponse)
async def voice_note(audio: UploadFile = File(...)) -> VoiceNoteResponse:
    """Transcribe a free-form Hebrew clip + generate a title (no write).

    Body is the verbatim transcript; the title is LLM-generated. The family-os
    app opens its note editor pre-filled for review before saving.
    """
    transcript = await _transcribe(audio)
    if not transcript:
        return VoiceNoteResponse(transcript="", title="", body="")
    title = await _generate_title(transcript, _NOTE_TITLE_SYSTEM)
    logger.info("voice_note: transcript=%r title=%r", transcript, title)
    return VoiceNoteResponse(transcript=transcript, title=title, body=transcript)


@router.post("/chore", response_model=VoiceChoreResponse)
async def voice_chore(audio: UploadFile = File(...)) -> VoiceChoreResponse:
    """Transcribe a Hebrew clip → a list of to-do tasks (no write).

    The family-os app reviews the tasks and adds them via its own chore CRUD.
    """
    transcript = await _transcribe(audio)
    if not transcript:
        return VoiceChoreResponse(transcript="", items=[])
    items = [VoiceChoreItem(title=x) for x in await _extract_tasks(transcript)]
    logger.info("voice_chore: transcript=%r -> %d task(s)", transcript, len(items))
    return VoiceChoreResponse(transcript=transcript, items=items)


@router.post("/project", response_model=VoiceProjectResponse)
async def voice_project(audio: UploadFile = File(...)) -> VoiceProjectResponse:
    """Transcribe a Hebrew clip → a project draft (name + description; no write).

    Description is the verbatim transcript; the name is LLM-generated. The
    family-os app opens its project editor pre-filled for review before saving.
    """
    transcript = await _transcribe(audio)
    if not transcript:
        return VoiceProjectResponse(transcript="", title="", description="")
    title = await _generate_title(transcript, _PROJECT_TITLE_SYSTEM)
    logger.info("voice_project: transcript=%r title=%r", transcript, title)
    return VoiceProjectResponse(transcript=transcript, title=title, description=transcript)
