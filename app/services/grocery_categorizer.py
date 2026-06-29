"""
grocery_categorizer.py — assign a shopping category + sub-category to grocery
items via the LLM, against a family-provided taxonomy.

Used by /voice/grocery so spoken items land in the right sub-category instead
of collapsing into "Other". One call handles the whole batch.

The taxonomy maps each main category ("grocery"/"home"/"health") to that
family's list of Hebrew sub-category names; the model must pick a sub-category
from the relevant list (verbatim) so the result maps cleanly back in the app.
"""

from __future__ import annotations

import json
import logging

from app.core.config import get_settings
from app.services.intent_parser import _get_client

logger = logging.getLogger(__name__)

_SYSTEM = (
    "אתה מסווג פריטי קניות בעברית. תקבל רשימת פריטים וטקסונומיה של קטגוריות "
    "ראשיות (grocery=מכולת/אוכל, home=מוצרים לבית/ניקיון/נייר, "
    "health=פארם/בריאות/טיפוח), כשלכל קטגוריה רשימת תת-קטגוריות. "
    "לכל פריט בחר בדיוק קטגוריה ראשית אחת (אחד המפתחות grocery/home/health) "
    "ותת-קטגוריה אחת בלבד — מילה במילה מתוך רשימת התת-קטגוריות של אותה קטגוריה. "
    "אם אין התאמה טובה בחר את תת-הקטגוריה 'אחר' אם היא קיימת באותה קטגוריה. "
    'החזר JSON בלבד בצורה {"items":[{"title":"...","shopping_category":"...","subcategory":"..."}]} '
    "באותו סדר ובאותו מספר פריטים שהתקבלו."
)


async def categorize_grocery(
    titles: list[str],
    taxonomy: dict[str, list[str]],
) -> list[dict]:
    """Return [{title, shopping_category, subcategory}] aligned to `titles`.

    `taxonomy` maps "grocery"/"home"/"health" → list of Hebrew sub-category
    names. Returns [] on any failure — the caller falls back gracefully.
    """
    if not titles or not taxonomy:
        return []

    client = _get_client()
    settings = get_settings()
    payload = json.dumps({"items": titles, "taxonomy": taxonomy}, ensure_ascii=False)

    try:
        resp = await client.chat.completions.create(
            model=settings.openai_model,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": payload},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        items = data.get("items", [])
        return items if isinstance(items, list) else []
    except Exception:  # noqa: BLE001 — categorization is best-effort
        logger.exception("categorize_grocery failed")
        return []
