"""
The "family brain" — free-form Q&A over the whole family.

When the intent parser classifies a message as `general_query` (a question
about the family that no specific intent covers — a summary, a cross-topic
question, recalling something written in a note), we fetch a single
name-resolved snapshot of the entire family from family-os and let a stronger
model answer in free Hebrew text.

At family scale the full dataset fits in one context window, so this is
context-stuffing, not retrieval — no vector store. The brain is strictly
READ-ONLY: it answers, it never creates or mutates data (writes always go
through the structured intents).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.services.family_os_client import family_os_client

log = logging.getLogger(__name__)
TZ = ZoneInfo("Asia/Jerusalem")

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=get_settings().openai_api_key)
    return _client


SYSTEM_PROMPT = """\
אתה העוזר האישי החכם של המשפחה בתוך אפליקציית Family OS.

למטה מצורף תקציר מלא ועדכני של כל מה שהמשפחה עוקבת אחריו: בני המשפחה והילדים,
פתקים, פרוייקטים, אירועים (חד-פעמיים וחוזרים), משימות פתוחות, רשימת קניות,
תשלומים פתוחים, לוח החוגים של הילדים, וסיכום הוצאות.

ענה על שאלת המשתמש בעברית — קצר, חם וברור. כללים:
- הסתמך אך ורק על המידע המצורף. מותר ורצוי לחשב, להשוות, לסנן ולהסיק ממנו
  (למשל "מי הכי עמוס השבוע", "כמה נשאר לשלם", "מה מתוכנן מחר").
- אם התשובה אינה במידע — אמור בכנות שאין לך מידע על כך. אל תמציא פרטים.
- דבר בשמות בני המשפחה; אל תחשוף מזהים טכניים.
- אל תציע לבצע פעולות (להוסיף/למחוק/לעדכן) — רק ענה על השאלה.
- אל תמציא אירועים או תאריכים. אם המשתמש שואל על תאריך יחסי ("היום", "השבוע",
  "מחר") — חשב אותו לפי CURRENT_DATE שמופיע במידע.
"""

_ERR_REPLY = "⚠️ לא הצלחתי לעבד את השאלה כרגע. נסו שוב בעוד רגע."


async def answer_family_question(family_id: str, question: str) -> str:
    """Fetch the family snapshot and answer `question` in free Hebrew text."""
    s = get_settings()
    if not s.openai_api_key:
        return _ERR_REPLY

    try:
        snapshot = await family_os_client.get_snapshot(family_id)
    except (httpx.HTTPError, httpx.HTTPStatusError) as exc:
        log.warning("brain: snapshot fetch failed: %s", exc)
        return _ERR_REPLY

    now = datetime.now(timezone.utc).astimezone(TZ)
    family_json = json.dumps(snapshot, ensure_ascii=False)
    system = (
        f"{SYSTEM_PROMPT}\n"
        f"CURRENT_DATE: {now.strftime('%Y-%m-%d')} (Asia/Jerusalem)\n\n"
        f"FAMILY_DATA (JSON):\n{family_json}"
    )

    try:
        resp = await _get_client().chat.completions.create(
            model=s.openai_brain_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question.strip()},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("brain: LLM call failed: %s", exc)
        return _ERR_REPLY

    if not answer:
        return "🤔 לא מצאתי תשובה לכך במידע של המשפחה."

    # Observability for a new LLM feature (solo-dev / test-data family).
    log.info("brain Q=%r A=%r", question[:120], answer[:200])
    return answer
