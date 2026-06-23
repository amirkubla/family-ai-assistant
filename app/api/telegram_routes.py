"""
Telegram bot HTTP endpoints.

  POST /telegram/generate-code   ← called by the family-os web/native app
                                   from the Settings "Connect Telegram" CTA.
  POST /telegram/webhook         ← called by Telegram on every update.

Note: this router is mounted at the ROOT, NOT under /api. The family-os
frontend has the URL hardcoded as `${ASSISTANT_URL}/telegram/generate-code`
in src/lib/api/endpoints.ts:211.

Auth model:
  - generate-code: trusts the client-supplied family_id today. This is the
    same trust level the family-os auth helpers extend to localStorage.
    Tightening to a JWT exchange is a follow-up.
  - webhook: anyone-can-call by design — Telegram does. To distinguish real
    Telegram traffic, we check the `X-Telegram-Bot-Api-Secret-Token` header
    against TELEGRAM_BOT_TOKEN's last 16 chars (set as secret on
    setWebhook). Lightweight defense — Telegram's official recommendation.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.services import telegram_service
from app.services.family_os_client import family_os_client
from app.services.intent_parser import (
    ChoreIntent,
    FamilyEventIntent,
    GroceryIntent,
    NoteIntent,
    QueryChoresIntent,
    QueryEventsIntent,
    QueryGroceryIntent,
    UnsupportedIntent,
    parse_intent,
)
from app.services.telegram_client import send_message

log = logging.getLogger(__name__)

router = APIRouter(prefix="/telegram", tags=["telegram"])


# ────────────────────────────────────────────────────────────────────────
# generate-code  (called by the family-os frontend)
# ────────────────────────────────────────────────────────────────────────

class GenerateCodeRequest(BaseModel):
    # The frontend sends `family_id` (snake_case) — keep this name.
    family_id: str = Field(..., min_length=8, max_length=64)


class GenerateCodeResponse(BaseModel):
    code: str
    expires_in_minutes: int


@router.post("/generate-code", response_model=GenerateCodeResponse)
async def generate_code(
    body: GenerateCodeRequest,
    db: AsyncSession = Depends(get_db),
) -> GenerateCodeResponse:
    """Mint a one-time code for the user to redeem in Telegram."""
    s = get_settings()
    if not s.telegram_bot_token or not s.openai_api_key:
        # The bot itself wouldn't be able to handle the redemption — fail
        # fast rather than handing out codes that can never be used.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telegram integration is not configured",
        )
    code, ttl = await telegram_service.generate_code(db, body.family_id)
    return GenerateCodeResponse(code=code, expires_in_minutes=ttl)


# ────────────────────────────────────────────────────────────────────────
# webhook  (called by Telegram)
# ────────────────────────────────────────────────────────────────────────


def _bot_name() -> str:
    return "family_os_assistant_bot"


def _format_event_reply(intent: FamilyEventIntent) -> str:
    def hhmm(m: int) -> str:
        return f"{m // 60:02d}:{m % 60:02d}"

    when = f"{intent.date} בשעה {hhmm(intent.start_minutes)}-{hhmm(intent.end_minutes)}"
    base = f"✅ נוצר אירוע: {intent.title}\n📅 {when}"
    if intent.location:
        base += f"\n📍 {intent.location}"
    return base


_CATEGORY_EMOJI = {"grocery": "🛒", "home": "🏠", "health": "💊"}
_CATEGORY_HE = {"grocery": "מכולת", "home": "לבית", "health": "פארם"}


def _format_chore_reply(intent: ChoreIntent) -> str:
    base = f"✅ נוסף למשימות: {intent.title}"
    if intent.assigned_to:
        base += f"\n👤 {intent.assigned_to}"
    return base


def _format_note_reply(intent: NoteIntent) -> str:
    if intent.title:
        return f"📝 נשמרה תזכורת: {intent.title}\n{intent.body}"
    # Truncate long bodies in the reply so the chat doesn't get spammed —
    # the full body is saved server-side regardless.
    preview = intent.body if len(intent.body) <= 80 else intent.body[:77] + "..."
    return f"📝 נשמרה תזכורת: {preview}"


_RANGE_HE = {"today": "להיום", "tomorrow": "למחר", "week": "לשבוע הקרוב"}

# Sun=0 … Sat=6, matching family_events / schedule_blocks daysOfWeek.
_HE_DAYS = ["יום א׳", "יום ב׳", "יום ג׳", "יום ד׳", "יום ה׳", "יום ו׳", "שבת"]


def _hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def _day_prefix(item: dict[str, Any]) -> str:
    """
    A leading day label for week-view lines. One-time items carry a `date`
    ("YYYY-MM-DD"); recurring items carry `daysOfWeek` (Sun=0..Sat=6). Without
    this, recurring weekly activities render with only a time ("17:00-18:00"),
    which is useless in a week view — you can't tell which day they fall on.
    """
    date = item.get("date")
    if date:
        return f"{date} · "
    days = item.get("daysOfWeek")
    if isinstance(days, list) and days:
        labels = [
            _HE_DAYS[d]
            for d in sorted(set(days))
            if isinstance(d, int) and 0 <= d <= 6
        ]
        if labels:
            return f"{'/'.join(labels)} · "
    return ""


def _format_event_line(
    item: dict[str, Any],
    *,
    include_date: bool,
    name_by_id: dict[str, str] | None = None,
) -> str:
    """Render a single family-event or schedule-block row as a bullet line."""
    title = item.get("title", "(ללא כותרת)")
    start = item.get("startMinutes")
    end = item.get("endMinutes")
    # Only prefix the day in week view — in today/tomorrow views the day is
    # implied by the question.
    prefix = _day_prefix(item) if include_date else ""
    time_str = ""
    if isinstance(start, int) and isinstance(end, int):
        time_str = f" ({_hhmm(start)}-{_hhmm(end)})"
    location = item.get("location")
    loc_str = f" 📍{location}" if location else ""

    # Assignee badge — omit for family-wide events (the default).
    assignee_str = ""
    assignee_type = item.get("assigneeType")
    assignee_id = item.get("assigneeId")
    if assignee_type in ("kid", "member") and assignee_id and name_by_id:
        label = name_by_id.get(assignee_id)
        if label:
            assignee_str = f" [{label}]"

    return f"• {prefix}{title}{time_str}{loc_str}{assignee_str}"


def _format_events_query_reply(
    range_: str,
    events: list[dict[str, Any]],
    *,
    kid_name: str | None = None,
    schedule_blocks: list[dict[str, Any]] | None = None,
    name_by_id: dict[str, str] | None = None,
) -> str:
    """
    Render the events query reply. When `kid_name` is set and
    `schedule_blocks` is provided, the reply has two sections (events vs.
    schedule) so the user can tell family appointments apart from the kid's
    weekly classes/hobbies.

    `name_by_id`: maps assigneeId (UUID) → display label used as a badge on
    each event line (e.g. "🎮 שלומי" or "👩 מזל"). Family-wide events
    (assigneeType="family") get no badge.
    """
    label = _RANGE_HE.get(range_, "להיום")
    include_date = range_ == "week"

    def fmt(e: dict[str, Any]) -> str:
        return _format_event_line(e, include_date=include_date, name_by_id=name_by_id)

    if kid_name and schedule_blocks is not None:
        # Kid-scoped reply: two clearly labeled sections.
        if not events and not schedule_blocks:
            return f"📅 אין שום דבר על הלו\"ז של {kid_name} {label}."
        out: list[str] = []
        if events:
            out.append(f"📅 אירועים של {kid_name} {label}:")
            out.extend(fmt(e) for e in events)
        if schedule_blocks:
            if out:
                out.append("")  # blank line between sections
            out.append(f"📚 לוח שבועי של {kid_name}:")
            out.extend(fmt(b) for b in schedule_blocks)
        return "\n".join(out)

    # Family-wide (no kid scope): show assignee badge on each line so the
    # user can tell at a glance which events belong to which kid or parent.
    if not events:
        return f"📅 אין אירועים {label}."
    lines = [f"📅 אירועים {label}:"]
    lines.extend(fmt(e) for e in events)
    return "\n".join(lines)


def _format_grocery_query_reply(items: list[dict[str, Any]]) -> str:
    if not items:
        return "🛒 רשימת הקניות ריקה."
    lines = ["🛒 ברשימת הקניות:"]
    for it in items:
        emoji = _CATEGORY_EMOJI.get(it.get("shoppingCategory") or "grocery", "🛒")
        title = it.get("title", "(ללא שם)")
        qty = it.get("qty")
        qty_str = f" ({qty})" if qty else ""
        lines.append(f"{emoji} {title}{qty_str}")
    return "\n".join(lines)


def _format_chores_query_reply(
    chores: list[dict[str, Any]], *, scoped: bool = False
) -> str:
    if not chores:
        return "✅ אין משימות פתוחות שלך." if scoped else "✅ אין משימות פתוחות."
    header = "📋 המשימות שלך:" if scoped else "📋 משימות פתוחות:"
    lines = [header]
    for c in chores:
        title = c.get("title", "(ללא כותרת)")
        assignee = c.get("assignedTo")
        # Don't repeat the assignee when the list is already scoped to one
        # person — they know.
        suffix = "" if scoped else (f" — {assignee}" if assignee else "")
        lines.append(f"• {title}{suffix}")
    return "\n".join(lines)


def _format_grocery_reply(intent: GroceryIntent) -> str:
    if len(intent.items) == 1:
        it = intent.items[0]
        emoji = _CATEGORY_EMOJI.get(it.shopping_category, "🛒")
        shelf = _CATEGORY_HE.get(it.shopping_category, "מכולת")
        qty = f" ({it.qty})" if it.qty else ""
        return f"{emoji} נוסף ל{shelf}: {it.title}{qty}"
    # Multiple items — render a bulleted list with per-item category emoji.
    lines = ["✅ נוסף לרשימת קניות:"]
    for it in intent.items:
        emoji = _CATEGORY_EMOJI.get(it.shopping_category, "🛒")
        qty = f" ({it.qty})" if it.qty else ""
        lines.append(f"{emoji} {it.title}{qty}")
    return "\n".join(lines)


async def _handle_text_message(
    db: AsyncSession, chat_id: int, text: str
) -> str:
    """
    Dispatch a free-text message to the right family-os endpoint and return
    the reply text to send back to the user.
    """
    family_id, family_member_id = await telegram_service.get_binding_for_chat(
        db, chat_id
    )
    if not family_id:
        return (
            "אנא חברו את החשבון מתוך האפליקציה תחילה: "
            "הגדרות → חבר טלגרם, ואז שלחו לי את הקוד עם /start"
        )

    parsed = await parse_intent(text)

    if isinstance(parsed, UnsupportedIntent):
        return f"מצטער, {parsed.reason}"

    try:
        if isinstance(parsed, FamilyEventIntent):
            await family_os_client.create_family_event(
                family_id,
                title=parsed.title,
                start_minutes=parsed.start_minutes,
                end_minutes=parsed.end_minutes,
                is_recurring=False,
                date=parsed.date,
                location=parsed.location,
            )
            return _format_event_reply(parsed)

        if isinstance(parsed, GroceryIntent):
            # Create one row per item. We don't bail on a partial failure —
            # the items that succeeded are kept; failures fall through to
            # the outer except block with a generic error message.
            for it in parsed.items:
                await family_os_client.create_grocery_item(
                    family_id,
                    title=it.title,
                    qty=it.qty,
                    shopping_category=it.shopping_category,
                )
            return _format_grocery_reply(parsed)

        if isinstance(parsed, ChoreIntent):
            await family_os_client.create_chore(
                family_id,
                title=parsed.title,
                assigned_to=parsed.assigned_to,
            )
            return _format_chore_reply(parsed)

        if isinstance(parsed, NoteIntent):
            await family_os_client.create_note(
                family_id,
                body=parsed.body,
                title=parsed.title,
            )
            return _format_note_reply(parsed)

        if isinstance(parsed, QueryEventsIntent):
            # Fetch events, kids, and members in parallel so each event line
            # can be labelled with the assignee's name and emoji.
            fetch_schedule = parsed.kid_name is not None
            coros = [
                family_os_client.list_family_events(
                    family_id, range_=parsed.range, kid_name=parsed.kid_name
                ),
                family_os_client.list_kids(family_id),
                family_os_client.list_members(family_id),
            ]
            if fetch_schedule:
                coros.append(
                    family_os_client.list_schedule_blocks(
                        family_id, range_=parsed.range, kid_name=parsed.kid_name
                    )
                )
            results = await asyncio.gather(*coros, return_exceptions=True)

            events = results[0] if not isinstance(results[0], Exception) else []
            kids_list = results[1] if not isinstance(results[1], Exception) else []
            members_list = results[2] if not isinstance(results[2], Exception) else []
            schedule_blocks: list[dict[str, Any]] | None = None
            if fetch_schedule:
                schedule_blocks = results[3] if not isinstance(results[3], Exception) else []

            # Build id → "emoji name" map for both kids and members.
            name_by_id: dict[str, str] = {}
            for k in kids_list:
                emoji = k.get("emoji") or "👦"
                name_by_id[k["id"]] = f"{emoji} {k['name']}"
            for m in members_list:
                emoji = m.get("avatarEmoji") or "👤"
                name_by_id[m["id"]] = f"{emoji} {m['displayName']}"

            return _format_events_query_reply(
                parsed.range,
                events,
                kid_name=parsed.kid_name,
                schedule_blocks=schedule_blocks,
                name_by_id=name_by_id,
            )

        if isinstance(parsed, QueryGroceryIntent):
            items = await family_os_client.list_grocery(family_id)
            return _format_grocery_query_reply(items)

        if isinstance(parsed, QueryChoresIntent):
            # Resolve "my" → the chat's bound member. If the user asked
            # personal but never set /me, fall back to the family-wide view
            # with a hint so the bot is useful, not blocking.
            assignee_filter: str | None = None
            mine_hint = ""
            if parsed.mine:
                if family_member_id:
                    assignee_filter = family_member_id
                else:
                    mine_hint = (
                        "\n\n💡 כדי לסנן רק למשימות שלך, בחר מי אתה עם הפקודה /me"
                    )
            chores = await family_os_client.list_chores(
                family_id,
                assignee_member_id=assignee_filter,
                selected_for_today=parsed.today or None,
            )
            return _format_chores_query_reply(chores, scoped=parsed.mine) + mine_hint
    except httpx.HTTPStatusError as exc:
        log.warning("family-os API %s: %s", exc.response.status_code, exc.response.text[:200])
        return (
            f"⚠️ שגיאת שרת ({exc.response.status_code}). "
            f"נסו לנסח אחרת או פנו לתמיכה."
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("unexpected error handling message")
        return f"⚠️ שגיאה לא צפויה: {exc}"

    return "לא הצלחתי להבין. אפשר לנסות לנסח אחרת?"


@router.post("/webhook")
async def telegram_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Telegram POSTs every chat update here. Always return 200 — Telegram
    retries on non-2xx, which double-sends messages to the user.

    Shape: https://core.telegram.org/bots/api#update
    """
    s = get_settings()

    # Lightweight authenticity check (Telegram-recommended pattern).
    expected = s.telegram_bot_token[-16:] if s.telegram_bot_token else ""
    if expected:
        provided = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if provided != expected:
            log.warning("webhook: bad secret token, ignoring")
            return {"ok": True}

    try:
        update = await request.json()
    except Exception:
        return {"ok": True}

    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True}

    chat_id = message.get("chat", {}).get("id")
    text = (message.get("text") or "").strip()
    if not chat_id or not text:
        return {"ok": True}

    # /start <code> — redeem the one-time code and bind the chat.
    if text.startswith("/start"):
        parts = text.split(maxsplit=1)
        code = parts[1].strip() if len(parts) > 1 else ""
        if not code:
            await send_message(
                chat_id,
                "ברוכים הבאים ל-Family OS!\n\n"
                "כדי לחבר את החשבון, פתחו את האפליקציה → הגדרות → חבר טלגרם, "
                "ושלחו לי את הקוד עם הפקודה: /start <CODE>",
            )
            return {"ok": True}

        family_id = await telegram_service.redeem_code(db, code, chat_id)
        if family_id is None:
            await send_message(
                chat_id,
                "❌ קוד לא תקף או פג תוקפו. אנא חזרו לאפליקציה והפיקו קוד חדש.",
            )
        else:
            await send_message(
                chat_id,
                "✅ חיברתי! עכשיו אפשר לשלוח לי בקשות בעברית, למשל:\n"
                "• \"תקבע מסיבת תה ב-15 לאפריל ב-14:00\"\n"
                "• \"תוסיף חלב וביצים לקניות\"\n"
                "• \"תזכיר לעודד להוציא את הזבל\"\n"
                "• \"תרשום פתק שהמפתחות אצל השכן\"\n"
                "• \"מה יש לי היום?\" / \"מה ברשימת הקניות?\"\n"
                "\n"
                "💡 כדאי לרשום /me כדי לבחור מי אתה במשפחה — "
                "אחר כך אדע לסנן שאלות כמו \"המשימות שלי\".",
            )
        return {"ok": True}

    if text.startswith("/help"):
        await send_message(
            chat_id,
            "אני העוזר של משפחת Family OS 🏠\n"
            "אפשר לבקש ממני:\n"
            "• לתזמן אירוע: \"תקבע פגישה ביום ראשון ב-10\"\n"
            "• להוסיף לקניות: \"תוסיף לחם וחלב לרשימה\"\n"
            "• להוסיף משימה: \"תזכיר לעודד להוציא את הזבל\"\n"
            "• לרשום פתק: \"תרשום שהמפתחות אצל השכן\"\n"
            "• לשאול: \"מה יש לי היום?\" / \"מה ברשימת הקניות?\" / \"מה המשימות?\"\n"
            "\n"
            "פקודות:\n"
            "• /me        — בחר מי אתה במשפחה (בשביל שאלות אישיות כמו \"המשימות שלי\")",
        )
        return {"ok": True}

    # /me — bind this chat to a specific family member. Without args, list
    # the available members with numbers. With a number, bind.
    if text.startswith("/me"):
        family_id_bound = await telegram_service.get_family_for_chat(db, chat_id)
        if not family_id_bound:
            await send_message(
                chat_id,
                "אנא חברו את החשבון מתוך האפליקציה תחילה (הגדרות → חבר טלגרם), "
                "ואז שלחו /start <CODE>. אחר כך נוכל לבחור מי אתה.",
            )
            return {"ok": True}

        # Fetch members once and store via lookup-by-number in the reply.
        # The user picks a number from THAT list — no DB-side state needed.
        try:
            members = await family_os_client.list_members(family_id_bound)
        except httpx.HTTPStatusError as exc:
            log.warning(
                "/me list_members %s: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            await send_message(chat_id, "⚠️ שגיאה בטעינת רשימת המשפחה. נסו שוב בעוד רגע.")
            return {"ok": True}

        if not members:
            await send_message(
                chat_id,
                "לא נמצאו בני משפחה. נסו להוסיף אותם דרך האפליקציה תחילה.",
            )
            return {"ok": True}

        parts = text.split(maxsplit=1)
        arg = parts[1].strip() if len(parts) > 1 else ""

        if not arg:
            # No arg: print the numbered list.
            lines = ["מי אתה? ענה עם המספר, למשל: /me 2", ""]
            for i, m in enumerate(members, start=1):
                emoji = m.get("avatarEmoji") or "👤"
                role = m.get("role")
                role_str = f" ({role})" if role else ""
                lines.append(f"{i}. {emoji} {m['displayName']}{role_str}")
            await send_message(chat_id, "\n".join(lines))
            return {"ok": True}

        # Arg given: must be a number in range.
        try:
            n = int(arg)
        except ValueError:
            await send_message(
                chat_id, "אנא ענה במספר בלבד, למשל: /me 2"
            )
            return {"ok": True}
        if not (1 <= n <= len(members)):
            await send_message(
                chat_id, f"מספר לא תקף. הקלד /me כדי לראות את הרשימה (1-{len(members)})."
            )
            return {"ok": True}

        chosen = members[n - 1]
        ok = await telegram_service.set_member_for_chat(db, chat_id, chosen["id"])
        if not ok:
            await send_message(
                chat_id, "⚠️ לא הצלחתי לשמור את הבחירה. נסו שוב."
            )
            return {"ok": True}
        emoji = chosen.get("avatarEmoji") or "👤"
        await send_message(
            chat_id,
            f"✅ נשמר. אני יודע שאתה {emoji} {chosen['displayName']}.\n"
            f"עכשיו אפשר לשאול \"מה המשימות שלי?\" ואני אדע מי לחפש.",
        )
        return {"ok": True}

    # Free-form message → LLM intent → family-os.
    reply = await _handle_text_message(db, chat_id, text)
    await send_message(chat_id, reply)
    return {"ok": True}


# ────────────────────────────────────────────────────────────────────────
# admin: register / re-register the Telegram webhook
# ────────────────────────────────────────────────────────────────────────


class SetWebhookRequest(BaseModel):
    webhook_url: str


@router.post("/admin/set-webhook")
async def admin_set_webhook(
    body: SetWebhookRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Manually trigger setWebhook on Telegram. Call this once after deploy
    (or whenever the Cloud Run URL changes). Auth: Bearer FAMILY_OS_SERVICE_TOKEN.

    Curl:
      curl -X POST https://<assistant>/telegram/admin/set-webhook \\
        -H "Authorization: Bearer $SERVICE_TOKEN" \\
        -H "Content-Type: application/json" \\
        -d '{"webhook_url":"https://<assistant>/telegram/webhook"}'
    """
    s = get_settings()
    if not s.family_os_service_token:
        raise HTTPException(status_code=503, detail="service token not configured")
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {s.family_os_service_token}":
        raise HTTPException(status_code=401, detail="bad service token")

    from app.services.telegram_client import set_webhook

    res = await set_webhook(body.webhook_url)
    if res is None:
        raise HTTPException(status_code=500, detail="setWebhook failed")
    return res
