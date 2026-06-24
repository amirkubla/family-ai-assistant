"""
Hebrew natural-language → structured intent, via OpenAI.

The Telegram bot accepts free-form Hebrew text like:
  "תקבע לנו מסיבת תה ב-15 לאפריל ב-14:00"
  "תוסיף חלב לקניות"
  "תזכיר לי לקנות סוללות בערב"

We ask the model to pick ONE intent and emit a strict JSON envelope. If the
text doesn't match any supported intent, the model returns
`{"intent": "unsupported", "reason": "..."}` and the bot replies politely.

We keep the schema as tight as possible so the bot can route + call the
family-os API without further parsing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.core.config import get_settings

TZ = ZoneInfo("Asia/Jerusalem")


class FamilyEventIntent(BaseModel):
    intent: Literal["family_event"] = "family_event"
    title: str
    date: str = Field(..., description='ISO YYYY-MM-DD in Asia/Jerusalem')
    start_minutes: int = Field(..., ge=0, le=1439)
    end_minutes: int = Field(..., ge=0, le=1440)
    location: str | None = None


class GroceryItem(BaseModel):
    title: str
    qty: str | None = None
    # family-os has three shopping-list "shelves": grocery (food),
    # home (cleaning/laundry/paper), health (pharmacy/hygiene). Defaults
    # to grocery — the most common case — but the LLM should override
    # based on Hebrew keywords (see system prompt).
    shopping_category: Literal["grocery", "home", "health"] = "grocery"


class GroceryIntent(BaseModel):
    intent: Literal["grocery"] = "grocery"
    # Always a list, even for a single item — the user can ask for many at
    # once ("תוסיף עגבניות וביצים"). The webhook handler creates one row
    # per item.
    items: list[GroceryItem] = Field(..., min_length=1)


class ChoreIntent(BaseModel):
    intent: Literal["chore"] = "chore"
    title: str
    # Free-text Hebrew name of the assignee, if mentioned. family-os tries
    # to resolve it to a known familyMember; on no match it's stored as
    # free text.
    assigned_to: str | None = None


class NoteIntent(BaseModel):
    intent: Literal["note"] = "note"
    # Required body — the main text of the note. Pinning is always false
    # at creation time; the user pins manually in the app.
    body: str
    # Optional short title — if the user explicitly named one ("פתק עם
    # הכותרת X"). Most casual notes won't have one.
    title: str | None = None


class QueryEventsIntent(BaseModel):
    intent: Literal["query_events"] = "query_events"
    range: Literal["today", "tomorrow", "week"] = "today"
    # Optional Hebrew kid name to scope to ("של דני", "לבן"). When set, the
    # bot fetches both kid-assigned family-events AND the kid's schedule_blocks
    # (classes/hobbies). Server-side name lookup → kid_id.
    kid_name: str | None = None


class QueryGroceryIntent(BaseModel):
    intent: Literal["query_grocery"] = "query_grocery"


class QueryChoresIntent(BaseModel):
    intent: Literal["query_chores"] = "query_chores"
    # True when the user said "שלי" / "אני" / similar — bot should filter to
    # the member bound to this chat (via /me). False = all family chores.
    mine: bool = False
    # True when the user said "היום" — bot should filter to selectedForToday=true.
    # "מחר" doesn't map cleanly to the chores schema (no due-date), so the
    # LLM treats "tomorrow" as "today=false" too (returns all undone) — see
    # the system prompt.
    today: bool = False


class ProjectIntent(BaseModel):
    intent: Literal["project"] = "project"
    title: str
    # Default to in_progress — if the user is telling the bot about a project
    # they want to start, it's more likely active than just an idea.
    status: Literal["idea", "in_progress"] = "in_progress"


class QueryNotesIntent(BaseModel):
    intent: Literal["query_notes"] = "query_notes"
    # pinned_only: true when the user specifically asks about pinned/important notes.
    pinned_only: bool = False


class QueryProjectsIntent(BaseModel):
    intent: Literal["query_projects"] = "query_projects"
    # include_done: true when the user explicitly asks about finished projects.
    include_done: bool = False


class UnsupportedIntent(BaseModel):
    intent: Literal["unsupported"] = "unsupported"
    reason: str


# Discriminated union — Pydantic picks the right one based on the `intent` field.
ParsedIntent = (
    FamilyEventIntent
    | GroceryIntent
    | ChoreIntent
    | NoteIntent
    | ProjectIntent
    | QueryEventsIntent
    | QueryGroceryIntent
    | QueryChoresIntent
    | QueryNotesIntent
    | QueryProjectsIntent
    | UnsupportedIntent
)


SYSTEM_PROMPT = """\
You are an extraction layer for a Hebrew-language family-coordination bot.

The user sends free-form Hebrew text. Your job: choose EXACTLY ONE intent
from the list below and emit JSON matching that intent's schema.

CRITICAL: every response MUST include a top-level "intent" field whose value
is EXACTLY one of:
  "family_event" | "grocery" | "chore" | "note" | "project" |
  "query_events" | "query_grocery" | "query_chores" |
  "query_notes" | "query_projects" | "unsupported"

If you omit the "intent" field the bot cannot route the message and the
user gets a generic error. Always include it, in every example, on every
intent, even when other fields make the intent "obvious" from context.

Intents:

1. "family_event" — the user wants to schedule a one-time event for the
   whole family (a meeting, a meal, an appointment, etc.).
   Fields:
     title          — short Hebrew title (4–40 chars).
     date           — "YYYY-MM-DD" in Asia/Jerusalem timezone.
                       Resolve relative dates ("מחר", "ביום שלישי הבא",
                       "15 לאפריל") against the CURRENT_DATE you'll be given.
                       If the year is unspecified and the date already
                       passed this year, use NEXT year.
     start_minutes  — minutes since midnight (0–1439).
                       If the user gives a time range, this is the start.
                       If only one time is given, set duration = 60 min.
                       If no time is given, default to 18:00 (1080) and
                       21:00 (1260).
     end_minutes    — minutes since midnight (1–1440). Must be > start_minutes.
     location       — optional, only if explicitly stated.
   Examples:
     "תקבע מסיבת תה ב-15 לאפריל ב-14:00" →
       {"intent":"family_event","title":"מסיבת תה","date":"2027-04-15","start_minutes":840,"end_minutes":900}
     "פגישה מחר ב-9 בבוקר במשרד" →
       {"intent":"family_event","title":"פגישה","date":"<tomorrow>","start_minutes":540,"end_minutes":600,"location":"המשרד"}

2. "grocery" — the user wants to add one or more items to the shopping list.
   Field:
     items — list of objects, one per item the user mentioned. EXTRACT
             EVERY ITEM, not just the first.
             Per-item fields:
                title — short Hebrew name of the item.
                qty   — optional, the quantity as text (e.g. "2", "ליטר",
                         "חבילה", "תריסר"), if the user gave one for
                         THAT specific item.
                shopping_category — ONE of "grocery", "home", "health".
                         Choose PER ITEM based on what it is:
                          - "grocery"  food, drinks, snacks. Examples:
                            חלב, לחם, ביצים, בננות, גבינה, יוגורט,
                            קוטג׳, חומוס, קוקה קולה, שוקולד, אורז, פסטה.
                          - "home"     household, cleaning, laundry, paper,
                            light/electric. Examples: סבון כלים, נייר טואלט,
                            סקוטש, מטליות, אקונומיקה, אבקת כביסה, מרכך,
                            שקיות זבל, נורה, סוללות.
                          - "health"   pharmacy, hygiene, medicine. Examples:
                            תרופות, ויטמינים, שמפו, מרכך שיער, משחת שיניים,
                            דאודורנט, פלסטרים, מסיכות, סבון רחצה.
                         When in doubt, pick "grocery". If the user
                         explicitly says "לרשימת ניקיון"/"לחומרי ניקוי"/
                         "לפארם"/"לרוקחות" — use that category for ALL
                         items in the message.
   Examples:
     "תוסיף עגבניות וביצים" →
       {"intent":"grocery","items":[{"title":"עגבניות","shopping_category":"grocery"},{"title":"ביצים","shopping_category":"grocery"}]}
     "תוסיף חלב וביצים לקניות" →
       {"intent":"grocery","items":[{"title":"חלב","shopping_category":"grocery"},{"title":"ביצים","shopping_category":"grocery"}]}

3. "chore" — the user wants to add a household to-do / chore (something
   one person needs to DO, with no specific time). Distinguish from
   "family_event" by the absence of a clock time and the imperative,
   action-on-a-person feel ("תזכיר ל…", "X צריך…", "תוסיף משימה…").
   Fields:
     title        — short Hebrew action phrase (4–60 chars), starting with
                    a verb when natural ("להוציא את הזבל", "לעבור על
                    חשבונות", "לקנות מתנה ליום הולדת").
     assigned_to  — optional Hebrew name of who should do it, if the user
                    named one ("עודד", "אמא", "הילדים"). Leave null if
                    unspecified or generic ("מישהו"/"כולם").
   Examples:
     "תזכיר לעודד להוציא את הזבל" →
       {"intent":"chore","title":"להוציא את הזבל","assigned_to":"עודד"}
     "אני צריך לעבור על החשבונות" →
       {"intent":"chore","title":"לעבור על החשבונות"}
     "תוסיף משימה לקנות מתנה לסבתא" →
       {"intent":"chore","title":"לקנות מתנה לסבתא"}

4. "note" — the user wants to save a free-form note / reminder / piece of
   info for the family. Distinguish from "chore" by the absence of an
   action verb directed at a person — notes are pieces of INFORMATION to
   remember, not things TO DO. Triggers: "תרשום פתק…", "תוסיף לפתקים…",
   "תזכור ש…", "תעלה לי במחברת…", "שמור לי ש…".
   Fields:
     body  — the main text of the note. Use what the user actually wants
             to remember, not the framing ("תרשום ש-X" → body="X").
     title — optional short title, ONLY if the user explicitly named one
             ("פתק עם הכותרת X", "תוסיף פתק שכותרתו…"). Otherwise null.
   Examples:
     "תרשום פתק שהמפתחות אצל השכן" →
       {"intent":"note","body":"המפתחות אצל השכן"}
     "תזכור שיש לנו את הוואי-פיי חדש: SSID FamilyOS, סיסמה 12345" →
       {"intent":"note","body":"וואי-פיי חדש: SSID FamilyOS, סיסמה 12345"}
     "תעלה לי במחברת את מספר השרברב 050-1234567" →
       {"intent":"note","body":"מספר השרברב 050-1234567"}

5. "query_events" — the user is ASKING what's scheduled (not creating
   anything). Triggers: ANY question word ("מה" / "איזה" / "איזו" /
   "אילו" / "כמה") combined with events/schedule words ("אירועים",
   "תוכניות", "לוח זמנים") and/or a timeframe ("היום", "מחר", "השבוע",
   "בשבוע הבא", a specific weekday).
   Fields:
     range    — ONE of "today" / "tomorrow" / "week".
                Map: "היום" → today; "מחר" → tomorrow; "השבוע" / "בימים
                הקרובים" / "בשבוע הבא" / a specific weekday → week. Default
                "today" if no timeframe.
     kid_name — Hebrew name of a kid, ONLY when the user named one
                ("לדני", "של נועה", "מה יש לבן השבוע"). Strip prefixes:
                "לדני" → "דני", "של נועה" → "נועה". Leave null if the user
                didn't name a kid.
   Examples:
     "מה יש לי היום?" →
       {"intent":"query_events","range":"today"}
     "מה יש מחר?" →
       {"intent":"query_events","range":"tomorrow"}
     "איזה ארועים יש לנו שבוע הבא ביום שני?" →
       {"intent":"query_events","range":"week"}
     "אילו אירועים יש לי השבוע?" →
       {"intent":"query_events","range":"week"}
     "מה יש לדני השבוע?" →
       {"intent":"query_events","range":"week","kid_name":"דני"}
     "איזה אירועים יש לנועה היום" →
       {"intent":"query_events","range":"today","kid_name":"נועה"}

6. "query_grocery" — the user is ASKING what's on the shopping list.
   Triggers: "מה ברשימת הקניות", "מה צריך לקנות", "מה יש בקניות",
   "מה חסר במכולת", "איזה קניות יש".
   No fields beyond the intent name.
   Examples:
     "מה ברשימת הקניות?" → {"intent":"query_grocery"}
     "מה צריך לקנות?"    → {"intent":"query_grocery"}

7. "query_chores" — the user is ASKING what tasks are open.
   Triggers: ANY question word ("מה" / "איזה" / "איזו" / "אילו" / "כמה")
   combined with chores/tasks words ("משימות", "מטלות", "מה צריך לעשות",
   "מה יש לי לעשות").
   Fields:
     mine  — true if the user phrased it as personal ("שלי" / "אני צריך/ה"
             / "לי לעשות"). Default false (all family chores).
     today — true if the user said "היום". Default false. ("מחר" maps to
             false too — chores have no due-date in the data model, so
             "tomorrow's tasks" is the same as "open tasks".)
   Examples:
     "מה המשימות שלי להיום?" →
       {"intent":"query_chores","mine":true,"today":true}
     "איזה משימות יש להיום?" →
       {"intent":"query_chores","mine":false,"today":true}
     "מה יש לי לעשות?" →
       {"intent":"query_chores","mine":true,"today":false}
     "מה המטלות הפתוחות?" →
       {"intent":"query_chores","mine":false,"today":false}
     "מה צריך לעשות היום בבית?" →
       {"intent":"query_chores","mine":false,"today":true}

8. "project" — the user wants to add a project (something bigger than a chore,
   with a title and optional status). Distinguish from "chore" by scope words
   ("פרוייקט", "מיזם", "תוכנית", "שיפוץ", "בנייה", "ארגון", "פרויקט").
   Fields:
     title  — short Hebrew title (4–60 chars).
     status — ONE of "idea" / "in_progress". Default "in_progress". Use "idea"
              when the user says "רעיון" / "חולם על" / "אולי" / "בעתיד".
   Examples:
     "תוסיף פרוייקט: שיפוץ סלון" →
       {"intent":"project","title":"שיפוץ סלון","status":"in_progress"}
     "יש לי רעיון לפרוייקט: גינה על הגג" →
       {"intent":"project","title":"גינה על הגג","status":"idea"}
     "אנחנו מתחילים פרוייקט סידור ארכיון" →
       {"intent":"project","title":"סידור ארכיון","status":"in_progress"}

9. "query_notes" — the user is ASKING about saved notes / reminders.
   Triggers: question words combined with note words ("פתקים", "תזכורות",
   "מה רשמנו", "מה שמרנו", "מה כתוב").
   Fields:
     pinned_only — true ONLY if the user asked specifically about pinned /
                   starred / important notes. Default false.
   Examples:
     "אילו פתקים יש לנו?"   → {"intent":"query_notes","pinned_only":false}
     "מה הפתקים שלנו?"      → {"intent":"query_notes","pinned_only":false}
     "תראה לי את הפתקים"   → {"intent":"query_notes","pinned_only":false}
     "מה הפתקים הפיננסיים?" → {"intent":"query_notes","pinned_only":false}

10. "query_projects" — the user is ASKING about family projects.
    Triggers: question words combined with project words ("פרוייקטים",
    "פרויקטים", "מיזמים", "תוכניות", "מה עובדים על").
    Fields:
      include_done — true if the user asked about finished / done projects.
                     Default false (active projects only).
    Examples:
      "מה הפרוייקטים שלנו?"     → {"intent":"query_projects","include_done":false}
      "אילו פרוייקטים פעילים?"  → {"intent":"query_projects","include_done":false}
      "מה הפרוייקטים שסיימנו?"  → {"intent":"query_projects","include_done":true}

11. "unsupported" — the request is something else (kids' schedules,
    general chat, deleting/updating existing items, weather, budget).
    Field:
      reason — short Hebrew message the bot will show the user.
    Example:
      "מה מזג האוויר?" →
        {"intent":"unsupported","reason":"אני יודע להוסיף ולשאול לגבי אירועים, קניות, משימות, פתקים ופרוייקטים — שאר הדברים עוד לא."}

Rules:
- ALWAYS return valid JSON matching ONE of the schemas above.
- ALWAYS include the top-level "intent" field (see CRITICAL above).
- NEVER add fields not in the schema.
- Hebrew text only in user-visible fields.
- If multiple intents could apply, pick the one the user spent more words
  describing.
"""


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        s = get_settings()
        _client = AsyncOpenAI(api_key=s.openai_api_key)
    return _client


def _now_in_jerusalem() -> datetime:
    return datetime.now(timezone.utc).astimezone(TZ)


async def parse_intent(text: str) -> ParsedIntent:
    """
    Send `text` to OpenAI with the system prompt and parse the response.
    On any failure (network, malformed JSON, missing fields), return an
    UnsupportedIntent so the bot can fail gracefully.
    """
    s = get_settings()
    if not s.openai_api_key:
        return UnsupportedIntent(
            reason="השירות לא מוגדר כראוי (חסר מפתח OpenAI)."
        )

    now = _now_in_jerusalem()
    user_msg = (
        f"CURRENT_DATE: {now.strftime('%Y-%m-%d')} "
        f"(local time {now.strftime('%H:%M')} Asia/Jerusalem)\n\n"
        f"USER_TEXT:\n{text.strip()}"
    )

    try:
        resp = await _get_client().chat.completions.create(
            model=s.openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        content = resp.choices[0].message.content or "{}"
        raw: dict[str, Any] = json.loads(content)
    except Exception as exc:  # noqa: BLE001
        return UnsupportedIntent(reason=f"שגיאת LLM: {exc}")

    intent = raw.get("intent") or _infer_intent_from_shape(raw)
    try:
        if intent == "family_event":
            return FamilyEventIntent.model_validate(raw)
        if intent == "grocery":
            return GroceryIntent.model_validate(raw)
        if intent == "chore":
            return ChoreIntent.model_validate(raw)
        if intent == "note":
            return NoteIntent.model_validate(raw)
        if intent == "project":
            return ProjectIntent.model_validate(raw)
        if intent == "query_events":
            return QueryEventsIntent.model_validate(raw)
        if intent == "query_grocery":
            return QueryGroceryIntent.model_validate(raw)
        if intent == "query_chores":
            return QueryChoresIntent.model_validate(raw)
        if intent == "query_notes":
            return QueryNotesIntent.model_validate(raw)
        if intent == "query_projects":
            return QueryProjectsIntent.model_validate(raw)
        if intent == "unsupported":
            return UnsupportedIntent.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        return UnsupportedIntent(reason=f"שגיאה בפענוח הבקשה: {exc}")

    return UnsupportedIntent(
        reason="לא הצלחתי להבין את הבקשה. נסו לנסח אחרת."
    )


def _infer_intent_from_shape(raw: dict[str, Any]) -> str | None:
    """
    Defense in depth: gpt-4o-mini sometimes drops the explicit `intent`
    field when the surrounding fields make the intent obvious (see the
    prompt's "CRITICAL" rule which it's *supposed* to follow). Map the
    shape back to an intent so the dispatch still works.

    Field disambiguation (kept narrow — must be unique per intent):
      items                        → grocery
      body                         → note
      start_minutes / end_minutes  → family_event  (chore has no time)
      range                        → query_events  (chore has no range)
      mine / today                 → query_chores
      pinned_only                  → query_notes
      include_done                 → query_projects
      status (+ title)             → project  (chore has no status field)
      assigned_to or just title    → chore
      only `reason`                → unsupported
    """
    keys = set(raw.keys())
    if "items" in keys:
        return "grocery"
    if "body" in keys:
        return "note"
    if "start_minutes" in keys or "end_minutes" in keys:
        return "family_event"
    if "range" in keys:
        return "query_events"
    if "mine" in keys or "today" in keys:
        return "query_chores"
    if "pinned_only" in keys:
        return "query_notes"
    if "include_done" in keys:
        return "query_projects"
    if "assigned_to" in keys:
        return "chore"
    if keys == {"reason"} or keys == {"reason", "intent"}:
        return "unsupported"
    # `status` with `title` is a project; `title` alone leans chore.
    if "status" in keys and "title" in keys:
        return "project"
    if "title" in keys:
        return "chore"
    return None
