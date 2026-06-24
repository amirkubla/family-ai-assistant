"""
HTTP client for the family-os REST API.

The Assistant is a separate service and a separate database. To create
events or grocery items in family-os we POST through its REST API rather
than reaching into its database directly. This keeps family-os as the
single source of truth and respects its validation layer.

Auth: a shared bearer token (`FAMILY_OS_SERVICE_TOKEN`) configured on both
sides. family-os's internal-routes middleware accepts any request that
presents a matching token. There's no per-user identity on these calls —
the family_id is the only authorization scope.
"""
from __future__ import annotations

from typing import Any

import httpx

from app.core.config import get_settings


class FamilyOsClient:
    """Thin async wrapper over httpx — one method per family-os operation we need."""

    def __init__(self) -> None:
        s = get_settings()
        self._base = s.family_os_api_url.rstrip("/")
        self._token = s.family_os_service_token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    async def create_family_event(
        self,
        family_id: str,
        *,
        title: str,
        start_minutes: int,
        end_minutes: int,
        is_recurring: bool = False,
        days_of_week: list[int] | None = None,
        date: str | None = None,
        assignee_type: str = "family",
        location: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a one-time or recurring family event.

        For one-time events: pass `date="YYYY-MM-DD"`, leave `days_of_week`
        empty, `is_recurring=False`.
        For recurring: pass `days_of_week` as 0-6 (Sun=0), `is_recurring=True`.
        """
        body = {
            "title": title,
            "startMinutes": start_minutes,
            "endMinutes": end_minutes,
            "isRecurring": is_recurring,
            "daysOfWeek": days_of_week or [],
            "assigneeType": assignee_type,
        }
        if date is not None:
            body["date"] = date
        if location is not None:
            body["location"] = location

        url = f"{self._base}/v1/internal/family/{family_id}/family-events"
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(url, headers=self._headers(), json=body)
            r.raise_for_status()
            return r.json()

    async def create_note(
        self,
        family_id: str,
        *,
        body: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Add a note. Always created unpinned; user pins manually in the app."""
        payload: dict[str, Any] = {"body": body}
        if title is not None:
            payload["title"] = title

        url = f"{self._base}/v1/internal/family/{family_id}/notes"
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(url, headers=self._headers(), json=payload)
            r.raise_for_status()
            return r.json()

    async def create_chore(
        self,
        family_id: str,
        *,
        title: str,
        assigned_to: str | None = None,
    ) -> dict[str, Any]:
        """
        Add a chore. `assigned_to` is free-text Hebrew (e.g. "עודד"); the
        server tries to resolve it to a known familyMember.displayName and
        link `assignedToMemberId`, falling back to free-text on no match.
        """
        body: dict[str, Any] = {"title": title}
        if assigned_to is not None:
            body["assignedTo"] = assigned_to

        url = f"{self._base}/v1/internal/family/{family_id}/chores"
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(url, headers=self._headers(), json=body)
            r.raise_for_status()
            return r.json()

    async def create_grocery_item(
        self,
        family_id: str,
        *,
        title: str,
        qty: str | None = None,
        shopping_category: str = "grocery",
        subcategory: str | None = None,
    ) -> dict[str, Any]:
        """Add an item to the family's shopping list."""
        body: dict[str, Any] = {
            "title": title,
            "shoppingCategory": shopping_category,
        }
        if qty is not None:
            body["qty"] = qty
        if subcategory is not None:
            body["subcategory"] = subcategory

        url = f"{self._base}/v1/internal/family/{family_id}/grocery"
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(url, headers=self._headers(), json=body)
            r.raise_for_status()
            return r.json()


    # ── reads ─────────────────────────────────────────────────────────────

    async def list_family_events(
        self,
        family_id: str,
        *,
        range_: str = "today",
        kid_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        `range_`: 'today' / 'tomorrow' / 'week'.
        `kid_name`: optional Hebrew name; server filters to events assigned
                    to this kid (assigneeType='kid' + matching assigneeId).
        """
        url = f"{self._base}/v1/internal/family/{family_id}/family-events"
        params: dict[str, str] = {"range": range_}
        if kid_name is not None:
            params["kidName"] = kid_name
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, headers=self._headers(), params=params)
            r.raise_for_status()
            return r.json()

    async def list_schedule_blocks(
        self,
        family_id: str,
        *,
        range_: str = "today",
        kid_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Kid's weekly schedule (school / hobby / other). Same range semantics
        as family-events. kid_name is optional; without it the server returns
        all blocks in range for the family (not useful for the bot — pass it).
        """
        url = f"{self._base}/v1/internal/family/{family_id}/schedule-blocks"
        params: dict[str, str] = {"range": range_}
        if kid_name is not None:
            params["kidName"] = kid_name
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, headers=self._headers(), params=params)
            r.raise_for_status()
            return r.json()

    async def list_grocery(
        self,
        family_id: str,
        *,
        status: str = "unchecked",
    ) -> list[dict[str, Any]]:
        """`status` is one of: 'unchecked' (default) / 'all'."""
        url = f"{self._base}/v1/internal/family/{family_id}/grocery"
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(
                url, headers=self._headers(), params={"status": status}
            )
            r.raise_for_status()
            return r.json()

    async def list_chores(
        self,
        family_id: str,
        *,
        status: str = "undone",
        assignee_member_id: str | None = None,
        selected_for_today: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        `status`: 'undone' (default) / 'all'.
        `assignee_member_id`: filter to chores assigned to this familyMember.
        `selected_for_today`: filter on the selectedForToday flag.
        """
        url = f"{self._base}/v1/internal/family/{family_id}/chores"
        params: dict[str, str] = {"status": status}
        if assignee_member_id is not None:
            params["assigneeMemberId"] = assignee_member_id
        if selected_for_today is not None:
            params["selectedForToday"] = "true" if selected_for_today else "false"
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, headers=self._headers(), params=params)
            r.raise_for_status()
            return r.json()

    async def list_members(self, family_id: str) -> list[dict[str, Any]]:
        """List active family members (id, displayName, role, avatarEmoji)."""
        url = f"{self._base}/v1/internal/family/{family_id}/members"
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, headers=self._headers())
            r.raise_for_status()
            return r.json()

    async def list_kids(self, family_id: str) -> list[dict[str, Any]]:
        """List active kids (id, name, emoji)."""
        url = f"{self._base}/v1/internal/family/{family_id}/kids"
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, headers=self._headers())
            r.raise_for_status()
            return r.json()

    async def list_notes(
        self, family_id: str, *, kid_name: str | None = None
    ) -> list[dict[str, Any]]:
        """
        List family notes sorted pinned-first (id, title, body, pinned, kidId).
        `kid_name` optionally scopes to one kid's notes (kid-owned via kidId).
        """
        url = f"{self._base}/v1/internal/family/{family_id}/notes"
        params: dict[str, str] = {}
        if kid_name is not None:
            params["kidName"] = kid_name
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, headers=self._headers(), params=params)
            r.raise_for_status()
            return r.json()

    async def list_projects(
        self, family_id: str, *, status: str = "active", kid_name: str | None = None
    ) -> list[dict[str, Any]]:
        """
        `status`: 'active' (default, idea+in_progress) / 'done' / 'all'.
        `kid_name` optionally scopes to one kid's projects (kid-owned via kidId).
        """
        url = f"{self._base}/v1/internal/family/{family_id}/projects"
        params: dict[str, str] = {"status": status}
        if kid_name is not None:
            params["kidName"] = kid_name
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, headers=self._headers(), params=params)
            r.raise_for_status()
            return r.json()

    async def create_project(
        self,
        family_id: str,
        *,
        title: str,
        status: str = "in_progress",
    ) -> dict[str, Any]:
        """Create a new family project."""
        url = f"{self._base}/v1/internal/family/{family_id}/projects"
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(url, headers=self._headers(), json={"title": title, "status": status})
            r.raise_for_status()
            return r.json()

    # ── kid payments ────────────────────────────────────────────────────────

    async def list_payments(
        self, family_id: str, *, kid_name: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Outstanding kid payments with server-computed next-due dates.
        Each row: {id, note, amount (agorot), kidId, dueDate, isRecurring,
        recurrenceType}. `kid_name` optionally scopes to one kid.
        """
        url = f"{self._base}/v1/internal/family/{family_id}/payments"
        params: dict[str, str] = {}
        if kid_name is not None:
            params["kidName"] = kid_name
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, headers=self._headers(), params=params)
            r.raise_for_status()
            return r.json()

    async def create_payment(
        self,
        family_id: str,
        *,
        kid_name: str,
        note: str,
        amount: int,  # agorot
        date: str | None = None,
        is_recurring: bool = False,
        recurrence_type: str | None = None,
        recurrence_day: int | None = None,
    ) -> dict[str, Any]:
        """Create a kid payment (one-time or recurring). amount in agorot."""
        body: dict[str, Any] = {
            "kidName": kid_name,
            "note": note,
            "amount": amount,
            "isRecurring": is_recurring,
        }
        if date is not None:
            body["date"] = date
        if recurrence_type is not None:
            body["recurrenceType"] = recurrence_type
        if recurrence_day is not None:
            body["recurrenceDay"] = recurrence_day
        url = f"{self._base}/v1/internal/family/{family_id}/payments"
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(url, headers=self._headers(), json=body)
            r.raise_for_status()
            return r.json()

    async def pay_payment(self, family_id: str, payment_id: str) -> dict[str, Any]:
        """Settle a kid payment by id (occurrence for recurring, toggle for one-time)."""
        url = f"{self._base}/v1/internal/family/{family_id}/payments/pay"
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(url, headers=self._headers(), json={"id": payment_id})
            r.raise_for_status()
            return r.json()

    # ── expenses (general spending) ──────────────────────────────────────────

    async def create_expense(
        self,
        family_id: str,
        *,
        amount: int,  # agorot
        category_name: str | None = None,
        note: str | None = None,
        date: str | None = None,
    ) -> dict[str, Any]:
        """Log a settled expense in any budget category. amount in agorot."""
        body: dict[str, Any] = {"amount": amount}
        if category_name is not None:
            body["categoryName"] = category_name
        if note is not None:
            body["note"] = note
        if date is not None:
            body["date"] = date
        url = f"{self._base}/v1/internal/family/{family_id}/expenses"
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(url, headers=self._headers(), json=body)
            r.raise_for_status()
            return r.json()

    async def list_expenses(
        self, family_id: str, *, month: str | None = None
    ) -> dict[str, Any]:
        """Paid expenses for a month (default current). Returns {month, expenses}."""
        url = f"{self._base}/v1/internal/family/{family_id}/expenses"
        params: dict[str, str] = {}
        if month is not None:
            params["month"] = month
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, headers=self._headers(), params=params)
            r.raise_for_status()
            return r.json()


family_os_client = FamilyOsClient()
