import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import init_db, close_db

# Import all route modules directly
from app.api import user_routes
from app.api import family_routes
from app.api import family_member_routes
from app.api import task_routes
from app.api import reminder_routes
from app.api import recurring_pattern_routes
from app.api import telegram_routes

# App-level logging at INFO. uvicorn doesn't configure the root logger, so
# without this our logging.getLogger(__name__).info(...) calls are dropped
# (only WARNING+ reaches the last-resort handler). Enables observability for
# the bot intent/brain flow.
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # initialize resources here (db, clients, caches)
    print("🚀 Starting up application...")
    await init_db()
    print("✅ Database initialized")
    yield
    # clean up resources here
    print("🛑 Shutting down application...")
    await close_db()
    print("✅ Database closed")


app = FastAPI(
    title="Family AI Assistant API",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# Include all routers with /api prefix
app.include_router(user_routes.router, prefix="/api")
app.include_router(family_routes.router, prefix="/api")
app.include_router(family_member_routes.router, prefix="/api")
app.include_router(task_routes.router, prefix="/api")
app.include_router(reminder_routes.router, prefix="/api")
app.include_router(recurring_pattern_routes.router, prefix="/api")

# Telegram router is mounted at root (no /api prefix) — the family-os
# frontend hardcodes the URL as ${ASSISTANT_URL}/telegram/generate-code,
# and Telegram itself just POSTs to whatever webhook we register.
app.include_router(telegram_routes.router)


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    return {"message": "Family AI Assistant API"}


@app.get("/api/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/test", tags=["meta"])
async def test_endpoint() -> dict[str, str]:
    return {"message": "Test endpoint works!", "routes_count": len([r for r in app.routes if hasattr(r, 'methods')])}


# Print all registered routes - happens at import time
print("\n" + "="*80)
print("🛣️  ALL REGISTERED ROUTES (at import):")
print("="*80)
for route in app.routes:
    if hasattr(route, "methods") and hasattr(route, "path"):
        methods = ", ".join(sorted(route.methods))
        print(f"  {methods:12} {route.path}")
print("="*80 + "\n")


@app.on_event("startup")
async def print_routes_on_startup():
    print("\n" + "="*80)
    print("🛣️  ALL ROUTES AT STARTUP:")
    print("="*80)
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            methods = ", ".join(sorted(route.methods))
            print(f"  {methods:12} {route.path}")
    print("="*80)
    print(f"✅ Total routes: {len([r for r in app.routes if hasattr(r, 'methods')])}")
    print("="*80 + "\n")