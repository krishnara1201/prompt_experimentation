"""Shared test fixtures.

The production engine in ``app.db.session`` uses SQLAlchemy's default
``AsyncAdaptedQueuePool``. That is correct for a long-lived API server
running on a single event loop, but wrong for tests: several test modules
drive the database from separate ``asyncio.run()`` calls (and the FastAPI
``TestClient`` runs the app on its own portal loop). asyncpg connections are
bound to the loop that created them, so a pooled connection handed to a
second loop raises ``RuntimeError: ... attached to a different loop`` /
``Event loop is closed``.

Tests therefore use a ``NullPool`` engine defined here, which opens and
closes a fresh connection per checkout and is safe across loops.
"""
import socket
from urllib.parse import urlsplit

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.db.session import DATABASE_URL


def postgres_reachable() -> bool:
    """True when a TCP connection to the configured Postgres can be opened."""
    if not DATABASE_URL:
        return False
    parts = urlsplit(DATABASE_URL.replace("+asyncpg", ""))
    try:
        with socket.create_connection((parts.hostname or "localhost", parts.port or 5432), timeout=1):
            return True
    except OSError:
        return False


# Module-level so it can be shared by every DB-touching test module. NullPool
# means no connection is ever cached, so nothing is carried between event
# loops. Creating the engine does not open a connection.
db_test_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)


@pytest.fixture(scope="session")
def db_engine():
    """Session-scoped NullPool engine for tests that talk to Postgres.

    Most current call sites need the engine from module-level helper
    functions (outside any fixture), so they import ``db_test_engine``
    directly. This fixture exposes the same single engine to tests that
    would rather request it by injection.
    """
    return db_test_engine
