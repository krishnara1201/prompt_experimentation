from fastapi import APIRouter, Depends, Response
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.session import get_session

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(response: Response, session: AsyncSession = Depends(get_session)) -> dict[str, str]:
    """Liveness/readiness probe: reports API process health and DB connectivity.

    Returns 200 with ``database: "ok"`` when a trivial query succeeds, or 503
    with ``database: "unreachable"`` when the connection or query fails. Used by
    the docker-compose ``api`` healthcheck and the ``pe`` CLI readiness poll.
    """
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        response.status_code = 503
        return {"status": "error", "database": "unreachable"}
    return {"status": "ok", "database": "ok"}
