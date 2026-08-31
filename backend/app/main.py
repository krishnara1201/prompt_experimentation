from fastapi import FastAPI

from app.api.routes.arms import router as arms_router
from app.api.routes.health import router as health_router
from app.api.routes.runs import router as runs_router
from app.api.routes.stats import router as stats_router
from app.api.routes.tasks import router as tasks_router
from app.db.session import lifespan

app = FastAPI(title="Prompt Experimentation API", lifespan=lifespan)
app.include_router(health_router)
app.include_router(arms_router)
app.include_router(runs_router)
app.include_router(stats_router)
app.include_router(tasks_router)
