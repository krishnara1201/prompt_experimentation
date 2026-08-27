from fastapi import FastAPI

from app.api.routes.runs import router as runs_router
from app.db.session import lifespan

app = FastAPI(title="Prompt Experimentation API", lifespan=lifespan)
app.include_router(runs_router)
