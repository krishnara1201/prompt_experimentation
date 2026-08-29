from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config.arms import load_arms

router = APIRouter(prefix="/arms", tags=["arms"])

# Same file the run-creation route resolves arms against.
ARMS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "arms.yaml"


class ArmInfo(BaseModel):
    name: str
    adapter: str
    model: str | None


@router.get("", response_model=list[ArmInfo])
def list_arms() -> list[ArmInfo]:
    # load_arms validates the config and instantiates every adapter, so a
    # broken arms.yaml fails here the same way POST /runs would. It drops
    # the adapter-type string, so re-read the raw YAML for that one field.
    try:
        adapters = load_arms(str(ARMS_PATH))
        raw = yaml.safe_load(ARMS_PATH.read_text())
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Cannot load arms.yaml: {exc}") from exc

    adapter_type_by_name = {entry["name"]: entry.get("adapter", "") for entry in raw["arms"]}
    return [
        ArmInfo(
            name=name,
            adapter=adapter_type_by_name.get(name, ""),
            model=getattr(adapter, "model", None),
        )
        for name, adapter in adapters.items()
    ]
