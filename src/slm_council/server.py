"""FastAPI application – the HTTP gateway to the SLM Coding Council."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from slm_council.config import settings
from slm_council.models import CouncilResult, UserRequest
from slm_council.orchestrator.loop import CouncilLoop
from slm_council.agents.registry import registry
from slm_council.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    setup_logging()
    logger.info("server.startup", port=settings.api_port)
    yield
    logger.info("server.shutdown")



app = FastAPI(
    title="SLM Coding Council",
    description="Many-to-One autonomous coding engine powered by specialised SLMs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The coding task or question.")
    language: str = Field(default="python", description="Target programming language.")
    context: dict[str, Any] = Field(default_factory=dict, description="Optional extra context.")
    max_iterations: int | None = Field(default=None, ge=1, le=50, description="Optional per-request refinement limit.")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    timestamp: str = ""


@app.get("/")
async def root():
    return {"message": "SLM Coding Council API is running", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/council/run", response_model=CouncilResult)
async def run_council(req: QueryRequest) -> CouncilResult:
    """Submit a coding task to the full council pipeline."""
    request_context = dict(req.context)
    if req.max_iterations is not None:
        request_context["max_iterations"] = req.max_iterations

    user_req = UserRequest(
        query=req.query,
        language=req.language,
        context=request_context,
    )
    loop = CouncilLoop()
    try:
        result = await loop.run(user_req)
    except Exception as exc:
        logger.error("api.council_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@app.get("/config/agents")
async def get_agent_config() -> dict[str, Any]:
    """Return the current agent endpoint configuration (no secrets)."""
    agent_configs: dict[str, Any] = {}
    agent_names = [
        "researcher", "planner", "generator", "reviewer",
        "debugger", "tester", "optimizer", "refactorer",
    ]
    for name in agent_names:
        endpoint = getattr(settings, f"{name}_endpoint", "")
        model = getattr(settings, f"{name}_model", "")
        agent_configs[name] = {
            "endpoint": endpoint,
            "model": model,
            "configured": bool(endpoint and model),
        }

    return {
        "orchestrator": {
            "model": settings.orchestrator_model_name,
        },
        "agents": agent_configs,
        "registered_roles": [r.value for r in registry.available_roles()],
        "max_refinement_passes": settings.max_refinement_passes,
        "consensus_threshold": settings.consensus_threshold,
    }
