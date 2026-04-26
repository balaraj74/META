"""Router-based FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from triage.api.middleware.cors import add_cors_middleware
from triage.api.middleware.logging import add_logging_middleware
from triage.api.routers import (
    agents_router,
    command_router,
    episodes_router,
    metrics_router,
    openenv_router,
    patients_router,
    training_router,
    websocket_router,
)
from triage.api.schemas import ApiResponse, EpisodeConfig, HealthResponse, TrainingConfig
from triage.api.service import backend_service
from triage.agents.model_router import ModelRouter


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    actual_mode = ModelRouter.initialize_from_env()
    logger.info("ModelRouter initialized in %s mode: %s", actual_mode, ModelRouter.get_instance().status())
    await backend_service.initialize()
    yield
    await backend_service.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="TRIAGE API",
        description="AI Hospital Crisis Simulation Environment",
        version="1.0.0",
        lifespan=lifespan,
    )
    add_cors_middleware(app)
    add_logging_middleware(app)

    app.include_router(episodes_router, prefix="/api/episodes", tags=["Episodes"])
    app.include_router(agents_router, prefix="/api/agents", tags=["Agents"])
    app.include_router(patients_router, prefix="/api/patients", tags=["Patients"])
    app.include_router(metrics_router, prefix="/api/metrics", tags=["Metrics"])
    app.include_router(training_router, prefix="/api/training", tags=["Training"])
    app.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])
    app.include_router(command_router, tags=["Command Center"])
    app.include_router(openenv_router, tags=["OpenEnv"])

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(**backend_service.health_payload())

    # Compatibility aliases for the original /api/simulation surface.
    @app.post("/api/simulation/start", response_model=ApiResponse, include_in_schema=False)
    async def compat_start(config: EpisodeConfig) -> ApiResponse:
        return ApiResponse(success=True, data=await backend_service.start_episode(config))

    @app.post("/api/simulation/step", response_model=ApiResponse, include_in_schema=False)
    async def compat_step() -> ApiResponse:
        session = backend_service.get_latest_episode()
        if session is None:
            return ApiResponse(success=False, error="No active episode")
        return ApiResponse(success=True, data=await backend_service.step_episode(session.episode_id))

    @app.post("/api/simulation/stop", response_model=ApiResponse, include_in_schema=False)
    async def compat_stop() -> ApiResponse:
        return ApiResponse(success=True, data=await backend_service.stop_latest_episode())

    @app.get("/api/simulation/state", response_model=ApiResponse, include_in_schema=False)
    async def compat_state() -> ApiResponse:
        session = backend_service.get_latest_episode()
        data = backend_service.build_state_payload(session) if session else {"status": "idle"}
        return ApiResponse(success=True, data=data)

    @app.get("/api/simulation/history", response_model=ApiResponse, include_in_schema=False)
    async def compat_history() -> ApiResponse:
        session = backend_service.get_latest_episode()
        data = backend_service.episode_history_payload(session) if session else {"steps": [], "messages": []}
        return ApiResponse(success=True, data=data)

    @app.post("/api/training/start", response_model=ApiResponse, include_in_schema=False)
    async def compat_training_start(config: EpisodeConfig) -> ApiResponse:
        return ApiResponse(
            success=True,
            data=await backend_service.start_dpo_training(
                TrainingConfig(
                    n_episodes=5,
                    difficulty=config.difficulty,
                    mock_llm=config.mock_llm,
                    mock_training=True,
                )
            ),
        )

    return app


app = create_app()


def run() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    )
    uvicorn.run(
        "triage.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
