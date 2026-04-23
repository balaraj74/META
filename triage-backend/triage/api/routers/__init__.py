"""API router exports — lazy imports so one broken router can't crash the rest."""

# Each router is imported inside a try/except so that missing optional
# dependencies (e.g. sqlalchemy) only skip that specific router instead of
# taking down the whole application.

try:
    from triage.api.routers.command import router as command_router
except Exception:  # pragma: no cover
    command_router = None  # type: ignore[assignment]

try:
    from triage.api.routers.agents import router as agents_router
except Exception:  # pragma: no cover
    agents_router = None  # type: ignore[assignment]

try:
    from triage.api.routers.episodes import router as episodes_router
except Exception:  # pragma: no cover
    episodes_router = None  # type: ignore[assignment]

try:
    from triage.api.routers.metrics import router as metrics_router
except Exception:  # pragma: no cover
    metrics_router = None  # type: ignore[assignment]

try:
    from triage.api.routers.patients import router as patients_router
except Exception:  # pragma: no cover
    patients_router = None  # type: ignore[assignment]

try:
    from triage.api.routers.training import router as training_router
except Exception:  # pragma: no cover
    training_router = None  # type: ignore[assignment]

try:
    from triage.api.routers.websocket import router as websocket_router
except Exception:  # pragma: no cover
    websocket_router = None  # type: ignore[assignment]

try:
    from triage.api.routers.openenv_routes import router as openenv_router
except Exception:  # pragma: no cover
    openenv_router = None  # type: ignore[assignment]

__all__ = [
    "agents_router",
    "command_router",
    "episodes_router",
    "metrics_router",
    "openenv_router",
    "patients_router",
    "training_router",
    "websocket_router",
]
