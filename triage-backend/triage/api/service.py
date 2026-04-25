"""Shared backend services for episodes, metrics, persistence, and training."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, WebSocket
from sqlalchemy import delete, func, select

from config.settings import Settings
from triage.agents.orchestrator import AgentOrchestrator, OrchestratorStepResult
from triage.agents.strategy_memory import StrategyMemory
from triage.api.schemas import (
    AgentOverrideRequest,
    DriftEventSnapshot,
    EpisodeConfig,
    EpisodeHistory,
    EpisodeRunRequest,
    MetricsComparison,
    RewardCurvePoint,
    ResourcePoint,
    SimulationState,
    SimulationStatus,
    TrainingConfig,
    TrainingStatus,
    WSMessage,
)
from triage.db import (
    AgentMessageRecord,
    EpisodeRecord,
    PatientRecord,
    RewardRecord,
    StrategyLessonRecord,
    get_session,
    init_db,
)
from triage.env.state import AgentType
from triage.training.dataset_adapter import DatasetAdapter
from triage.training.dpo_trainer import TRIAGEDPOTrainer, TrainingConfig as DPOTrainingConfig
from triage.training.preference_labeler import PreferenceLabeler
from triage.training.reporting import generate_training_report
from triage.training.trajectory_collector import Trajectory, TrajectoryCollector

logger = logging.getLogger(__name__)


@dataclass
class EpisodeSession:
    """In-memory runtime handle for an episode."""

    episode_id: str
    config: EpisodeConfig
    orchestrator: AgentOrchestrator
    status: SimulationStatus = SimulationStatus.RUNNING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_task: asyncio.Task | None = None
    persisted_message_ids: set[str] = field(default_factory=set)
    resource_history: list[dict[str, Any]] = field(default_factory=list)


class BackendService:
    """Application service layer used by API, scripts, and worker flows."""

    def __init__(self) -> None:
        self.settings = Settings()
        self._episodes: dict[str, EpisodeSession] = {}
        self._episode_order: list[str] = []
        self._websockets: list[WebSocket] = []
        self._startup_time = datetime.now(timezone.utc)
        self._training_status = TrainingStatus(phase="not_started", progress=0.0)
        self._training_task: asyncio.Task | None = None
        self._last_trajectories: list[Trajectory] = []
        self._last_pairs: list[dict[str, Any]] = []
        self._last_output_dir = "./data/episodes"

    async def initialize(self) -> None:
        init_db()

    async def shutdown(self) -> None:
        for session in self._episodes.values():
            if session.run_task and not session.run_task.done():
                session.run_task.cancel()

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "version": "1.0.0",
            "uptime_seconds": round(
                (datetime.now(timezone.utc) - self._startup_time).total_seconds(),
                1,
            ),
            "components": {
                "episodes": "ready",
                "database": "ready",
                "training": self._training_status.phase,
            },
        }

    async def start_episode(self, config: EpisodeConfig) -> dict[str, Any]:
        episode_id = str(uuid.uuid4())
        orchestrator = AgentOrchestrator(
            agents_config_path=str(self.settings.agents_yaml_path),
            mock_llm=config.mock_llm,
            seed=config.seed or 42,
            max_steps=config.max_steps,
            difficulty=config.difficulty,
        )
        scenario = {"difficulty": config.difficulty}
        if config.crisis_type:
            scenario["crisis_type"] = config.crisis_type.value
        await orchestrator.reset(scenario)

        session = EpisodeSession(
            episode_id=episode_id,
            config=config,
            orchestrator=orchestrator,
            status=SimulationStatus.RUNNING,
        )
        self._episodes[episode_id] = session
        self._episode_order.append(episode_id)
        self._persist_episode_snapshot(session)
        await self._broadcast("state_update", self.build_state_payload(session))

        if config.auto_step:
            session.run_task = asyncio.create_task(
                self.run_episode(
                    episode_id,
                    EpisodeRunRequest(delay_ms=config.step_delay_ms, max_steps=config.max_steps),
                )
            )

        return {
            "episode_id": episode_id,
            "episode_num": session.orchestrator.state.episode,
            "crisis_type": session.orchestrator.state.crisis.type.value,
            "status": session.status.value,
        }

    async def step_episode(self, episode_id: str) -> dict[str, Any]:
        session = self.get_episode(episode_id)
        if session.status not in (SimulationStatus.RUNNING, SimulationStatus.PAUSED):
            raise HTTPException(400, f"Episode {episode_id} is not running")

        result = await session.orchestrator.step()
        session.updated_at = datetime.now(timezone.utc)
        session.status = SimulationStatus.COMPLETED if result.terminated else SimulationStatus.RUNNING
        self._append_resource_history(session)
        self._persist_step(session, result)

        payload = self._format_step_result(session, result)
        await self._broadcast("step_complete", payload)
        await self._broadcast("reward_update", {"episode_id": episode_id, "reward": result.breakdown})
        
        if result.drift_events:
            for drift_event in result.drift_events:
                await self._broadcast("drift_event", {"episode_id": episode_id, "event": drift_event})
                
        new_blocks = [b.to_dict() for b in session.orchestrator.state.safety_blocks if b.step == session.orchestrator.state.step_count]
        for block in new_blocks:
            await self._broadcast("safety_block", {"episode_id": episode_id, "block": block})

        if result.terminated:
            await self._broadcast("episode_end", self.episode_summary_payload(session))
        else:
            await self._broadcast("state_update", self.build_state_payload(session))
        return payload

    async def run_episode(self, episode_id: str, request: EpisodeRunRequest | None = None) -> dict[str, Any]:
        session = self.get_episode(episode_id)
        request = request or EpisodeRunRequest()
        max_steps = request.max_steps or session.config.max_steps
        while (
            session.status == SimulationStatus.RUNNING
            and not session.orchestrator.env.is_terminal
            and session.orchestrator.state.step_count < max_steps
        ):
            result = await self.step_episode(episode_id)
            if result["terminated"]:
                break
            if request.delay_ms > 0:
                await asyncio.sleep(request.delay_ms / 1000.0)
        return self.episode_summary_payload(session)

    async def reset_episode(self, episode_id: str) -> dict[str, Any]:
        session = self.get_episode(episode_id)
        if session.run_task and not session.run_task.done():
            session.run_task.cancel()
        session.orchestrator = AgentOrchestrator(
            agents_config_path=str(self.settings.agents_yaml_path),
            mock_llm=session.config.mock_llm,
            seed=session.config.seed or 42,
            max_steps=session.config.max_steps,
            difficulty=session.config.difficulty,
        )
        scenario = {"difficulty": session.config.difficulty}
        if session.config.crisis_type:
            scenario["crisis_type"] = session.config.crisis_type.value
        await session.orchestrator.reset(scenario)
        session.status = SimulationStatus.RUNNING
        session.updated_at = datetime.now(timezone.utc)
        session.persisted_message_ids.clear()
        session.resource_history = []
        self._clear_episode_records(episode_id)
        self._persist_episode_snapshot(session)
        await self._broadcast("state_update", self.build_state_payload(session))
        return self.build_state_payload(session)

    async def stop_latest_episode(self) -> dict[str, Any]:
        session = self.get_latest_episode()
        if session is None:
            raise HTTPException(404, "No episodes available")
        if session.run_task and not session.run_task.done():
            session.run_task.cancel()
        session.status = SimulationStatus.PAUSED
        session.updated_at = datetime.now(timezone.utc)
        self._persist_episode_snapshot(session)
        return {"status": "stopped", "episode_id": session.episode_id}

    def get_episode(self, episode_id: str) -> EpisodeSession:
        session = self._episodes.get(episode_id)
        if session is None:
            raise HTTPException(404, f"Episode {episode_id} not found")
        return session

    def get_latest_episode(self) -> EpisodeSession | None:
        if not self._episode_order:
            return None
        return self._episodes[self._episode_order[-1]]

    def list_episodes(self) -> list[dict[str, Any]]:
        with get_session() as db:
            rows = db.scalars(select(EpisodeRecord).order_by(EpisodeRecord.started_at.desc())).all()
        if rows:
            return [
                {
                    "episode_id": row.id,
                    "crisis_type": row.crisis_type,
                    "steps": row.total_steps,
                    "total_reward": row.total_reward,
                    "survival_rate": row.survival_rate,
                    "deceased": 0,
                    "discharged": 0,
                    "duration_seconds": 0.0,
                }
                for row in rows
            ]
        return [self.episode_summary_payload(self._episodes[episode_id]) for episode_id in self._episode_order]

    def build_state_payload(self, session: EpisodeSession) -> dict[str, Any]:
        state = session.orchestrator.state
        recent_actions = []
        for item in session.orchestrator.step_history[-10:]:
            action = item["action"]
            agent_index = action.get("agent_id", 0)
            agent_name = list(AgentType)[agent_index].value if agent_index < len(AgentType) else "unknown"
            recent_actions.append(
                {
                    "step": item["step"],
                    "agent": agent_name,
                    "action_type": str(action.get("action_type")),
                    "target": str(action.get("target_id", "")),
                    "reasoning": action.get("reasoning", ""),
                    "reward": item["breakdown"]["total"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        drift_events = getattr(session.orchestrator.env, "_drift_events", [])
        return SimulationState(
            episode_id=session.episode_id,
            episode_num=state.episode,
            status=session.status,
            step=state.step_count,
            max_steps=session.config.max_steps,
            crisis_type=state.crisis.type.value,
            difficulty=session.config.difficulty,
            patients=[self._patient_snapshot(patient) for patient in state.patients],
            resources=self._resource_snapshot(state),
            agents=self._agent_snapshots(session),
            metrics=self._metrics_snapshot(session),
            recent_actions=recent_actions,
            drift_events=[
                DriftEventSnapshot(
                    step=index + 1,
                    event_type=event.get("type", "drift"),
                    description=event.get("description", json.dumps(event)),
                    impact=event.get("impact", "medium"),
                )
                for index, event in enumerate(drift_events[-10:])
            ],
        ).model_dump()

    def episode_history_payload(self, session: EpisodeSession) -> dict[str, Any]:
        return EpisodeHistory(
            episode_id=session.episode_id,
            steps=session.orchestrator.step_history,
            messages=[message.to_dict() for message in session.orchestrator.bus.history],
        ).model_dump()

    def episode_summary_payload(self, session: EpisodeSession) -> dict[str, Any]:
        state = session.orchestrator.state
        return {
            "episode_id": session.episode_id,
            "crisis_type": state.crisis.type.value,
            "steps": state.step_count,
            "total_reward": round(session.orchestrator.total_reward, 4),
            "survival_rate": round(state.survival_rate, 4),
            "deceased": state.deceased_count,
            "discharged": state.discharged_count,
            "duration_seconds": round(
                (session.updated_at - session.created_at).total_seconds(),
                2,
            ),
        }

    def get_agent_statuses(self) -> list[dict[str, Any]]:
        session = self.get_latest_episode()
        if session is None:
            return []
        return self._agent_snapshots(session)

    def get_agent_status(self, agent_type: AgentType) -> dict[str, Any]:
        session = self.get_latest_episode()
        if session is None:
            raise HTTPException(404, "No episodes available")
        for agent in self._agent_snapshots(session):
            if agent["agent_type"] == agent_type.value:
                return agent
        raise HTTPException(404, f"Agent {agent_type.value} not found")

    def get_agent_messages(self, agent_type: AgentType) -> list[dict[str, Any]]:
        session = self.get_latest_episode()
        if session is None:
            return []
        return session.orchestrator.get_agent_messages(agent_type)

    def get_message_bus_stats(self) -> dict[str, Any]:
        session = self.get_latest_episode()
        if session is None:
            return {}
        # Assuming session.orchestrator has the 'bus' attribute natively.
        if hasattr(session.orchestrator, "bus"):
            return session.orchestrator.bus.stats()
        elif hasattr(session.orchestrator, "message_bus"):
            return session.orchestrator.message_bus.stats()
        return {}

    async def override_agent(self, agent_type: AgentType, request: AgentOverrideRequest) -> dict[str, Any]:
        session = self.get_latest_episode()
        if session is None:
            raise HTTPException(404, "No episodes available")
        result = await session.orchestrator.manual_override(
            agent_type=agent_type,
            action_type=request.action_type,
            target_id=request.target_id,
            priority=request.priority,
            reasoning=request.reasoning,
            reasoning_tokens=request.reasoning_tokens,
        )
        session.status = SimulationStatus.COMPLETED if result.terminated else SimulationStatus.RUNNING
        self._append_resource_history(session)
        self._persist_step(session, result)
        payload = self._format_step_result(session, result)
        await self._broadcast("oversight_alert", {"agent": agent_type.value, "result": payload})
        return payload

    def get_patients(self, critical_only: bool = False) -> list[dict[str, Any]]:
        session = self.get_latest_episode()
        if session is None:
            return []
        patients = [
            self._patient_snapshot(patient)
            for patient in session.orchestrator.state.patients
            if not critical_only or patient.status.value == "CRITICAL"
        ]
        return patients

    def get_patient(self, patient_id: str) -> dict[str, Any]:
        session = self.get_latest_episode()
        if session is None:
            raise HTTPException(404, "No episodes available")
        for patient in session.orchestrator.state.patients:
            if patient.id == patient_id:
                return self._patient_snapshot(patient)
        raise HTTPException(404, f"Patient {patient_id} not found")

    def get_patient_timeline(self) -> list[dict[str, Any]]:
        session = self.get_latest_episode()
        if session is None:
            return []
        timeline = []
        for patient in session.orchestrator.state.patients:
            for event in patient.history:
                timeline.append(
                    {
                        "patient_id": patient.id,
                        "timestamp": event.timestamp.isoformat(),
                        "event_type": event.event_type,
                        "description": event.description,
                    }
                )
        return sorted(timeline, key=lambda item: item["timestamp"])

    def get_reward_curve(self) -> dict[str, Any]:
        with get_session() as db:
            rows = db.scalars(select(EpisodeRecord).order_by(EpisodeRecord.episode_num)).all()
        if rows:
            return {
                "curve": [
                    RewardCurvePoint(
                        episode=row.episode_num,
                        reward=row.total_reward,
                        total_reward=row.total_reward,
                        is_trained=row.is_trained,
                    ).model_dump()
                    for row in rows
                ]
            }

        session = self.get_latest_episode()
        if session is None:
            return {"curve": []}
        cumulative = 0.0
        curve = []
        for step in session.orchestrator.step_history:
            cumulative += step["breakdown"]["total"]
            curve.append(
                RewardCurvePoint(
                    step=step["step"],
                    reward=step["breakdown"]["total"],
                    cumulative=round(cumulative, 4),
                ).model_dump()
            )
        return {"curve": curve}

    def get_episode_metrics(self, episode_id: str) -> dict[str, Any]:
        session = self.get_episode(episode_id)
        return {
            "episode_id": episode_id,
            "reward_breakdown": [step["breakdown"] for step in session.orchestrator.step_history],
        }

    def get_comparison_metrics(self) -> dict[str, Any]:
        with get_session() as db:
            baseline = db.scalars(select(EpisodeRecord).where(EpisodeRecord.is_trained.is_(False))).all()
            trained = db.scalars(select(EpisodeRecord).where(EpisodeRecord.is_trained.is_(True))).all()

        def mean(values: list[float]) -> float:
            return round(sum(values) / len(values), 4) if values else 0.0

        baseline_rewards = [row.total_reward for row in baseline]
        trained_rewards = [row.total_reward for row in trained]
        baseline_survival = [row.survival_rate for row in baseline]
        trained_survival = [row.survival_rate for row in trained]
        payload = MetricsComparison(
            baseline_mean_reward=mean(baseline_rewards),
            trained_mean_reward=mean(trained_rewards),
            reward_delta=round(mean(trained_rewards) - mean(baseline_rewards), 4),
            baseline_mean_survival=mean(baseline_survival),
            trained_mean_survival=mean(trained_survival),
            survival_delta=round(mean(trained_survival) - mean(baseline_survival), 4),
            episode_counts={"baseline": len(baseline), "trained": len(trained)},
        )
        return payload.model_dump()

    def get_agent_metrics(self) -> list[dict[str, Any]]:
        session = self.get_latest_episode()
        if session is None:
            return []
        return self._agent_snapshots(session)

    def get_resource_metrics(self) -> dict[str, Any]:
        session = self.get_latest_episode()
        if session is None:
            return {"points": []}
        points = [
            ResourcePoint(
                step=point["step"],
                icu_occupancy=point["icu_occupancy"],
                staff_ratio=point["staff_ratio"],
                pharmacy_stock=point["pharmacy_stock"],
                equipment_status=point["equipment_status"],
            ).model_dump()
            for point in session.resource_history
        ]
        return {"points": points}

    async def collect_training_data(self, config: TrainingConfig) -> dict[str, Any]:
        collector = TrajectoryCollector(
            output_dir=config.output_dir,
            mock_llm=config.mock_llm,
        )
        converted_crisis_types = None
        if config.crisis_types:
            from triage.env.state import CrisisType

            converted_crisis_types = [CrisisType(item.value) for item in config.crisis_types]
        self._last_trajectories = await collector.collect(
            n_episodes=config.n_episodes,
            crisis_types=converted_crisis_types,
            difficulty=config.difficulty,
        )
        self._last_output_dir = config.output_dir
        self._sync_strategy_memory(Path(config.output_dir) / "strategy_memory.json")
        return collector.collector.get_summary()

    def label_preferences(self, output_path: str, min_delta: float = 20.0) -> dict[str, Any]:
        if not self._last_trajectories:
            raise HTTPException(400, "No collected trajectories available")
        labeler = PreferenceLabeler()
        pairs = labeler.label_trajectories(self._last_trajectories, min_delta=min_delta)
        dataset = labeler.export_as_hf_dataset(pairs, output_path)
        self._last_pairs = [pair.__dict__ for pair in pairs]
        return {"pairs": len(pairs), "dataset": dataset}

    async def start_dpo_training(self, config: TrainingConfig) -> dict[str, Any]:
        if self._training_task and not self._training_task.done():
            raise HTTPException(400, "Training already in progress")

        self._training_status = TrainingStatus(
            phase="collecting",
            progress=0.0,
            current_episode=0,
            total_episodes=config.n_episodes,
        )

        async def _run() -> None:
            try:
                summary = await self.collect_training_data(config)
                self._training_status = TrainingStatus(
                    phase="labeling",
                    progress=0.45,
                    total_episodes=config.n_episodes,
                    metrics={"collection": summary},
                )
                label_path = str(Path(config.output_dir) / "preference_dataset.json")
                label_result = self.label_preferences(label_path)
                if config.external_dataset_path:
                    adapter = DatasetAdapter()
                    real_records = adapter.load(config.external_dataset_path)
                    real_pairs = adapter.records_to_pairs(real_records)
                    synthetic_pairs = label_result["dataset"].get("train", [])
                    mixed_pairs = adapter.mix_pairs(
                        real_pairs,
                        synthetic_pairs,
                        real_fraction=config.real_data_fraction,
                    )
                    label_result["dataset"] = adapter.export_preference_dataset(mixed_pairs, label_path)
                    label_result["real_pairs"] = len(real_pairs)
                    label_result["mixed_pairs"] = len(mixed_pairs)
                self._training_status = TrainingStatus(
                    phase="training",
                    progress=0.7,
                    total_episodes=config.n_episodes,
                    metrics={"collection": summary, "labeling": label_result},
                )
                trainer = TRIAGEDPOTrainer(
                    DPOTrainingConfig(
                        model_name=config.model_name,
                        output_dir=str(Path(config.output_dir) / "model"),
                        data_dir=config.output_dir,
                        learning_rate=config.learning_rate,
                        num_epochs=config.num_epochs,
                        mock_mode=config.mock_training,
                        preset=config.model_preset,
                    )
                )
                train_metrics = await trainer.train()
                report = generate_training_report(
                    artifacts_dir=Path(config.output_dir),
                    output_dir=Path(config.output_dir) / "report",
                    training_metrics=train_metrics,
                )
                self._training_status = TrainingStatus(
                    phase="completed",
                    progress=1.0,
                    total_episodes=config.n_episodes,
                    metrics={
                        "collection": summary,
                        "labeling": label_result,
                        "training": train_metrics,
                        "report": report,
                    },
                )
            except Exception as exc:
                logger.exception("Training pipeline failed")
                self._training_status = TrainingStatus(
                    phase="error",
                    progress=0.0,
                    error=str(exc),
                )

        self._training_task = asyncio.create_task(_run())
        return {"status": "training_started"}

    def get_training_status(self) -> dict[str, Any]:
        # Search for training_live.json in all possible locations
        _here = Path(__file__).resolve().parent  # triage/api/
        candidates = [
            Path("data/training_live.json"),                    # CWD-relative (most reliable)
            _here.parent.parent / "data" / "training_live.json",  # triage-backend/data/
            _here.parent / "data" / "training_live.json",
        ]
        live_path: Path | None = None
        for c in candidates:
            if c.exists():
                live_path = c
                break

        if live_path is not None:
            try:
                import time as _time
                age = _time.time() - live_path.stat().st_mtime
                if age < 1800:  # < 30 minutes — show completed training results
                    raw = json.loads(live_path.read_text())
                    phase = raw.get("phase", "training")
                    progress = raw.get("progress", 0.0)
                    step = raw.get("step", 0)
                    total_steps = raw.get("total_steps", 1)
                    epoch = raw.get("epoch", 0)
                    total_epochs = raw.get("total_epochs", 1)
                    loss = raw.get("loss")
                    avg_loss = raw.get("avg_loss")
                    eta = raw.get("eta_seconds", 0)
                    vram_used = raw.get("vram_used_gb", 0)
                    vram_total = raw.get("vram_total_gb", 4)
                    gpu_pct = raw.get("gpu_pct", 0)
                    elapsed = raw.get("elapsed_seconds", 0)
                    return {
                        "phase": phase,
                        "progress": progress,
                        "current_episode": step,
                        "total_episodes": total_steps,
                        "metrics": {
                            "step": step,
                            "total_steps": total_steps,
                            "epoch": epoch,
                            "total_epochs": total_epochs,
                            "loss": loss,
                            "avg_loss": avg_loss,
                            "eta_seconds": eta,
                            "eta_minutes": round(eta / 60, 1) if eta else 0,
                            "elapsed_seconds": elapsed,
                            "vram_used_gb": vram_used,
                            "vram_total_gb": vram_total,
                            "gpu_pct": gpu_pct,
                            "model": raw.get("model", "Qwen2.5-0.5B"),
                            "train_samples": raw.get("train_samples", 0),
                        },
                        "error": None,
                    }
            except Exception:
                pass  # Fall through to internal status

        return self._training_status.model_dump()

    def get_safety_blocks(self) -> list[dict[str, Any]]:
        session = self.get_latest_episode()
        if not session:
            return []
        return [b.to_dict() for b in session.orchestrator.state.safety_blocks]

    def get_safety_stats(self) -> dict[str, Any]:
        session = self.get_latest_episode()
        if not session:
            return {}
        return session.orchestrator.constitution.get_constitution_report()

    def get_strategy_memory(self) -> dict[str, Any]:
        memory = StrategyMemory()
        
        result = {}
        for agent_enum in AgentType:
            agent_type = agent_enum.value
            summary = memory.summarize(agent_type)
            top_3 = memory.get_best_lessons(agent_type, limit=3)
            result[agent_type] = {
                "count": summary["count"],
                "avg_reward_delta": summary["avg_reward_delta"],
                "top_crisis_type": summary["top_crisis_type"],
                "top_lessons": top_3
            }
        return result

    async def register_websocket(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._websockets.append(websocket)
        session = self.get_latest_episode()
        if session is not None:
            await websocket.send_json(WSMessage(type="state_update", data=self.build_state_payload(session)).model_dump())

    def unregister_websocket(self, websocket: WebSocket) -> None:
        if websocket in self._websockets:
            self._websockets.remove(websocket)

    async def handle_ws_command(self, websocket: WebSocket, command: str, params: dict[str, Any]) -> None:
        if command == "start":
            result = await self.start_episode(EpisodeConfig(**params))
            await websocket.send_json(WSMessage(type="episode_started", data=result).model_dump())
            return
        if command == "step":
            session = self.get_latest_episode()
            if session is None:
                raise HTTPException(404, "No episodes available")
            result = await self.step_episode(session.episode_id)
            await websocket.send_json(WSMessage(type="step_complete", data=result).model_dump())
            return
        if command == "run":
            session = self.get_latest_episode()
            if session is None:
                raise HTTPException(404, "No episodes available")
            result = await self.run_episode(session.episode_id, EpisodeRunRequest(**params))
            await websocket.send_json(WSMessage(type="episode_end", data=result).model_dump())
            return
        if command == "state":
            session = self.get_latest_episode()
            data = self.build_state_payload(session) if session else {}
            await websocket.send_json(WSMessage(type="state_update", data=data).model_dump())
            return
        if command == "stop":
            result = await self.stop_latest_episode()
            await websocket.send_json(WSMessage(type="episode_stopped", data=result).model_dump())
            return
        raise HTTPException(400, f"Unknown command: {command}")

    async def _broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        if not self._websockets:
            return
        message = WSMessage(type=event_type, data=data).model_dump()
        stale: list[WebSocket] = []
        for websocket in self._websockets:
            try:
                await websocket.send_json(message)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.unregister_websocket(websocket)

    def _agent_snapshots(self, session: EpisodeSession) -> list[dict[str, Any]]:
        state = session.orchestrator.state
        snapshots = []
        for agent in session.orchestrator.agents.values():
            agent_state = state.agent_states.get(agent.agent_type)
            stats = agent.get_stats()
            snapshots.append(
                {
                    "agent_type": agent.agent_type.value,
                    "role": stats["role"],
                    "actions_taken": stats["actions_taken"],
                    "total_tokens": stats["total_tokens"],
                    "last_action": agent_state.current_action if agent_state else None,
                    "is_active": agent_state.is_active if agent_state else True,
                    "inbox_size": stats["inbox_size"],
                    "messages_sent": stats["messages_sent"],
                    "tool_usage": stats.get("tool_usage", {}),
                }
            )
        return snapshots

    def _patient_snapshot(self, patient: Any) -> dict[str, Any]:
        return patient.to_dict()

    def _resource_snapshot(self, state: Any) -> dict[str, Any]:
        resources = state.resources
        return {
            "icu_beds_total": resources.icu_beds_total,
            "icu_beds_used": resources.icu_beds_occupied,
            "ventilators_total": resources.ventilators_total,
            "ventilators_used": resources.ventilators_in_use,
            "staff_ratio": resources.staff_ratio,
            "pharmacy_stock": resources.pharmacy_stock,
            "equipment_status": resources.equipment_status,
            "it_uptime": resources.it_uptime,
        }

    def _metrics_snapshot(self, session: EpisodeSession) -> dict[str, Any]:
        state = session.orchestrator.state
        return {
            "survival_rate": state.survival_rate,
            "deceased_count": state.deceased_count,
            "discharged_count": state.discharged_count,
            "critical_count": state.critical_count,
            "alive_count": state.alive_count,
            "icu_occupancy": state.icu_occupancy,
            "total_reward": session.orchestrator.total_reward,
            "violations_caught": state.violations_caught,
            "violations_injected": state.violations_injected,
            "compliance_rate": (
                state.violations_caught / max(state.violations_injected, 1)
                if state.violations_injected
                else 1.0
            ),
        }

    def _append_resource_history(self, session: EpisodeSession) -> None:
        state = session.orchestrator.state
        session.resource_history.append(
            {
                "step": state.step_count,
                "icu_occupancy": state.icu_occupancy,
                "staff_ratio": state.resources.staff_ratio,
                "pharmacy_stock": state.resources.pharmacy_stock,
                "equipment_status": state.resources.equipment_status,
            }
        )

    def _format_step_result(self, session: EpisodeSession, result: OrchestratorStepResult) -> dict[str, Any]:
        return {
            "episode_id": session.episode_id,
            "step": result.step,
            "action": result.action,
            "reward": result.reward,
            "breakdown": result.breakdown,
            "terminated": result.terminated,
            "drift_events": result.drift_events,
        }

    def _persist_episode_snapshot(self, session: EpisodeSession) -> None:
        state = session.orchestrator.state
        with get_session() as db:
            record = db.get(EpisodeRecord, session.episode_id)
            if record is None:
                record = EpisodeRecord(
                    id=session.episode_id,
                    episode_num=state.episode,
                    crisis_type=state.crisis.type.value,
                )
                db.add(record)
            record.crisis_type = state.crisis.type.value
            record.episode_num = state.episode
            record.total_steps = state.step_count
            record.total_reward = session.orchestrator.total_reward
            record.survival_rate = state.survival_rate
            record.compliance_score = (
                state.violations_caught / max(state.violations_injected, 1)
                if state.violations_injected
                else 1.0
            )
            record.is_trained = session.config.is_trained
            record.config = session.config.model_dump()
            if session.status == SimulationStatus.COMPLETED:
                record.ended_at = datetime.now(timezone.utc)

    def _persist_step(self, session: EpisodeSession, result: OrchestratorStepResult) -> None:
        self._persist_episode_snapshot(session)
        state = session.orchestrator.state
        with get_session() as db:
            reward_record = RewardRecord(
                id=f"{session.episode_id}:{result.step}",
                episode_id=session.episode_id,
                step=result.step,
                total_reward=result.breakdown["total"],
                survival_score=result.breakdown.get("survival", 0.0),
                compliance_score=result.breakdown.get("compliance", 0.0),
                coordination_score=result.breakdown.get("coordination", 0.0),
                oversight_score=result.breakdown.get("oversight", 0.0),
                depth_score=result.breakdown.get("depth", 0.0),
                adaptation_score=result.breakdown.get("adaptation", 0.0),
                expert_score=result.breakdown.get("expert_alignment", 0.0),
                penalties=result.breakdown.get("penalties", 0.0),
                terminal_bonus=result.breakdown.get("terminal_bonus", 0.0),
            )
            db.merge(reward_record)
            for patient in state.patients:
                record = PatientRecord(
                    id=f"{session.episode_id}:{patient.id}",
                    episode_id=session.episode_id,
                    patient_id=patient.id,
                    name=patient.name,
                    age=patient.age,
                    condition=patient.condition,
                    final_status=patient.status.value,
                    triage_score=patient.triage_score,
                    admitted_at=patient.admitted_at,
                    discharged_at=patient.last_updated if patient.status.value == "DISCHARGED" else None,
                    insurance_verified=patient.insurance_verified,
                    treatment_timeline=[
                        {
                            "timestamp": event.timestamp.isoformat(),
                            "event_type": event.event_type,
                            "description": event.description,
                        }
                        for event in patient.history
                    ],
                )
                db.merge(record)

            for message in session.orchestrator.bus.history:
                if message.id in session.persisted_message_ids:
                    continue
                db.merge(
                    AgentMessageRecord(
                        id=message.id,
                        episode_id=session.episode_id,
                        from_agent=message.from_agent.value if hasattr(message.from_agent, "value") else str(message.from_agent),
                        to_agent=message.to_agent.value if hasattr(message.to_agent, "value") else str(message.to_agent),
                        content=message.content,
                        msg_type=message.msg_type.value,
                        token_count=message.token_count,
                        timestamp=message.timestamp,
                        patient_id=message.patient_id,
                    )
                )
                session.persisted_message_ids.add(message.id)

    def _clear_episode_records(self, episode_id: str) -> None:
        with get_session() as db:
            db.execute(delete(RewardRecord).where(RewardRecord.episode_id == episode_id))
            db.execute(delete(AgentMessageRecord).where(AgentMessageRecord.episode_id == episode_id))
            db.execute(delete(PatientRecord).where(PatientRecord.episode_id == episode_id))
            record = db.get(EpisodeRecord, episode_id)
            if record is not None:
                record.ended_at = None
                record.total_steps = 0
                record.total_reward = 0.0
                record.survival_rate = 0.0
                record.compliance_score = 0.0

    def _sync_strategy_memory(self, storage_path: Path) -> None:
        memory = StrategyMemory(storage_path=str(storage_path))
        with get_session() as db:
            for key, strategies in memory.get_all().items():
                for strategy in strategies:
                    record = StrategyLessonRecord(
                        id=strategy["id"],
                        episode_observed=strategy["episode"],
                        pattern=strategy["description"],
                        context=json.dumps(strategy.get("context", {}), sort_keys=True),
                        correction=strategy["description"],
                        confidence=float(strategy.get("success_rate", 0.0)),
                        outcome_delta=float(strategy.get("reward", 0.0)),
                        agent_type=strategy.get("agent_type"),
                        crisis_type=strategy.get("crisis_type"),
                        times_applied=int(strategy.get("times_used", 0)),
                        times_successful=int(
                            round(strategy.get("times_used", 0) * strategy.get("success_rate", 0.0))
                        ),
                    )
                    db.merge(record)


backend_service = BackendService()
