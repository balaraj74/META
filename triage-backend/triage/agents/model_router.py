"""Model loading and routing for TRIAGE agents.

Clinical agents use the II-Medical-8B backbone; operations agents use
Qwen3-4B-Instruct. The router is a process-wide singleton so heavy models are
loaded once and shared by all agent instances.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - python-dotenv is optional at import time
    load_dotenv = None

logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env", override=False)


class ModelTier(str, Enum):
    CLINICAL = "clinical"
    OPERATIONS = "operations"
    MOCK = "mock"


AGENT_MODEL_TIER: dict[str, ModelTier] = {
    "ambulance_dispatch": ModelTier.OPERATIONS,
    "cmo_oversight": ModelTier.CLINICAL,
    "er_triage": ModelTier.CLINICAL,
    "infection_control": ModelTier.CLINICAL,
    "icu_management": ModelTier.CLINICAL,
    "pharmacy": ModelTier.CLINICAL,
    "hr_rostering": ModelTier.OPERATIONS,
    "it_systems": ModelTier.OPERATIONS,
    "blood_bank": ModelTier.CLINICAL,
    "ethics_committee": ModelTier.CLINICAL,
}

MODEL_IDS: dict[ModelTier, str] = {
    ModelTier.CLINICAL: os.getenv("CLINICAL_MODEL", "Intelligent-Internet/II-Medical-8B"),
    ModelTier.OPERATIONS: os.getenv("OPERATIONS_MODEL", "Qwen/Qwen3-4B-Instruct"),
}

LOCAL_MODEL_PATHS: dict[ModelTier, str] = {
    ModelTier.CLINICAL: os.getenv("CLINICAL_MODEL_PATH", "./models/ii-medical-8b"),
    ModelTier.OPERATIONS: os.getenv("OPERATIONS_MODEL_PATH", "./models/qwen3-4b"),
}

GRPO_ADAPTER_PATHS: dict[ModelTier, str | None] = {
    ModelTier.CLINICAL: os.getenv("CLINICAL_GRPO_ADAPTER"),
    ModelTier.OPERATIONS: os.getenv("OPERATIONS_GRPO_ADAPTER"),
}


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    tier: ModelTier
    model_id: str
    has_grpo_adapter: bool = False


class ModelRouter:
    """Singleton router for HF, Ollama, and mock model modes."""

    _instance: ModelRouter | None = None
    _models: dict[ModelTier, LoadedModel] = {}
    _mode: str = "mock"
    _initialized: bool = False

    def __init__(self) -> None:
        raise RuntimeError("Use ModelRouter.get_instance()")

    @classmethod
    def get_instance(cls) -> ModelRouter:
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._load_in_4bit = True
            cls._instance._max_seq_length = 512
            cls._instance._device_map = "auto"
        return cls._instance

    @classmethod
    def initialize(
        cls,
        mode: str = "auto",
        load_in_4bit: bool = True,
        max_seq_length: int = 512,
        device_map: str = "auto",
        force: bool = False,
    ) -> str:
        """Initialize the selected backend and return the actual mode."""
        requested = mode.lower().strip()
        if requested not in {"auto", "hf", "ollama", "mock"}:
            raise ValueError(f"Unsupported LLM mode: {mode}")

        if cls._initialized and not force:
            if requested == "auto" or requested == cls._mode:
                return cls._mode
            if requested == "hf" and cls._mode == "hf":
                return cls._mode

        inst = cls.get_instance()
        inst._load_in_4bit = load_in_4bit
        inst._max_seq_length = max_seq_length
        inst._device_map = device_map

        if requested == "mock":
            cls._mode = "mock"
            cls._initialized = True
            logger.info("ModelRouter initialized in mock mode")
            return cls._mode

        if requested in {"auto", "hf"}:
            try:
                inst._load_hf_models()
                cls._mode = "hf"
                cls._initialized = True
                logger.info("ModelRouter initialized in HuggingFace mode")
                return cls._mode
            except Exception as exc:
                logger.warning("ModelRouter HF load failed: %s", exc)
                if requested == "hf":
                    cls._initialized = True
                    raise

        if requested in {"auto", "ollama"}:
            try:
                inst._verify_ollama()
                cls._mode = "ollama"
                cls._initialized = True
                logger.info("ModelRouter initialized in Ollama mode")
                return cls._mode
            except Exception as exc:
                logger.warning("ModelRouter Ollama check failed: %s", exc)
                if requested == "ollama":
                    cls._initialized = True
                    raise

        cls._mode = "mock"
        cls._initialized = True
        logger.warning("ModelRouter falling back to mock mode")
        return cls._mode

    @classmethod
    def initialize_from_env(cls, force: bool = False) -> str:
        if os.getenv("PYTEST_CURRENT_TEST") and os.getenv("TRIAGE_FORCE_MODEL_STARTUP", "").lower() != "true":
            return cls.initialize(mode="mock", force=force)
        use_mock = os.getenv("USE_MOCK_LLM", os.getenv("MOCK_LLM", "true")).lower() == "true"
        mode = "mock" if use_mock else os.getenv("LLM_MODE", "auto")
        load_in_4bit = os.getenv("MODEL_LOAD_IN_4BIT", "true").lower() == "true"
        max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "512"))
        return cls.initialize(
            mode=mode,
            load_in_4bit=load_in_4bit,
            max_seq_length=max_seq_length,
            force=force,
        )

    def _load_hf_models(self) -> None:
        try:
            from unsloth import FastLanguageModel

            use_unsloth = True
        except ImportError:
            FastLanguageModel = None
            use_unsloth = False
            logger.warning("Unsloth not available; using transformers for model loading")

        for tier in (ModelTier.CLINICAL, ModelTier.OPERATIONS):
            if tier in self._models:
                continue

            local_path = Path(LOCAL_MODEL_PATHS[tier])
            model_src = str(local_path) if local_path.is_dir() else MODEL_IDS[tier]
            token = os.getenv("HF_TOKEN") or None
            has_adapter = False

            logger.info("Loading %s model from %s", tier.value, model_src)
            if use_unsloth and FastLanguageModel is not None:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_src,
                    max_seq_length=self._max_seq_length,
                    dtype=None,
                    load_in_4bit=self._load_in_4bit,
                    token=token,
                )
                adapter_path = GRPO_ADAPTER_PATHS[tier]
                if adapter_path and Path(adapter_path).is_dir():
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(model, adapter_path)
                    has_adapter = True
                    logger.info("Loaded %s GRPO adapter from %s", tier.value, adapter_path)
                FastLanguageModel.for_inference(model)
            else:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    model_src,
                    token=token,
                    trust_remote_code=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_src,
                    torch_dtype=torch.float16,
                    device_map=self._device_map,
                    load_in_4bit=self._load_in_4bit,
                    token=token,
                    trust_remote_code=True,
                )
                adapter_path = GRPO_ADAPTER_PATHS[tier]
                if adapter_path and Path(adapter_path).is_dir():
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(model, adapter_path)
                    has_adapter = True
                    logger.info("Loaded %s GRPO adapter from %s", tier.value, adapter_path)

            self._models[tier] = LoadedModel(
                model=model,
                tokenizer=tokenizer,
                tier=tier,
                model_id=model_src,
                has_grpo_adapter=has_adapter,
            )
            logger.info("%s model ready", tier.value)

    def _verify_ollama(self) -> None:
        base_url = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))
        base_url = base_url.removesuffix("/api/generate").removesuffix("/api/chat").rstrip("/")
        response = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        response.raise_for_status()
        tags = [item.get("name", "") for item in response.json().get("models", [])]
        logger.info("Ollama models available: %s", tags)

    def get_model_for_agent(self, agent_type: Any) -> LoadedModel | None:
        if self.mode != "hf":
            return None
        key = getattr(agent_type, "value", str(agent_type))
        tier = AGENT_MODEL_TIER.get(key, ModelTier.CLINICAL)
        return self._models.get(tier)

    def get_tier_for_agent(self, agent_type: Any) -> ModelTier:
        key = getattr(agent_type, "value", str(agent_type))
        return AGENT_MODEL_TIER.get(key, ModelTier.CLINICAL)

    @property
    def initialized(self) -> bool:
        return type(self)._initialized

    @property
    def mode(self) -> str:
        return type(self)._mode

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "initialized": self.initialized,
            "models_loaded": [
                {
                    "tier": tier.value,
                    "model_id": loaded.model_id,
                    "has_grpo_adapter": loaded.has_grpo_adapter,
                }
                for tier, loaded in self._models.items()
            ],
            "agent_routing": {agent: tier.value for agent, tier in AGENT_MODEL_TIER.items()},
            "model_ids": {tier.value: model_id for tier, model_id in MODEL_IDS.items()},
            "local_model_paths": {tier.value: path for tier, path in LOCAL_MODEL_PATHS.items()},
        }
