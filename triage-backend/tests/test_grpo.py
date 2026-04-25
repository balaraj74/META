from __future__ import annotations

import pytest


def test_env_adapter_reset() -> None:
    from triage.env.grpo_env_adapter import HospitalGRPOEnvironment

    env = HospitalGRPOEnvironment()
    observation = env.reset(crisis_type="mass_casualty")
    assert observation
    assert "crisis" in observation.lower()
    assert "patients" in observation.lower()


def test_env_adapter_tool_triage() -> None:
    from triage.env.grpo_env_adapter import HospitalGRPOEnvironment

    env = HospitalGRPOEnvironment()
    env.reset(crisis_type="mass_casualty")
    patient_id = env.current_state.patients[0].id
    observation = env.triage_patient(patient_id, 9, "ICU")
    assert patient_id in observation
    assert "acuity 9" in observation


def test_env_adapter_terminal_reward() -> None:
    from triage.env.grpo_env_adapter import HospitalGRPOEnvironment

    env = HospitalGRPOEnvironment()
    env.reset(crisis_type="outbreak")
    patient_id = env.current_state.patients[0].id
    for _ in range(5):
        env.triage_patient(patient_id, 8, "ER")
    reward = env._get_terminal_reward()
    assert isinstance(reward, float)
    assert -1.0 <= reward <= 1.0


def test_format_observation_token_count() -> None:
    transformers = pytest.importorskip("transformers")
    from triage.env.grpo_env_adapter import HospitalGRPOEnvironment

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-27B",
            local_files_only=True,
        )
    except Exception as exc:
        pytest.skip(f"Qwen3 tokenizer is not available locally: {exc}")

    env = HospitalGRPOEnvironment()
    env.reset(crisis_type="staff_shortage")
    observation = env._format_observation(env.current_state)
    assert len(tokenizer.encode(observation)) < 800


def test_crisis_dataset_shape() -> None:
    from scripts.build_grpo_dataset import build_crisis_prompt_dataset

    dataset = build_crisis_prompt_dataset()
    assert len(dataset) == 500
    assert dataset.column_names == ["crisis_type", "difficulty", "prompt"]
