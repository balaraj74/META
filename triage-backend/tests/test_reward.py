from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_reward_model_exposes_prompt_and_legacy_keys() -> None:
    from triage.env.hospital_env import HospitalEnv
    from triage.rewards.reward_model import RewardModel

    env = HospitalEnv(seed=3, max_steps=10)
    await env.reset()
    breakdown = RewardModel().compute(env.state, [], [])
    payload = breakdown.to_dict()
    for key in ["survival", "compliance", "coordination", "oversight", "depth", "adaptation", "expert_alignment"]:
        assert key in payload
    for key in ["patient_outcomes", "communication_quality", "drift_adaptation", "token_economy"]:
        assert key in payload


def test_prompt_aligned_reward_components_compute() -> None:
    from triage.env.hospital_env import HospitalEnv
    from triage.reward.components import (
        AdaptationReward,
        ComplianceReward,
        CoordinationReward,
        DepthReward,
        ExpertAlignmentReward,
        OversightReward,
        SurvivalReward,
    )

    env = HospitalEnv(seed=5, max_steps=10)
    import asyncio

    asyncio.run(env.reset())
    state = env.state
    assert isinstance(SurvivalReward().compute(state), float)
    assert isinstance(ComplianceReward().compute(state), float)
    assert isinstance(CoordinationReward().compute(state), float)
    assert isinstance(OversightReward().compute(state), float)
    assert isinstance(DepthReward().compute([]), float)
    assert isinstance(AdaptationReward().compute(state, []), float)
    assert isinstance(ExpertAlignmentReward().compute(state), float)


@pytest.mark.asyncio
async def test_reward_weights_shift_with_expert_signals_and_depth_proxy() -> None:
    from triage.env.hospital_env import HospitalEnv
    from triage.env.state import ActionType, AgentAction, AgentType
    from triage.rewards.reward_model import RewardModel

    env = HospitalEnv(seed=11, max_steps=10)
    await env.reset()
    env.state.expert_signals = {
        "cost_weight": 0.72,
        "quality_weight": 0.18,
        "speed_weight": 0.10,
    }
    actions = [
        AgentAction(
            agent_type=AgentType.CMO_OVERSIGHT,
            action_type=ActionType.OVERRIDE_DECISION,
            reasoning="Escalate ICU review, validate queue priority, and document why the override is justified.",
        )
    ]
    breakdown = RewardModel().compute(env.state, actions, [{"type": "contract_drift"}])
    assert breakdown.depth > 0.0
    assert breakdown.weights["depth"] < RewardModel.DEFAULT_WEIGHTS["depth"]
    assert breakdown.details["expert_profile"]["dominant_signal"] == "cost"


def test_schema_drift_supports_contract_and_regulatory_domains() -> None:
    from triage.env.hospital_env import HospitalEnv
    from triage.env.schema_drift import SchemaDrift

    env = HospitalEnv(seed=7, max_steps=10)
    asyncio.run(env.reset())
    drift = SchemaDrift(seed=7)

    contract_event = drift._apply_single_drift("contract_drift", env.state)
    regulatory_event = drift._apply_single_drift("regulatory_drift", env.state)

    assert contract_event is not None
    assert regulatory_event is not None
    assert contract_event["type"] == "contract_drift"
    assert regulatory_event["type"] == "regulatory_drift"
    assert env.state.contract_constraints["insurance_portal"]["schema_version"].startswith("v")
    assert env.state.regulatory_constraints


@pytest.mark.asyncio
async def test_enterprise_registry_uses_standalone_hris_insurance_and_it_modules() -> None:
    from triage.env.enterprise_registry import EnterpriseAppRegistry
    from triage.env.hospital_env import HospitalEnv
    from triage.env.state import AgentType

    env = HospitalEnv(seed=13, max_steps=10)
    await env.reset()
    registry = EnterpriseAppRegistry()
    patient_id = env.state.patients[0].id

    roster = registry.execute_tool("get_roster", {}, env.state, AgentType.HR_ROSTERING)
    insurance = registry.execute_tool(
        "verify_insurance",
        {"patient_id": patient_id},
        env.state,
        AgentType.IT_SYSTEMS,
    )
    it_status = registry.execute_tool("get_equipment_status", {}, env.state, AgentType.IT_SYSTEMS)

    assert roster["status"] == "approved"
    assert insurance["status"] == "approved"
    assert it_status["status"] == "approved"
