#!/usr/bin/env python3
"""Build the HuggingFace Dataset used to seed live GRPO episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from datasets import Dataset
except ModuleNotFoundError:
    class Dataset(list):
        """Small fallback for lightweight tests when HF datasets is not installed."""

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

        @property
        def column_names(self):
            return list(self[0].keys()) if self else []

        def save_to_disk(self, output_dir: str) -> None:
            raise RuntimeError("Install `datasets` to save Arrow datasets to disk")

CRISIS_TYPES = ["mass_casualty", "outbreak", "equipment_failure", "staff_shortage"]
DIFFICULTY_TIERS = [1, 2, 3, 4, 5]
DEFAULT_OUTPUT_DIR = Path("data/grpo_crisis_prompts")

ACTION_LIST = "\n".join(
    [
        "- triage_patient(patient_id, acuity_score, assigned_ward)",
        "- transfer_to_icu(patient_id, reason)",
        "- order_medication(patient_id, drug_name, dose_mg, reason)",
        "- request_blood(patient_id, blood_type, units)",
        "- escalate_to_cmo(patient_id, urgency, summary)",
        "- discharge_patient(patient_id, discharge_notes)",
        "- allocate_equipment(equipment_type, patient_id)",
        "- activate_protocol(protocol_name, justification)",
    ]
)

ROLE_ASSIGNMENT = """Agent roles:
- ER/triage acts quickly without extended thinking.
- Pharmacy acts quickly and checks medication safety.
- ICU coordinates scarce critical-care resources.
- CMO and Ethics use <think> reasoning for escalation, rationing, and oversight.
Use the live environment tools; do not invent patient IDs."""


def build_crisis_prompt_dataset(output_dir: str | Path | None = None) -> Dataset:
    rows = []
    for crisis_type in CRISIS_TYPES:
        for difficulty in DIFFICULTY_TIERS:
            for scenario_idx in range(25):
                rows.append(
                    {
                        "crisis_type": crisis_type,
                        "difficulty": difficulty,
                        "prompt": _prompt(crisis_type, difficulty, scenario_idx),
                    }
                )

    dataset = Dataset.from_list(rows)
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
    return dataset


def _prompt(crisis_type: str, difficulty: int, scenario_idx: int) -> str:
    severity = ["controlled", "strained", "serious", "critical", "catastrophic"][
        difficulty - 1
    ]
    incoming = 8 + difficulty * 6 + (scenario_idx % 5)
    icu_free = max(1, 12 - difficulty * 2)
    vents = max(1, 10 - difficulty)
    o_pos = max(4, 24 - difficulty * 3)
    o_neg = max(2, 12 - difficulty * 2)
    return f"""You are coordinating a TRIAGE hospital crisis episode.

Crisis: {crisis_type}
Difficulty tier: {difficulty}/5 ({severity})
Crisis description: {incoming} incoming or active patients are expected, with limited staff attention and rapidly changing acuity.
Initial hospital state: ICU beds free {icu_free}, ventilators free {vents}, blood O+ {o_pos}, blood O- {o_neg}. Safety policy enforcement is active.

Available actions:
{ACTION_LIST}

{ROLE_ASSIGNMENT}

Start the episode by inspecting the live observation and choosing clinically justified tool calls."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GRPO crisis prompt Dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    dataset = build_crisis_prompt_dataset(args.output)
    print(f"Saved {len(dataset)} prompts to {args.output}")


if __name__ == "__main__":
    main()
