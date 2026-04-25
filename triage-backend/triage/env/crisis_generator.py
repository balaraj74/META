"""
CrisisGenerator — procedurally generates crisis scenarios with patients,
resource constraints, and policy violations.

Each crisis comes with:
  - A typed Crisis object (mass casualty, outbreak, etc.)
  - A list of patients with conditions and deterioration rates
  - Drug inventory, staff roster, ICU configuration
  - Pre-seeded policy violations for the IT agent to detect
"""

from __future__ import annotations

import random
import uuid
from typing import Any

import numpy as np

from triage.env.state import (
    Crisis,
    CrisisType,
    Patient,
    PatientStatus,
    Policy,
    WardType,
)

# ─── Patient Name Pool ──────────────────────────────────────

_FIRST_NAMES = [
    "Marcus", "Priya", "Raj", "Elena", "Tomás", "Ayesha", "Chen",
    "Fatima", "James", "Yuki", "Olga", "Diego", "Amina", "Wei",
    "Sarah", "Kofi", "Leila", "Dmitri", "Isabella", "Kwame",
    "Nadia", "Haruto", "Chiara", "Hassan", "Maya", "Viktor",
    "Anika", "Omar", "Zara", "Pavel", "Isla", "Jin", "Lucia",
    "Ravi", "Hana", "André", "Mika", "Suki", "Emmanuel", "Talia",
]
_LAST_NAMES = [
    "Chen", "Patel", "Kim", "García", "Müller", "Santos", "Liu",
    "Johnson", "Nakamura", "Ali", "Singh", "Park", "Silva", "Taylor",
    "Petrov", "Ibrahim", "Sato", "Rodríguez", "Okafor", "Johansson",
    "Martinez", "Tanaka", "Ivanov", "Brown", "Costa", "Lee", "Wilson",
    "Nguyen", "Kumar", "Fischer", "Anderson", "Yamamoto", "Fernández",
    "Wang", "Jones", "Hernández", "López", "González", "Pérez", "Moreau",
]

# ─── Condition Libraries ────────────────────────────────────

_MASS_CASUALTY_CONDITIONS = [
    ("crush injury — thorax", 0.12, 8),
    ("blast lung — bilateral pneumothorax", 0.15, 9),
    ("traumatic amputation — left leg below knee", 0.10, 7),
    ("severe burns — 40% TBSA", 0.14, 9),
    ("penetrating abdominal trauma", 0.11, 8),
    ("traumatic brain injury — GCS 6", 0.13, 9),
    ("open fracture — femur with arterial bleed", 0.09, 7),
    ("spinal cord injury — C4 incomplete", 0.08, 8),
    ("smoke inhalation — respiratory distress", 0.10, 6),
    ("cardiac contusion — hemodynamically unstable", 0.12, 8),
    ("facial lacerations — minor", 0.02, 2),
    ("ankle sprain — walking wounded", 0.01, 1),
    ("anxiety attack — no physical injury", 0.01, 1),
    ("concussion — GCS 14", 0.03, 3),
    ("forearm fracture — closed", 0.02, 3),
]

_OUTBREAK_CONDITIONS = [
    ("respiratory failure — viral pneumonia", 0.11, 8),
    ("septic shock — bacterial meningitis", 0.13, 9),
    ("acute hepatitis — fulminant liver failure", 0.12, 9),
    ("hemorrhagic fever — DIC", 0.15, 10),
    ("encephalitis — altered mental status", 0.10, 8),
    ("severe dehydration — pediatric", 0.06, 5),
    ("high fever — awaiting culture", 0.03, 3),
    ("mild gastroenteritis — self-limiting", 0.01, 2),
    ("contact trace — asymptomatic exposure", 0.00, 1),
    ("suspected case — pending PCR", 0.02, 2),
]

_EQUIPMENT_CONDITIONS = [
    ("ventilator-dependent COPD exacerbation", 0.08, 7),
    ("post-op cardiac — needs monitoring", 0.06, 6),
    ("dialysis patient — missed session", 0.09, 7),
    ("insulin pump failure — DKA", 0.07, 6),
    ("pacemaker malfunction — bradycardia", 0.10, 8),
    ("ICU transfer blocked — equipment offline", 0.05, 5),
]

_STAFF_CONDITIONS = [
    ("routine surgery — delayed", 0.03, 3),
    ("pediatric asthma — moderate", 0.04, 4),
    ("elderly fall — hip fracture", 0.05, 5),
    ("chest pain — rule out MI", 0.07, 6),
    ("stroke symptoms — time-critical", 0.10, 8),
    ("labor complications — emergency C-section", 0.08, 7),
]

# ─── Drug Inventory Pool ────────────────────────────────────

_DRUG_INVENTORY = {
    "epinephrine": 50,
    "morphine": 30,
    "antibiotics_broad": 100,
    "blood_thinners": 40,
    "insulin": 60,
    "propofol": 20,
    "ketamine": 15,
    "norepinephrine": 25,
    "saline_iv": 200,
    "dextrose": 80,
    "naloxone": 10,
    "atropine": 12,
    "amiodarone": 8,
    "fentanyl": 10,
    "midazolam": 20,
}

# ─── Staff Roster Template ──────────────────────────────────

_BASE_STAFF_ROSTER = {
    "er_physicians": 4,
    "icu_physicians": 3,
    "surgeons": 2,
    "anesthesiologists": 2,
    "er_nurses": 8,
    "icu_nurses": 6,
    "ward_nurses": 12,
    "pharmacists": 2,
    "respiratory_therapists": 3,
    "lab_technicians": 4,
    "radiology_techs": 2,
    "social_workers": 1,
}


# ─── Default Policies ───────────────────────────────────────

def _default_policies(episode: int) -> dict[str, Policy]:
    return {
        "triage_protocol": Policy(
            id="POL-001",
            name="Mass Casualty Triage Protocol",
            version="3.1",
            rules=[
                "All incoming patients MUST receive triage score within 5 minutes",
                "Triage score 8-10: immediate ICU or OR referral",
                "Triage score 5-7: urgent care within 30 minutes",
                "Triage score 1-4: delayed care acceptable",
                "Re-triage every 15 minutes for waiting patients",
            ],
            effective_from=episode,
        ),
        "icu_admission": Policy(
            id="POL-002",
            name="ICU Admission Criteria",
            version="2.0",
            rules=[
                "ICU admission requires triage score >= 7 OR physician override",
                "Ventilator allocation prioritizes by triage score descending",
                "No more than 2 patients per nurse in ICU",
                "Overflow protocol activates at 90% ICU occupancy",
            ],
            effective_from=episode,
        ),
        "medication_safety": Policy(
            id="POL-003",
            name="Medication Safety Protocol",
            version="4.2",
            rules=[
                "Double-verification required for all controlled substances",
                "Drug interaction check mandatory before dispensing",
                "Maximum single dose limits must be enforced",
                "Patient allergy check before every medication order",
            ],
            effective_from=episode,
        ),
        "staff_fatigue": Policy(
            id="POL-004",
            name="Staff Fatigue Management",
            version="1.5",
            rules=[
                "No physician may work more than 16 consecutive hours",
                "Mandatory 30-minute break every 6 hours",
                "Fatigued staff must not perform critical procedures alone",
                "Shift handoff report is mandatory — no verbal-only handoffs",
            ],
            effective_from=episode,
        ),
        "data_privacy": Policy(
            id="POL-005",
            name="Patient Data Privacy (HIPAA)",
            version="5.0",
            rules=[
                "All EHR access must be logged with user ID and timestamp",
                "No patient data may be shared outside care team",
                "Screen lock after 60 seconds of inactivity on terminals",
                "Minimum necessary information principle for all queries",
            ],
            effective_from=episode,
        ),
    }


# ─── Generator ───────────────────────────────────────────────

class CrisisGenerator:
    """Generates randomized but reproducible crisis scenarios."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate(
        self,
        crisis_type: CrisisType | None = None,
        episode: int = 0,
        difficulty: float = 0.5,
    ) -> tuple[Crisis, dict[str, Policy]]:
        """Generate a crisis scenario with patients and policies.

        Args:
            crisis_type: Force a specific crisis type, or random if None.
            episode: Current episode number (for policy versioning).
            difficulty: 0.0 (easy) to 1.0 (nightmare).

        Returns:
            Tuple of (Crisis, dict of active policies).
        """
        if crisis_type is None:
            crisis_type = self.rng.choice(list(CrisisType))

        patient_count = int(10 + difficulty * 40)
        incoming_rate = max(1, int(1 + difficulty * 5))

        crisis = self._build_crisis(crisis_type, patient_count, incoming_rate, difficulty)
        policies = _default_policies(episode)

        return crisis, policies

    def _build_crisis(
        self,
        ctype: CrisisType,
        patient_count: int,
        incoming_rate: int,
        difficulty: float,
    ) -> Crisis:
        conditions, name, severity = self._crisis_profile(ctype, difficulty)

        patients = [
            self._generate_patient(conditions, difficulty)
            for _ in range(patient_count)
        ]
        # Sort by severity so the most critical arrive first
        patients.sort(key=lambda p: p.triage_score, reverse=True)

        # Split into immediate admits and pending wave
        immediate = patients[:max(3, int(len(patients) * 0.3))]
        pending = patients[len(immediate):]
        for p in immediate:
            p.status = PatientStatus.INCOMING

        if ctype == CrisisType.OUTBREAK:
            for p in immediate[:3]:
                p.condition = "unknown_pathogen infection"
                p.status = PatientStatus.SERIOUS
                p.triage_score = max(p.triage_score, 6)
                p.ward = WardType.WARD_A

        drug_inv = dict(_DRUG_INVENTORY)
        if difficulty > 0.7:
            # Shortage scenario — cut some drugs
            shortage_drugs = self.rng.sample(list(drug_inv.keys()), k=4)
            for d in shortage_drugs:
                drug_inv[d] = max(1, drug_inv[d] // 4)

        staff_roster = dict(_BASE_STAFF_ROSTER)
        staff_reduction = 1.0
        if ctype == CrisisType.STAFF_SHORTAGE:
            staff_reduction = max(0.3, 1.0 - difficulty * 0.7)
            for role in staff_roster:
                staff_roster[role] = max(1, int(staff_roster[role] * staff_reduction))

        blood_inv = {
            "O+": 20, "O-": 10, "A+": 15, "A-": 8, "B+": 12, "B-": 6, "AB+": 5, "AB-": 3
        }
        if ctype == CrisisType.MASS_CASUALTY:
            blood_inv["O+"] = 8
            blood_inv["O-"] = 4

        return Crisis(
            type=ctype,
            name=name,
            severity=severity,
            patient_count=patient_count,
            incoming_rate=incoming_rate,
            typical_conditions=[c[0] for c in conditions[:5]],
            special_rules=self._special_rules(ctype),
            patient_list=immediate,
            drug_inventory=drug_inv,
            blood_inventory=blood_inv,
            staff_roster=staff_roster,
            icu_config={
                "beds": 2 if ctype == CrisisType.MASS_CASUALTY else 20,
                "ventilators": 3 if ctype == CrisisType.MASS_CASUALTY else 15,
                "overflow_threshold": 0.9,
            },
            insurance_policies={
                "verify_all": True,
                "emergency_bypass": difficulty < 0.3,
            },
            staff_reduction=staff_reduction,
        )

    def _crisis_profile(
        self, ctype: CrisisType, difficulty: float
    ) -> tuple[list[tuple[str, float, int]], str, str]:
        severity = "critical" if difficulty > 0.7 else "high" if difficulty > 0.4 else "medium"

        profiles: dict[CrisisType, tuple[list[tuple[str, float, int]], str]] = {
            CrisisType.MASS_CASUALTY: (
                _MASS_CASUALTY_CONDITIONS,
                self.rng.choice([
                    "Multi-Vehicle Highway Pileup",
                    "Industrial Explosion",
                    "Building Collapse",
                    "Stadium Stampede",
                    "Train Derailment",
                ]),
            ),
            CrisisType.OUTBREAK: (
                _OUTBREAK_CONDITIONS,
                self.rng.choice([
                    "Viral Respiratory Outbreak",
                    "Bacterial Meningitis Cluster",
                    "Hemorrhagic Fever Containment",
                    "Foodborne Illness Surge",
                ]),
            ),
            CrisisType.EQUIPMENT_FAILURE: (
                _EQUIPMENT_CONDITIONS,
                self.rng.choice([
                    "Main Power Grid Failure",
                    "Ventilator Fleet Recall",
                    "Medical Gas Supply Disruption",
                    "IT Infrastructure Collapse",
                ]),
            ),
            CrisisType.STAFF_SHORTAGE: (
                _STAFF_CONDITIONS,
                self.rng.choice([
                    "Mass Staff Illness",
                    "Strike Action — 60% Walkout",
                    "Blizzard — Staff Cannot Reach Hospital",
                    "Post-Holiday Staffing Crisis",
                ]),
            ),
        }
        conditions, name = profiles[ctype]
        return conditions, name, severity

    def _generate_patient(
        self,
        conditions: list[tuple[str, float, int]],
        difficulty: float,
    ) -> Patient:
        first = self.rng.choice(_FIRST_NAMES)
        last = self.rng.choice(_LAST_NAMES)
        condition_name, base_deterioration, triage_score = self.rng.choice(conditions)
        deterioration = base_deterioration * (0.5 + difficulty)

        age = self.rng.randint(3, 92)
        if age > 70:
            deterioration *= 1.3
        if age < 10:
            deterioration *= 1.1

        if triage_score >= 8:
            status = PatientStatus.CRITICAL
        elif triage_score >= 5:
            status = PatientStatus.SERIOUS
        else:
            status = PatientStatus.STABLE

        allergies = []
        if self.rng.random() < 0.2:
            allergies.append(self.rng.choice(["penicillin", "morphine", "midazolam"]))
        insurance_plan = self.rng.choice(
            ["PPO_GOLD", "HMO_BASIC", "MEDICAID", "UNINSURED", "EMERGENCY_ONLY"]
        )

        return Patient(
            id=str(uuid.uuid4())[:8],
            name=f"{first} {last}",
            age=age,
            condition=condition_name,
            status=status,
            triage_score=triage_score,
            allergies=allergies,
            insurance_plan=insurance_plan,
            icu_required=triage_score >= 8,
            deterioration_rate=deterioration,
        )

    def _special_rules(self, ctype: CrisisType) -> list[str]:
        rules: dict[CrisisType, list[str]] = {
            CrisisType.MASS_CASUALTY: [
                "Activate hospital-wide Code Orange",
                "Cancel all elective procedures",
                "Establish secondary triage area in parking structure",
                "Blood bank on emergency release protocol",
            ],
            CrisisType.OUTBREAK: [
                "Activate infection control team",
                "Isolate suspected cases in negative-pressure rooms",
                "Mandatory PPE for all patient contact",
                "Report to county health department within 1 hour",
            ],
            CrisisType.EQUIPMENT_FAILURE: [
                "Switch to backup generator within 30 seconds",
                "Manual documentation until IT restored",
                "Prioritize life-support equipment repair",
                "External equipment request to neighboring hospitals",
            ],
            CrisisType.STAFF_SHORTAGE: [
                "Activate emergency callback roster",
                "Request mutual aid from partner hospitals",
                "Suspend non-essential services",
                "Extend shift limits with CMO approval only",
            ],
        }
        return rules.get(ctype, [])

    def inject_violation(self, crisis: Crisis) -> dict[str, Any]:
        """Generate a policy violation for agent detection exercises.

        Returns a dict describing the violation to be embedded in the state.
        """
        violation_types = [
            {
                "type": "medication_error",
                "description": "Controlled substance dispensed without double-verification",
                "severity": "high",
                "policy_id": "POL-003",
                "detectable_by": ["pharmacy", "cmo_oversight"],
            },
            {
                "type": "staff_fatigue",
                "description": "Surgeon operating after 18 consecutive hours",
                "severity": "critical",
                "policy_id": "POL-004",
                "detectable_by": ["hr_rostering", "cmo_oversight"],
            },
            {
                "type": "ehr_access",
                "description": "Non-care-team member accessed patient records",
                "severity": "high",
                "policy_id": "POL-005",
                "detectable_by": ["it_systems", "cmo_oversight"],
            },
            {
                "type": "triage_delay",
                "description": "Patient waited 12 minutes without triage score assignment",
                "severity": "medium",
                "policy_id": "POL-001",
                "detectable_by": ["er_triage", "cmo_oversight"],
            },
            {
                "type": "icu_overcapacity",
                "description": "ICU nurse-to-patient ratio exceeded 1:3",
                "severity": "high",
                "policy_id": "POL-002",
                "detectable_by": ["icu_management", "hr_rostering", "cmo_oversight"],
            },
        ]

        return self.rng.choice(violation_types)
