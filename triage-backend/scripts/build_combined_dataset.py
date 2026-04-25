#!/usr/bin/env python3
"""
build_combined_dataset.py — Fast local dataset builder + HF cloud augmentation.

Phase 1 (LOCAL — runs in seconds):
  - Loads existing env rollout prompts
  - Generates synthetic crisis prompts from templates
  - Creates a ready-to-push JSONL

Phase 2 (CLOUD — runs in train_grpo_hf.py on A100):
  - Downloads + integrates the 34 HF datasets on-the-fly during training

Usage:
    python scripts/build_combined_dataset.py
    python scripts/build_combined_dataset.py --target 3000 --output data/grpo/combined_train.jsonl
"""
from __future__ import annotations
import argparse, json, random, logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

AGENT_TYPES = ["er_triage","icu_management","pharmacy","cmo_oversight","hr_rostering","it_systems"]
CRISIS_TYPES = ["mass_casualty","outbreak","equipment_failure","staff_shortage"]
ACTION_MAP = {
    "er_triage": ["TRIAGE_PATIENT","ASSIGN_TREATMENT","UPDATE_EHR"],
    "icu_management": ["TRANSFER_TO_ICU","TRANSFER_TO_WARD","ACTIVATE_OVERFLOW"],
    "pharmacy": ["ORDER_MEDICATION","FLAG_POLICY_VIOLATION"],
    "cmo_oversight": ["OVERRIDE_DECISION","ACTIVATE_OVERFLOW"],
    "hr_rostering": ["REQUEST_STAFF","FLAG_POLICY_VIOLATION"],
    "it_systems": ["UPDATE_EHR","FLAG_POLICY_VIOLATION","VERIFY_INSURANCE"],
}
ROLE_DESC = {
    "er_triage": "ER Triage — patient assessment, severity classification, treatment assignment",
    "icu_management": "ICU Management — bed allocation, transfers, overflow protocols",
    "pharmacy": "Pharmacy — medication orders, drug interactions, formulary compliance",
    "cmo_oversight": "CMO Oversight — policy enforcement, escalation decisions, resource authorization",
    "hr_rostering": "HR Rostering — staff allocation, shift management, fatigue monitoring",
    "it_systems": "IT Systems — EHR integrity, backup protocols, policy compliance monitoring",
}

# ── Diverse medical scenarios for synthetic generation ──────────────────────
MEDICAL_SCENARIOS = [
    "45-year-old male presenting with chest pain, diaphoresis, ST elevation in leads II, III, aVF. Troponin positive. History of hypertension and diabetes.",
    "72-year-old female with acute respiratory distress, SpO2 82% on room air, bilateral crackles. Known CHF with EF 25%. BNP 1800 pg/mL.",
    "28-year-old trauma patient, MVC at 60mph. GCS 8, hypotensive BP 70/40, tachycardic HR 130. FAST exam positive for free fluid.",
    "6-month-old infant with fever 40.2°C, bulging fontanelle, petechial rash. WBC 22,000. Pending LP results.",
    "55-year-old diabetic with infected foot ulcer, septic. Lactate 6.2, MAP 58. On vasopressors. Creatinine rising from 1.2 to 3.8.",
    "30-year-old pregnant female, 34 weeks, presenting with seizures. BP 180/110, proteinuria 3+. Suspected eclampsia.",
    "80-year-old on warfarin, INR 8.2, GI bleeding. Hemoglobin dropped from 12 to 7.4 in 6 hours. 2 units pRBC ordered.",
    "42-year-old with acute pancreatitis, Ranson score 5. CT shows necrosis >30%. Requires ICU monitoring and TPN.",
    "19-year-old with suspected meningococcal meningitis. Purpura fulminans developing. Contacts need prophylaxis.",
    "65-year-old post-CABG day 2, sudden onset atrial fibrillation with RVR, HR 160. Hemodynamically unstable.",
    "38-year-old anaphylaxis from penicillin. Angioedema, stridor, BP 60/30 despite 2 doses epinephrine. Requires intubation.",
    "50-year-old stroke code, NIHSS 18, last known normal 90 minutes ago. CT negative for hemorrhage. tPA candidate.",
    "Burn patient, 40% TBSA, second and third degree. Parkland formula initiated. Airway edema developing.",
    "Multiple casualty incident: bus crash, 15 patients incoming. Current ED at 85% capacity. 3 trauma bays available.",
    "Cardiac arrest in radiology department. Patient on CT table. Code blue activated. ROSC after 12 minutes CPR.",
    "Drug-resistant TB patient in isolation. HEPA filter malfunction on floor 3. 12 staff members potentially exposed.",
    "Pediatric drowning, submersion time estimated 8 minutes. Core temp 32°C. Pupils fixed. ECMO considered.",
    "Mass casualty from chemical plant explosion. 8 patients with inhalation injuries. Cyanide exposure suspected.",
    "Post-operative patient developing DIC. Platelets 28,000, fibrinogen 80. Massive transfusion protocol activated.",
    "ICU bed crisis: 20/20 beds occupied. 3 patients boarding in ED awaiting ICU. New septic shock patient arriving.",
    "Medication error reported: patient received 10x dose of methotrexate. Current renal function declining.",
    "67-year-old with acute liver failure, ammonia 180, INR 4.5. Listed for emergent transplant. MELD score 38.",
    "Staff shortage crisis: night shift has 40% nursing coverage. 3 nurses called in sick. Agency staff unavailable.",
    "IT system downtime: EHR offline for 45 minutes. Paper charting in effect. Medication reconciliation compromised.",
    "Neonatal emergency: premature 26-week infant, RDS, requiring surfactant. NICU at capacity.",
    "Suspected opioid overdose cluster: 6 patients in 2 hours from same batch. Naloxone shortage in pharmacy.",
    "Elderly patient with hip fracture, on dual antiplatelet therapy. Surgical timing vs bleeding risk decision.",
    "Psychiatric patient with acute psychosis, violent behavior. Chemical restraint needed. Medical clearance pending.",
    "Hospital-acquired C. diff outbreak on ward 4B. 8 confirmed cases in 5 days. Contact isolation strain on PPE supply.",
    "Blood bank shortage: O-negative supply at critical level (2 units). MTP request for trauma patient pending.",
]

ETHICAL_DILEMMAS = [
    "Two critical patients, one ICU bed available. Patient A: 45yo with reversible condition. Patient B: 80yo with multiple comorbidities.",
    "Family demands continued full code on brain-dead patient. Organ donation team waiting. Ethics consult requested.",
    "Physician fatigue: surgeon has been awake 28 hours. Emergency case requires their specific expertise. No backup available.",
    "Pharmaceutical company offers free experimental drug for dying patient. Not FDA approved. Family consents.",
    "Undocumented patient needs emergent dialysis. No insurance. Hospital at financial risk. EMTALA obligations apply.",
    "Whistleblower report: senior attending making errors. Two near-misses this week. No formal complaint filed yet.",
    "Rationing scenario: ventilator shortage during surge. Crisis standards of care activated. Scoring criteria disputed.",
    "Patient refuses blood transfusion on religious grounds. Hemoglobin 4.2. Actively bleeding. Competent adult.",
    "Research protocol deviation discovered mid-trial. Affects 30 enrolled patients. IRB notification required.",
    "Staff member tests positive for infectious disease after patient contact. Contact tracing vs privacy concerns.",
]

DRUG_INTERACTIONS = [
    "Patient on warfarin prescribed amiodarone. INR expected to increase 30-50%. Requires dose reduction and monitoring.",
    "Concurrent methotrexate and trimethoprim ordered. Risk of pancytopenia from folate antagonism. Review required.",
    "Serotonin syndrome risk: patient on SSRI prescribed tramadol. Alternative analgesic needed.",
    "Digoxin toxicity risk with new amiodarone addition. Digoxin level 2.8. Reduce digoxin dose by 50%.",
    "Contraindicated combination: MAO inhibitor with meperidine. Hypertensive crisis risk. Switch to morphine.",
    "Nephrotoxic combination: vancomycin + piperacillin/tazobactam. Monitor creatinine q8h. Consider alternative.",
    "QT prolongation risk: azithromycin + ondansetron in cardiac patient. Baseline QTc 480ms.",
    "Immunosuppressant interaction: tacrolimus level tripled after fluconazole initiation. Toxic level 28 ng/mL.",
]


def build_prompt(context: str, agent: str, crisis: str, diff: float, step: int, rng: random.Random) -> str:
    n_pat = rng.randint(5, 30)
    icu = round(rng.uniform(0.3, 0.98), 2)
    crit = rng.randint(1, min(12, n_pat))
    vi = rng.randint(0, 5); vc = rng.randint(0, vi)
    surv = round(1.0 - rng.uniform(0, 0.18) * diff, 3)
    pts = []
    for _ in range(min(crit, 5)):
        pid = rng.randint(1, 99)
        pts.append(f"  P-{pid:03d}: CRITICAL — BP {rng.randint(55,95)}/{rng.randint(25,65)}, HR {rng.randint(95,160)}, SpO2 {rng.randint(78,96)}%")
    pb = "\n".join(pts) if pts else "  (none)"
    acts = ", ".join(ACTION_MAP[agent])
    return (
        f"You are the {agent.upper()} agent in a hospital crisis simulation.\n\n"
        f"CRISIS: {crisis.upper()}\nSTEP: {step}/20\n"
        f"ICU OCCUPANCY: {int(icu*100)}% ({int(icu*20)}/20 beds)\n"
        f"CRITICAL PATIENTS ({crit} total — top 5):\n{pb}\n"
        f"VIOLATIONS INJECTED: {vi} | CAUGHT: {vc}\n"
        f"SURVIVAL RATE: {surv*100:.1f}%\n\n"
        f"MEDICAL CONTEXT:\n{context[:500]}\n\n"
        f"Your role: {ROLE_DESC[agent]}\n\n"
        f"Decide the single most important action right now. Respond with ONLY valid JSON:\n"
        f'{{\n  "action_type": "<one of: {acts}>",\n'
        f'  "target_id": <patient ID integer or 0>,\n'
        f'  "priority": <integer 1-10>,\n'
        f'  "reasoning": "<1-2 sentences citing specific patient data or metrics>"\n}}'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/grpo/combined_train.jsonl")
    parser.add_argument("--local-prompts", default="data/grpo/train_expanded.jsonl")
    parser.add_argument("--target", type=int, default=3000, help="Target total prompts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_records = []
    sources = Counter()

    # 1. Load existing env rollouts
    local_path = Path(args.local_prompts)
    if local_path.exists():
        with open(local_path) as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line.strip()))
        sources["env-rollout"] = len(all_records)
        logger.info("Loaded %d env rollouts", len(all_records))

    # 2. Generate synthetic medical crisis prompts
    needed = max(0, args.target - len(all_records))
    scenarios = MEDICAL_SCENARIOS + ETHICAL_DILEMMAS + DRUG_INTERACTIONS
    generated = 0

    while generated < needed:
        context = rng.choice(scenarios)
        # Add random variation
        if rng.random() < 0.3:
            context += f" Additionally, hospital currently has {rng.randint(2,8)} pending surgeries and {rng.randint(1,4)} incoming ambulances."
        if rng.random() < 0.2:
            context += f" Staff fatigue index: {rng.uniform(0.3,0.9):.1f}. Shift change in {rng.randint(1,6)} hours."

        agent = rng.choice(AGENT_TYPES)
        crisis = rng.choice(CRISIS_TYPES)
        diff = rng.choice([0.2, 0.4, 0.6, 0.8, 1.0])
        step = rng.randint(0, 19)

        prompt = build_prompt(context, agent, crisis, diff, step, rng)
        all_records.append({
            "prompt": prompt,
            "crisis_type": crisis,
            "difficulty": diff,
            "step": step,
            "episode": -1,
            "agent_type": agent,
            "source": "synthetic-crisis",
        })
        generated += 1
        sources["synthetic-crisis"] += 1

    rng.shuffle(all_records)

    with open(output, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{'═'*60}")
    print(f"  Combined GRPO Dataset Ready (Local Phase)")
    print(f"  Total: {len(all_records)} prompts")
    print(f"  Size:  {output.stat().st_size / 1024:.1f} KB")
    for src, cnt in sources.most_common():
        print(f"    {src:25s} {cnt:5d}")
    print(f"\n  NOTE: 34 HF datasets will be integrated on cloud")
    print(f"  via prepare_hf_dataset_v2.py during training")
    print(f"{'═'*60}\n")

if __name__ == "__main__":
    main()
