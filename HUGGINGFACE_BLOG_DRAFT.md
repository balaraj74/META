# TRIAGE: Training Multi-Agent Hospital Responders With OpenEnv, DPO, and Schema Drift

TRIAGE is our hackathon project for high-stakes hospital crisis response. We built a simulated hospital where six specialist agents collaborate through typed messages, enterprise application workflows, and shifting expert preferences while handling mass-casualty events, outbreaks, equipment failures, and staffing shortages.

## Why this project

Real hospital operations are not a single-agent problem. Critical decisions depend on coordination across emergency medicine, ICU operations, pharmacy safety, staffing, and IT systems. TRIAGE turns that reality into a training environment:

- `HospitalEnv` provides an OpenEnv-compatible crisis simulator.
- Six agents act as CMO, ER, ICU, Pharmacy, HR, and IT specialists.
- A typed `MessageBus` captures cross-agent escalation and handoffs.
- Enterprise app simulators model EHR, ICU, Pharmacy, HRIS, Insurance, and IT tracker workflows.
- A seven-component reward model scores survival, compliance, coordination, oversight, depth, adaptation, and expert alignment.

## System architecture

At the core is `HospitalEnv`, which exposes reset/step/state lifecycle methods and structured observations for patients, resources, agent states, policy state, crisis state, and expert signals.

The agents run in two modes:

- Rule-based fallback for fast local simulation and demo reliability
- LLM-backed mode for richer reasoning and DPO data collection

All agents communicate through a typed message bus with priority handling. That lets us model realistic patterns like:

- ER escalating a deteriorating patient to the CMO
- ICU requesting explicit override authority before bypassing queue order
- Pharmacy requiring lookup and interaction prechecks before controlled medication fulfillment
- IT reacting to schema drift in insurance and regulatory workflows

## Enterprise workflow realism

One of our biggest design goals was making the environment feel like an actual hospital stack instead of a toy gridworld.

TRIAGE now simulates six operational systems:

1. EHR for patient lookup and record mutation
2. ICU manager for bed queue checks, allocation, and override-gated transfers
3. Pharmacy workflow with interaction checks, coverage checks, and audit trails
4. HRIS for fatigue monitoring and staff callback requests
5. Insurance portal with contract/schema drift awareness
6. IT tracker for ventilator allocation, uptime status, and drift-linked incidents

This matters because real failures are often workflow failures, not just clinical failures.

## Schema drift as a first-class training signal

We expanded drift beyond a single policy mutation stream into three explicit domains:

- Policy drift: triage protocol or operational policy changes
- Contract drift: insurance portal schema changes, field renames, and authorization workflow changes
- Regulatory drift: new compliance requirements, tighter medication signoff rules, or consent/audit constraints

Agents do not just need to solve the base crisis. They need to adapt when the rules move underneath them.

## Reward model

Our reward model combines seven components:

- Survival
- Compliance
- Coordination
- Oversight
- Depth
- Adaptation
- Expert alignment

Two pieces we cared about in particular:

- Depth reward now uses token-scaled reasoning with diminishing returns and padding penalties
- Expert signals now shift the reward weighting dynamically so the environment can emphasize quality, speed, or cost depending on the current simulated expert preference vector

That means the agents are not being optimized against a frozen static rubric.

## DPO pipeline

For post-training, we built a DPO pipeline:

- `EpisodeCollector` generates trajectories
- `PreferenceLabeler` and `DatasetAdapter` build chosen/rejected training pairs
- `TRIAGEDPOTrainer` fine-tunes compact Qwen models locally and larger variants on Colab

This gives us a clean story for before/after improvement:

- baseline multi-agent performance
- collected preference data
- DPO fine-tuning
- improved reward and crisis-handling behavior after training

## Demo story

For the live demo, we built a command-center frontend with:

- dashboard, replay, training, visualizer, mobile, and sponsor pages
- a dedicated pitch mode with scripted narrative
- keyboard controls for presenter flow
- fullscreen support
- WebSocket fallback to mock simulation so the pitch survives backend disconnects

The result is a demo that can tell a coherent three-minute story even under hackathon pressure.

## What we learned

The biggest lesson was that multi-agent evaluation gets much more compelling when:

- workflows are stateful
- enterprise APIs have rules and failure modes
- expert preferences are allowed to change
- adaptation is rewarded, not just final outcomes

That combination turned the environment from a static simulator into a training loop for operational judgment.

## What’s next

Our next steps are:

1. Run a clean 10-episode baseline and publish the reward curve
2. Fine-tune the 1.5B DPO model and compare before/after behavior
3. Push the trained model and dataset artifacts to the Hugging Face Hub
4. Publish this write-up as the final Hugging Face blog post for the submission

## Reproducibility

- Backend: FastAPI + OpenEnv-compatible simulation
- Frontend: React + TypeScript + TanStack Router + WebSocket live view
- Training: Hugging Face TRL DPO + LoRA adapters
- Retrieval: BM25 StrategyMemory across prior episodes

TRIAGE is our attempt to show that hospital crisis response can be trained as a multi-agent, workflow-grounded adaptation problem rather than a single static benchmark.
