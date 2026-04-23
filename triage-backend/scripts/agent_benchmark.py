#!/usr/bin/env python3
"""
TRIAGE Multi-Agent Benchmark Suite
====================================
Measures real performance of the agent system, NOT just the HTTP API.

Suites:
  1. Message Bus      — send/broadcast throughput + token budget tracking
  2. Agent Decision   — per-agent decide() latency (mock mode, fast)
  3. Orchestrator     — single step latency with all 6 agents
  4. Full Episode     — N-step rollout throughput (steps/sec)
  5. LLM vs Mock      — compare latency when LLM backend is live
  6. Crisis Scenarios — step throughput per crisis type

Run:
    cd triage-backend
    python scripts/agent_benchmark.py                   # all suites (mock LLM)
    python scripts/agent_benchmark.py --llm             # include real LLM tests
    python scripts/agent_benchmark.py --suite bus       # single suite
    python scripts/agent_benchmark.py --steps 20        # episode length
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from triage.agents.message_bus import MessageBus
from triage.agents.orchestrator import AgentOrchestrator
from triage.env.hospital_env import HospitalEnv
from triage.env.state import (
    AgentMessage,
    AgentType,
    CrisisType,
    MessageType,
)

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
BOLD    = "\033[1m"
RESET   = "\033[0m"
DIM     = "\033[2m"

# ── Helpers ──────────────────────────────────────────────────────────────────
def _hr(char: str = "─", width: int = 68) -> str:
    return char * width

def _p(label: str, value: str, unit: str = "", color: str = GREEN) -> None:
    print(f"  {label:<40} {color}{BOLD}{value:>10}{RESET} {DIM}{unit}{RESET}")

def _header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{_hr('═')}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{_hr('═')}{RESET}")

def _section(title: str) -> None:
    print(f"\n{BOLD}  ── {title} {DIM}{_hr('·', 48 - len(title))}{RESET}")

def _latency_color(ms: float) -> str:
    if ms < 50:   return GREEN
    if ms < 300:  return YELLOW
    return RED

# ── Result container ─────────────────────────────────────────────────────────
@dataclass
class BenchResult:
    suite: str
    label: str
    samples: list[float] = field(default_factory=list)  # milliseconds
    errors: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def p95(self) -> float:
        if len(self.samples) < 2:
            return self.samples[0] if self.samples else 0.0
        return sorted(self.samples)[int(len(self.samples) * 0.95)]

    @property
    def p99(self) -> float:
        if len(self.samples) < 2:
            return self.samples[0] if self.samples else 0.0
        return sorted(self.samples)[int(len(self.samples) * 0.99)]

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    def print_row(self) -> None:
        color = _latency_color(self.median)
        _p(self.label, f"{self.median:.1f}", "ms  (median)", color)
        print(f"    {DIM}mean={self.mean:.1f}ms  p95={self.p95:.1f}ms  "
              f"p99={self.p99:.1f}ms  σ={self.stdev:.1f}ms  n={self.n}  "
              f"errors={self.errors}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════
# Suite 1 — Message Bus
# ═══════════════════════════════════════════════════════════════════════════
async def bench_message_bus(iterations: int = 1000) -> list[BenchResult]:
    """Measure raw send() and broadcast() throughput."""
    _section("Message Bus Throughput")

    def _make_msg(msg_type: MessageType = MessageType.ALERT) -> AgentMessage:
        return AgentMessage(
            from_agent=AgentType.ER_TRIAGE,
            to_agent=AgentType.CMO_OVERSIGHT,
            content="Benchmark message — patient critical",
            msg_type=msg_type,
            priority=8,
        )

    # --- direct send ---
    bus = MessageBus(token_budget=10_000_000)
    samples_send: list[float] = []
    for _ in range(iterations):
        msg = _make_msg()
        t0 = time.perf_counter()
        await bus.send(msg)
        samples_send.append((time.perf_counter() - t0) * 1000)

    r_send = BenchResult("bus", "Direct send()", samples_send)
    r_send.meta["total_messages"] = bus.message_count

    # --- broadcast ---
    bus2 = MessageBus(token_budget=10_000_000)
    # register dummy subscribers so broadcast actually delivers
    async def _noop(m: AgentMessage) -> None: ...
    for agent in AgentType:
        bus2.subscribe(agent, _noop)
    bus2.subscribe_broadcast(_noop)

    samples_bc: list[float] = []
    for _ in range(iterations // 10):  # broadcast is heavier
        msg = _make_msg(MessageType.BROADCAST)
        t0 = time.perf_counter()
        await bus2.send(msg)
        samples_bc.append((time.perf_counter() - t0) * 1000)

    r_bc = BenchResult("bus", "Broadcast send()", samples_bc)

    # --- token budget enforcement ---
    bus3 = MessageBus(token_budget=100)  # tiny budget
    rejected = 0
    for _ in range(20):
        msg = _make_msg()
        delivered = await bus3.send(msg)
        if not delivered:
            rejected += 1
    r_budget = BenchResult("bus", "Token budget enforcement", [0.1])
    r_budget.meta["rejected"] = rejected

    results = [r_send, r_bc, r_budget]
    for r in results:
        r.print_row()

    throughput = 1000.0 / (r_send.mean / 1000.0) if r_send.mean > 0 else 0
    print(f"\n  {DIM}Bus throughput: ~{throughput:,.0f} msgs/sec{RESET}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Suite 2 — Per-Agent Decision Latency (mock mode, no LLM)
# ═══════════════════════════════════════════════════════════════════════════
async def bench_agent_decisions(iterations: int = 50) -> list[BenchResult]:
    """Benchmark each agent's act() → decide() path in mock (rule-based) mode."""
    _section("Per-Agent Decision Latency  [mock LLM]")

    config_path = ROOT / "config" / "agents.yaml"
    env = HospitalEnv(seed=42)
    await env.reset()
    orch = AgentOrchestrator(
        env=env,
        agents_config_path=str(config_path),
        mock_llm=True,
        seed=42,
    )
    state = orch.state

    results: list[BenchResult] = []
    for agent_type, agent in orch.agents.items():
        samples: list[float] = []
        errors = 0
        for _ in range(iterations):
            try:
                t0 = time.perf_counter()
                actions = await agent.act(state)
                elapsed = (time.perf_counter() - t0) * 1000
                samples.append(elapsed)
            except Exception as exc:
                errors += 1
                print(f"    {RED}Error in {agent_type.value}: {exc}{RESET}")

        r = BenchResult("agent", agent_type.value, samples, errors)
        r.meta["avg_actions_per_call"] = (
            len(actions) / iterations if samples else 0
        )
        results.append(r)
        r.print_row()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Suite 3 — Orchestrator Single-Step Latency
# ═══════════════════════════════════════════════════════════════════════════
async def bench_orchestrator_step(iterations: int = 30) -> list[BenchResult]:
    """Measure full orchestrator.step() — all 6 agents act, env transitions."""
    _section("Orchestrator Step Latency  [mock LLM]")

    config_path = ROOT / "config" / "agents.yaml"

    samples: list[float] = []
    reward_samples: list[float] = []
    errors = 0

    for i in range(iterations):
        try:
            # fresh env per step keeps results deterministic
            env = HospitalEnv(seed=i)
            orch = AgentOrchestrator(
                env=env,
                agents_config_path=str(config_path),
                mock_llm=True,
                seed=i,
            )
            await orch.reset()

            t0 = time.perf_counter()
            result = await orch.step()
            elapsed = (time.perf_counter() - t0) * 1000
            samples.append(elapsed)
            reward_samples.append(result.reward)
        except Exception as exc:
            errors += 1
            print(f"    {RED}Step error [{i}]: {exc}{RESET}")

    r_step = BenchResult("orch", "orchestrator.step()", samples, errors)
    r_step.meta["mean_reward"] = statistics.mean(reward_samples) if reward_samples else 0

    r_step.print_row()
    avg_reward = statistics.mean(reward_samples) if reward_samples else 0
    print(f"    {DIM}avg reward per step: {avg_reward:.4f}{RESET}")

    return [r_step]


# ═══════════════════════════════════════════════════════════════════════════
# Suite 4 — Full Episode Throughput
# ═══════════════════════════════════════════════════════════════════════════
async def bench_episode(n_steps: int = 20) -> list[BenchResult]:
    """Run a complete N-step episode and measure steps/sec."""
    _section(f"Full Episode Throughput  [{n_steps} steps, mock LLM]")

    config_path = ROOT / "config" / "agents.yaml"
    env = HospitalEnv(seed=99, max_steps=n_steps)
    orch = AgentOrchestrator(
        env=env,
        agents_config_path=str(config_path),
        mock_llm=True,
        seed=99,
    )

    await orch.reset()

    step_times: list[float] = []
    rewards: list[float] = []
    step = 0

    t_episode_start = time.perf_counter()
    while not env.is_terminal and step < n_steps:
        t0 = time.perf_counter()
        result = await orch.step()
        step_times.append((time.perf_counter() - t0) * 1000)
        rewards.append(result.reward)
        step += 1
        if result.terminated:
            break

    t_episode_total = (time.perf_counter() - t_episode_start)

    r = BenchResult("episode", f"{n_steps}-step rollout", step_times)
    r.meta.update({
        "actual_steps": step,
        "total_time_s": t_episode_total,
        "steps_per_sec": step / t_episode_total,
        "total_reward": sum(rewards),
        "bus_messages": orch.bus.message_count,
        "bus_tokens": orch.bus.tokens_used,
    })

    r.print_row()
    print(f"    {DIM}episode wall-time: {t_episode_total:.2f}s  "
          f"| steps/sec: {r.meta['steps_per_sec']:.1f}  "
          f"| total_reward: {r.meta['total_reward']:.4f}  "
          f"| bus_msgs: {r.meta['bus_messages']}  "
          f"| bus_tokens: {r.meta['bus_tokens']}{RESET}")

    return [r]


# ═══════════════════════════════════════════════════════════════════════════
# Suite 5 — Crisis Type Comparison
# ═══════════════════════════════════════════════════════════════════════════
async def bench_crisis_scenarios(steps_per_crisis: int = 10) -> list[BenchResult]:
    """Compare step latency and reward across all crisis types."""
    _section(f"Crisis Scenario Comparison  [{steps_per_crisis} steps each]")

    config_path = ROOT / "config" / "agents.yaml"
    results: list[BenchResult] = []

    for crisis in CrisisType:
        step_times: list[float] = []
        rewards: list[float] = []

        env = HospitalEnv(seed=7, max_steps=steps_per_crisis)
        orch = AgentOrchestrator(
            env=env,
            agents_config_path=str(config_path),
            mock_llm=True,
            seed=7,
        )
        scenario = {"crisis_type": crisis.value}
        try:
            await orch.reset(scenario)
        except Exception:
            await orch.reset()  # fallback if scenario injection not supported

        step = 0
        while not env.is_terminal and step < steps_per_crisis:
            t0 = time.perf_counter()
            try:
                result = await orch.step()
                step_times.append((time.perf_counter() - t0) * 1000)
                rewards.append(result.reward)
            except Exception as exc:
                print(f"    {RED}Crisis {crisis.value} step error: {exc}{RESET}")
            step += 1

        r = BenchResult("crisis", crisis.value, step_times)
        r.meta["avg_reward"] = statistics.mean(rewards) if rewards else 0
        results.append(r)
        r.print_row()
        print(f"    {DIM}avg_reward={r.meta['avg_reward']:.4f}  "
              f"steps={len(step_times)}{RESET}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Suite 6 — LLM vs Mock Comparison (optional — requires backend running)
# ═══════════════════════════════════════════════════════════════════════════
async def bench_llm_vs_mock(iterations: int = 5) -> list[BenchResult]:
    """Compare agent decision latency: mock rule-based vs real LLM backend."""
    _section(f"LLM vs Mock Decision Latency  [{iterations} calls each]")

    import httpx

    config_path = ROOT / "config" / "agents.yaml"

    # -- mock mode --
    mock_samples: list[float] = []
    env = HospitalEnv(seed=1)
    await env.reset()
    orch_mock = AgentOrchestrator(
        env=env, agents_config_path=str(config_path), mock_llm=True, seed=1
    )
    agent = orch_mock.agents.get(AgentType.ER_TRIAGE)
    if agent:
        for _ in range(iterations):
            t0 = time.perf_counter()
            await agent.act(orch_mock.state)
            mock_samples.append((time.perf_counter() - t0) * 1000)

    r_mock = BenchResult("llm", "ER_TRIAGE  (mock)", mock_samples)
    r_mock.print_row()

    # -- LLM mode via HTTP backend --
    llm_samples: list[float] = []
    errors = 0
    backend_url = "http://localhost:8000"
    payload = {
        "agent_type": "ER_TRIAGE",
        "message": "Patient incoming — chest pain, BP 90/60, HR 120. Recommend action.",
        "context": {"step": 0, "crisis_type": "MASS_CASUALTY"},
    }

    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            # warm-up
            await client.post(f"{backend_url}/api/chat", json=payload)

            for i in range(iterations):
                t0 = time.perf_counter()
                resp = await client.post(f"{backend_url}/api/chat", json=payload)
                elapsed = (time.perf_counter() - t0) * 1000
                if resp.status_code == 200:
                    llm_samples.append(elapsed)
                else:
                    errors += 1

    except Exception as exc:
        print(f"    {YELLOW}Backend unreachable — skipping LLM test: {exc}{RESET}")
        return [r_mock]

    r_llm = BenchResult("llm", "ER_TRIAGE  (LLM /api/chat)", llm_samples, errors)
    r_llm.print_row()

    if mock_samples and llm_samples:
        speedup = statistics.median(llm_samples) / statistics.median(mock_samples)
        color = RED if speedup > 50 else YELLOW if speedup > 10 else GREEN
        print(f"\n  LLM overhead: {color}{BOLD}{speedup:.1f}×{RESET} slower than mock")

    return [r_mock, r_llm]


# ═══════════════════════════════════════════════════════════════════════════
# Suite 7 — Message Bus Stats (end of episode)
# ═══════════════════════════════════════════════════════════════════════════
async def bench_bus_stats(n_steps: int = 20) -> list[BenchResult]:
    """Run episode and report message bus usage stats."""
    _section(f"Message Bus Usage Stats  [{n_steps}-step episode]")

    config_path = ROOT / "config" / "agents.yaml"
    env = HospitalEnv(seed=42, max_steps=n_steps)
    orch = AgentOrchestrator(
        env=env, agents_config_path=str(config_path), mock_llm=True, seed=42
    )
    await orch.reset()

    step = 0
    while not env.is_terminal and step < n_steps:
        await orch.step()
        step += 1

    stats = orch.bus.stats()
    print(f"  {'Total messages':<40} {BOLD}{stats['total_messages']}{RESET}")
    print(f"  {'Tokens used':<40} {BOLD}{stats['tokens_used']:,}{RESET}")
    print(f"  {'Tokens remaining':<40} {BOLD}{stats['tokens_remaining']:,}{RESET}")
    print(f"  {'Budget utilization':<40} {BOLD}{stats['budget_utilization']:.1%}{RESET}")

    if stats.get("by_type"):
        print(f"\n  {DIM}By message type:{RESET}")
        for mtype, count in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
            print(f"    {mtype:<35} {count}")

    if stats.get("by_agent"):
        print(f"\n  {DIM}By sender agent:{RESET}")
        for agent, count in sorted(stats["by_agent"].items(), key=lambda x: -x[1]):
            print(f"    {agent:<35} {count}")

    return []


# ═══════════════════════════════════════════════════════════════════════════
# Final summary table
# ═══════════════════════════════════════════════════════════════════════════
def _print_summary(all_results: list[BenchResult]) -> None:
    _header("BENCHMARK SUMMARY")
    print(f"\n  {'Suite':<14} {'Label':<42} {'Median':>9} {'p95':>9} {'Errors':>7}")
    print(f"  {_hr('─', 85)}")

    for r in all_results:
        if not r.samples:
            continue
        color = _latency_color(r.median)
        err_str = f"{RED}{r.errors}{RESET}" if r.errors else f"{DIM}0{RESET}"
        print(
            f"  {DIM}{r.suite:<14}{RESET}"
            f"{r.label:<42}"
            f"{color}{BOLD}{r.median:>8.1f}{RESET}ms"
            f"{r.p95:>8.1f}ms"
            f"  {err_str}"
        )

    total_errors = sum(r.errors for r in all_results)
    print(f"\n  {_hr()}")
    status_color = GREEN if total_errors == 0 else RED
    status_text  = "ALL CLEAN" if total_errors == 0 else f"{total_errors} ERRORS"
    print(f"  {status_color}{BOLD}Overall status: {status_text}{RESET}\n")

    # Throughput summary
    print(f"  {DIM}Latency guide: {GREEN}< 50ms{RESET}{DIM} = excellent  "
          f"{YELLOW}< 300ms{RESET}{DIM} = acceptable  "
          f"{RED}>= 300ms{RESET}{DIM} = needs tuning{RESET}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
async def main() -> None:
    parser = argparse.ArgumentParser(description="TRIAGE Agent Benchmark Suite")
    parser.add_argument("--suite", choices=["bus", "agent", "orch", "episode", "crisis", "llm", "stats", "all"],
                        default="all", help="Which benchmark suite to run")
    parser.add_argument("--steps", type=int, default=20, help="Steps per episode (default: 20)")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations per per-agent test (default: 50)")
    parser.add_argument("--llm", action="store_true", help="Include LLM vs mock comparison (requires backend)")
    args = parser.parse_args()

    _header("TRIAGE Multi-Agent Benchmark")
    print(f"  {DIM}Episode steps  : {args.steps}")
    print(f"  Agent iterations: {args.iterations}")
    print(f"  LLM tests       : {'enabled' if args.llm else 'disabled (use --llm to enable)'}{RESET}\n")

    all_results: list[BenchResult] = []

    run_all   = args.suite == "all"
    run_bus   = run_all or args.suite == "bus"
    run_agent = run_all or args.suite == "agent"
    run_orch  = run_all or args.suite == "orch"
    run_ep    = run_all or args.suite == "episode"
    run_cr    = run_all or args.suite == "crisis"
    run_llm   = (run_all and args.llm) or args.suite == "llm"
    run_stats = run_all or args.suite == "stats"

    if run_bus:
        _header("Suite 1 — Message Bus")
        all_results.extend(await bench_message_bus(iterations=500))

    if run_agent:
        _header("Suite 2 — Per-Agent Decision Latency")
        all_results.extend(await bench_agent_decisions(iterations=args.iterations))

    if run_orch:
        _header("Suite 3 — Orchestrator Step")
        all_results.extend(await bench_orchestrator_step(iterations=min(args.iterations, 30)))

    if run_ep:
        _header("Suite 4 — Full Episode")
        all_results.extend(await bench_episode(n_steps=args.steps))

    if run_cr:
        _header("Suite 5 — Crisis Scenarios")
        all_results.extend(await bench_crisis_scenarios(steps_per_crisis=args.steps // 2))

    if run_llm:
        _header("Suite 6 — LLM vs Mock")
        all_results.extend(await bench_llm_vs_mock(iterations=5))

    if run_stats:
        _header("Suite 7 — Message Bus Stats")
        await bench_bus_stats(n_steps=args.steps)

    if all_results:
        _print_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())
