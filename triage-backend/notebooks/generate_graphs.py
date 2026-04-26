#!/usr/bin/env python3
"""
TRIAGE GRPO — Publication-Quality Visualization Suite
ALL data sourced from real files:
  - trainer_state.json  (training telemetry)
  - bench.json          (benchmark results)
Zero hardcoded mock values.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json, os, sys

# ── Paths ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT    = os.path.join(SCRIPT_DIR, "..")
TRAINER_STATE = os.path.join(PROJECT, "models", "triage_grpo_output",
                             "checkpoint-75", "trainer_state.json")
BENCH_JSON    = os.path.join(PROJECT, "results", "bench.json")
OUT           = os.path.join(PROJECT, "results", "graphs")
os.makedirs(OUT, exist_ok=True)

# ── Dark premium theme ────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
    "axes.grid": True, "grid.color": "#21262d", "grid.alpha": 0.6,
    "text.color": "#c9d1d9", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "font.family": "sans-serif",
    "font.size": 11, "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d", "legend.fontsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": "#0d1117",
})
CYAN="#58a6ff"; GREEN="#3fb950"; ORANGE="#d29922"; RED="#f85149"
PURPLE="#bc8cff"; PINK="#f778ba"; TEAL="#39d353"; WHITE="#c9d1d9"

# ══════════════════════════════════════════════════════════════
# LOAD REAL DATA
# ══════════════════════════════════════════════════════════════
def load_trainer_state():
    with open(TRAINER_STATE) as f:
        ts = json.load(f)
    logs = ts["log_history"]
    return {
        "steps":     [e["step"] for e in logs],
        "rewards":   [e["reward"] for e in logs],
        "entropy":   [e["entropy"] for e in logs],
        "grad_norm": [e["grad_norm"] for e in logs],
        "lr":        [e["learning_rate"] for e in logs],
        "reward_std":[e["reward_std"] for e in logs],
        "epochs":    [e["epoch"] for e in logs],
        "loss":      [e.get("loss", 0) for e in logs],
        "global_step": ts["global_step"],
        "total_tokens": ts.get("num_input_tokens_seen", 0),
    }

def load_bench():
    with open(BENCH_JSON) as f:
        b = json.load(f)
    return b

print("Loading trainer_state.json ...", end=" ")
T = load_trainer_state()
print(f"OK ({len(T['steps'])} log entries, {T['global_step']} steps)")
print("Loading bench.json ...", end=" ")
B = load_bench()
print(f"OK ({len(B['scenarios'])} scenarios)")

# ── Extract per-episode and per-agent stats from bench.json ──
all_episodes = []
for sc in B["scenarios"]:
    for ep in sc["episodes"]:
        all_episodes.append(ep)

agent_stats = {}  # agent_name -> {actions, correct, latencies}
for ep in all_episodes:
    for aname, adata in ep["agents"].items():
        if aname not in agent_stats:
            agent_stats[aname] = {"actions": 0, "correct": 0, "latencies": []}
        agent_stats[aname]["actions"] += adata["actions_taken"]
        agent_stats[aname]["correct"] += adata["correct_actions"]
        if adata["mean_latency_ms"] > 0:
            agent_stats[aname]["latencies"].append(adata["mean_latency_ms"])

# ══════════════════════════════════════════════════════════════
# GRAPH 1: GRPO Training Reward Curve
# ══════════════════════════════════════════════════════════════
def plot_reward_curve():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    steps, rews, rstd = T["steps"], T["rewards"], T["reward_std"]
    ax.fill_between(steps, [r-s for r,s in zip(rews,rstd)],
                    [r+s for r,s in zip(rews,rstd)], alpha=0.2, color=CYAN)
    ax.plot(steps, rews, color=CYAN, lw=2.5, marker="o", ms=7,
            label="Mean Reward", zorder=5)
    ax.axhline(y=rews[0], color=RED, ls="--", alpha=0.5,
               label=f"Baseline (Step {steps[0]}): {rews[0]:.3f}")
    ax.axhline(y=rews[-1], color=GREEN, ls="--", alpha=0.5,
               label=f"Final (Step {steps[-1]}): {rews[-1]:.3f}")
    ax.set_xlabel("Training Step", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Reward (0-1)", fontsize=13, fontweight="bold")
    ax.set_title("GRPO Training Reward Progression", fontsize=16,
                 fontweight="bold", color=WHITE, pad=15)
    ax.legend(loc="lower right")
    delta = ((rews[-1] - rews[0]) / max(rews[0], 1e-9)) * 100
    ax.annotate(f"+{delta:.0f}% improvement", xy=(steps[-1], rews[-1]),
                xytext=(steps[-3], rews[-1]+0.04), fontsize=11, color=GREEN,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
    fig.savefig(os.path.join(OUT, "01_reward_curve.png")); plt.close(fig)
    print("  [ok] 01_reward_curve.png")

# ══════════════════════════════════════════════════════════════
# GRAPH 2: Training Metrics Dashboard (4-panel)
# ══════════════════════════════════════════════════════════════
def plot_training_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GRPO Training Metrics Dashboard", fontsize=16,
                 fontweight="bold", color=WHITE, y=0.98)
    s = T["steps"]
    axes[0,0].plot(s, T["rewards"], color=CYAN, lw=2, marker="o", ms=5)
    axes[0,0].set_title("Mean Reward", color=GREEN, fontweight="bold")
    axes[0,0].set_ylabel("Reward")
    axes[0,1].plot(s, T["entropy"], color=PURPLE, lw=2, marker="s", ms=5)
    axes[0,1].set_title("Policy Entropy", color=PURPLE, fontweight="bold")
    axes[0,1].set_ylabel("Entropy")
    axes[1,0].plot(s, T["grad_norm"], color=ORANGE, lw=2, marker="^", ms=5)
    axes[1,0].set_title("Gradient Norm", color=ORANGE, fontweight="bold")
    axes[1,0].set_xlabel("Training Step"); axes[1,0].set_ylabel("Grad Norm")
    axes[1,1].plot(s, [lr*1e5 for lr in T["lr"]], color=PINK, lw=2, marker="D", ms=5)
    axes[1,1].set_title("Learning Rate (x1e-5)", color=PINK, fontweight="bold")
    axes[1,1].set_xlabel("Training Step"); axes[1,1].set_ylabel("LR x1e-5")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "02_training_dashboard.png")); plt.close(fig)
    print("  [ok] 02_training_dashboard.png")

# ══════════════════════════════════════════════════════════════
# GRAPH 3: Per-Scenario Performance (from bench.json)
# ══════════════════════════════════════════════════════════════
def plot_scenario_bars():
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [sc["scenario"] for sc in B["scenarios"]]
    survivals = [sc["mean_survival"] for sc in B["scenarios"]]
    rewards = [sc["mean_reward"] for sc in B["scenarios"]]
    violations = [sc["mean_violation_detection"] for sc in B["scenarios"]]
    x = np.arange(len(names)); w = 0.25
    ax.bar(x - w, survivals, w, label="Survival Rate", color=GREEN, alpha=0.85)
    ax.bar(x, violations, w, label="Violation Detection", color=CYAN, alpha=0.85)
    ax.bar(x + w, [r/10 for r in rewards], w, label="Reward (norm /10)", color=PURPLE, alpha=0.85)
    for i in range(len(names)):
        ax.text(x[i]-w, survivals[i]+0.02, f"{survivals[i]*100:.0f}%", ha="center", fontsize=8, color=GREEN)
        ax.text(x[i], violations[i]+0.02, f"{violations[i]*100:.0f}%", ha="center", fontsize=8, color=CYAN)
        ax.text(x[i]+w, rewards[i]/10+0.02, f"{rewards[i]:.1f}", ha="center", fontsize=8, color=PURPLE)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Score (0-1)", fontsize=13, fontweight="bold")
    ax.set_title("Benchmark: Per-Scenario Performance (from bench.json)",
                 fontsize=15, fontweight="bold", color=WHITE, pad=15)
    ax.set_ylim(0, 1.2); ax.legend(fontsize=11)
    fig.savefig(os.path.join(OUT, "03_scenario_performance.png")); plt.close(fig)
    print("  [ok] 03_scenario_performance.png")

# ══════════════════════════════════════════════════════════════
# GRAPH 4: Scenario Heatmap (from bench.json)
# ══════════════════════════════════════════════════════════════
def plot_scenario_heatmap():
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [sc["scenario"] for sc in B["scenarios"]]
    metrics = ["Survival Rate", "Violation Detection", "Episode Reward"]
    data = np.array([
        [sc["mean_survival"] for sc in B["scenarios"]],
        [sc["mean_violation_detection"] for sc in B["scenarios"]],
        [sc["mean_reward"]/10 for sc in B["scenarios"]],
    ])
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0.5, vmax=1.05)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(metrics, fontsize=11)
    for i in range(3):
        for j in range(len(names)):
            v = data[i,j]
            d = f"{v*100:.0f}%" if i < 2 else f"{v*10:.1f}"
            ax.text(j, i, d, ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#0d1117")
    ax.set_title("Scenario-Level Performance Matrix (bench.json)",
                 fontsize=15, fontweight="bold", color=WHITE, pad=15)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(colors="#8b949e")
    fig.savefig(os.path.join(OUT, "04_scenario_heatmap.png")); plt.close(fig)
    print("  [ok] 04_scenario_heatmap.png")

# ══════════════════════════════════════════════════════════════
# GRAPH 5: Per-Episode Reward Distribution (from bench.json)
# ══════════════════════════════════════════════════════════════
def plot_episode_rewards():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ep_rewards = [ep["reward"] for ep in all_episodes]
    ep_labels = [f"{ep['scenario'][:4]}-E{ep['episode']}" for ep in all_episodes]
    colors = [GREEN if r >= 9 else (ORANGE if r >= 7 else RED) for r in ep_rewards]
    bars = ax.bar(range(len(ep_rewards)), ep_rewards, color=colors, alpha=0.85,
                  edgecolor="#30363d", width=0.7)
    mean_r = np.mean(ep_rewards)
    ax.axhline(y=mean_r, color=CYAN, ls="--", lw=1.5,
               label=f"Mean: {mean_r:.2f}")
    ax.set_xticks(range(len(ep_rewards)))
    ax.set_xticklabels(ep_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Episode Reward", fontsize=13, fontweight="bold")
    ax.set_title(f"Per-Episode Reward Distribution ({len(all_episodes)} episodes, bench.json)",
                 fontsize=14, fontweight="bold", color=WHITE, pad=15)
    ax.legend(fontsize=11)
    fig.savefig(os.path.join(OUT, "05_episode_rewards.png")); plt.close(fig)
    print("  [ok] 05_episode_rewards.png")

# ══════════════════════════════════════════════════════════════
# GRAPH 6: Agent Action Accuracy (from bench.json)
# ══════════════════════════════════════════════════════════════
def plot_agent_accuracy():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    DISPLAY = {"er_triage":"ER Triage","icu_management":"ICU Mgmt",
               "pharmacy":"Pharmacy","cmo_oversight":"CMO Oversight",
               "hr_rostering":"HR Rostering","it_systems":"IT Systems"}
    COLORS = [CYAN, GREEN, PURPLE, ORANGE, PINK, TEAL]
    agents_ordered = ["er_triage","icu_management","pharmacy",
                      "cmo_oversight","hr_rostering","it_systems"]
    names, accs, acts = [], [], []
    for i, a in enumerate(agents_ordered):
        s = agent_stats.get(a, {"actions":0,"correct":0})
        names.append(DISPLAY.get(a, a))
        acts.append(s["actions"])
        accs.append(s["correct"]/max(s["actions"],1))
    bars = ax.bar(names, accs, color=COLORS[:len(names)], alpha=0.85,
                  edgecolor="#30363d", width=0.6)
    for bar, acc, act in zip(bars, accs, acts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{acc*100:.0f}%\n({act} acts)", ha="center", va="bottom",
                fontsize=9, color=WHITE, fontweight="bold")
    ax.set_ylabel("Action Accuracy", fontsize=13, fontweight="bold")
    ax.set_title("Per-Agent Decision Accuracy (bench.json)",
                 fontsize=15, fontweight="bold", color=WHITE, pad=15)
    ax.set_ylim(0, 1.25)
    ax.axhline(y=0.95, color=GREEN, ls="--", alpha=0.4, label="95% Target")
    ax.legend()
    fig.savefig(os.path.join(OUT, "06_agent_accuracy.png")); plt.close(fig)
    print("  [ok] 06_agent_accuracy.png")

# ══════════════════════════════════════════════════════════════
# GRAPH 7: Radar Chart — Multi-Metric (from bench.json)
# ══════════════════════════════════════════════════════════════
def plot_radar():
    labels = [sc["scenario"] for sc in B["scenarios"]]
    N = len(labels)
    surv = [sc["mean_survival"] for sc in B["scenarios"]]
    viol = [sc["mean_violation_detection"] for sc in B["scenarios"]]
    rew  = [sc["mean_reward"]/10 for sc in B["scenarios"]]
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    surv_c = surv + [surv[0]]; viol_c = viol + [viol[0]]
    rew_c = rew + [rew[0]]; angles_c = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#161b22"); fig.patch.set_facecolor("#0d1117")
    ax.fill(angles_c, surv_c, alpha=0.15, color=GREEN)
    ax.plot(angles_c, surv_c, color=GREEN, lw=2, label="Survival")
    ax.fill(angles_c, viol_c, alpha=0.15, color=CYAN)
    ax.plot(angles_c, viol_c, color=CYAN, lw=2, label="Violation Det.")
    ax.fill(angles_c, rew_c, alpha=0.15, color=PURPLE)
    ax.plot(angles_c, rew_c, color=PURPLE, lw=2, label="Reward (norm)")
    ax.set_xticks(angles); ax.set_xticklabels(labels, fontsize=9, color=WHITE)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8, color="#8b949e")
    ax.set_title("Multi-Metric Radar (bench.json)", fontsize=15,
                 fontweight="bold", color=WHITE, pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.savefig(os.path.join(OUT, "07_radar_chart.png")); plt.close(fig)
    print("  [ok] 07_radar_chart.png")

# ══════════════════════════════════════════════════════════════
# GRAPH 8: Training Loss + Reward Overlay
# ══════════════════════════════════════════════════════════════
def plot_loss_reward():
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    s = T["steps"]
    ax1.plot(s, T["rewards"], color=CYAN, lw=2.5, marker="o", ms=6, label="Reward")
    ax1.set_xlabel("Training Step", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Mean Reward", fontsize=13, fontweight="bold", color=CYAN)
    ax1.tick_params(axis="y", labelcolor=CYAN)
    ax2 = ax1.twinx()
    ax2.plot(s, T["reward_std"], color=ORANGE, lw=2, marker="s", ms=5,
             alpha=0.8, label="Reward Std")
    ax2.set_ylabel("Reward Std Dev", fontsize=13, fontweight="bold", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper left")
    ax1.set_title("Reward Convergence + Variance (trainer_state.json)",
                  fontsize=15, fontweight="bold", color=WHITE, pad=15)
    fig.savefig(os.path.join(OUT, "08_loss_reward.png")); plt.close(fig)
    print("  [ok] 08_loss_reward.png")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  TRIAGE GRPO — Generating Publication Graphs (REAL DATA)")
    print("=" * 60)
    plot_reward_curve()
    plot_training_dashboard()
    plot_scenario_bars()
    plot_scenario_heatmap()
    plot_episode_rewards()
    plot_agent_accuracy()
    plot_radar()
    plot_loss_reward()
    print(f"\n  All 8 graphs saved -> {OUT}/")
    print("  DATA SOURCES:")
    print(f"    trainer_state.json: {TRAINER_STATE}")
    print(f"    bench.json:         {BENCH_JSON}")
    print("=" * 60)
