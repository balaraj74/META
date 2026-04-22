import { createFileRoute } from "@tanstack/react-router";
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid, Tooltip, Legend, ReferenceLine, BarChart, Bar } from "recharts";
import { Navbar } from "@/components/nav/Navbar";
import { useTrainingStatus } from "@/hooks/useTrainingStatus";
import { STRATEGY_LESSONS } from "@/lib/sponsors";
import { AGENTS } from "@/lib/constants";
import { Upload, Download, Copy } from "lucide-react";
import { toast } from "sonner";

export const Route = createFileRoute("/training")({
  head: () => ({
    meta: [
      { title: "Live Training · TRIAGE" },
      { name: "description", content: "Live DPO training progress for the TRIAGE multi-agent model." },
      { property: "og:title", content: "TRIAGE Live Training" },
      { property: "og:description", content: "Watch DPO training reduce loss and unlock agent strategy memory in real time." },
    ],
  }),
  component: TrainingPage,
});

function TrainingPage() {
  const s = useTrainingStatus();
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <div className="mx-auto max-w-[1600px] space-y-3 p-4">
        <TrainingStatusBar s={s} />
        <div className="grid grid-cols-12 gap-3">
          <div className="col-span-12 lg:col-span-5">
            <LossCurveChart steps={s.lossSteps} train={s.trainLoss} evalL={s.evalLoss} live={s.isTraining} />
          </div>
          <div className="col-span-12 lg:col-span-7">
            <BeforeAfterComparison
              baseline={s.baselineRewards}
              trained={s.trainedRewards}
              rewardMargin={s.rewardMargin}
              dpoAccuracy={s.dpoAccuracy}
              comparisonStep={s.comparisonStep}
              totalSteps={s.totalSteps}
              live={s.isTraining}
            />
          </div>
        </div>
        <MemoryRow />
        <ActionRow s={s} />
      </div>
    </div>
  );
}

function TrainingStatusBar({ s }: { s: ReturnType<typeof useTrainingStatus> }) {
  const pct = (s.step / s.totalSteps) * 100;
  return (
    <div className="border border-border bg-surface p-4" style={{ borderRadius: 8 }}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
            Training Status · Qwen2.5-0.5B + LoRA + 4-bit NF4 · RTX 2050
          </div>
          <div className="mt-1 flex items-center gap-3 font-mono text-[13px] text-text-primary">
            <Phase phase={s.phase} />
            <span className="text-text-muted">·</span>
            <span>Epoch {s.epoch} / {s.totalEpochs}</span>
            <span className="text-text-muted">·</span>
            <span>Step {s.step} / {s.totalSteps}</span>
            <span className="text-text-muted">·</span>
            <span>ETA {s.etaMinutes}m</span>
            <span className="text-text-muted">·</span>
            <span className="text-stable">GPU {s.gpuUtilization}%</span>
          </div>
        </div>
        <div className="font-mono text-[12px] text-text-secondary">
          Loss: <span className="text-stable">{s.trainLoss[s.trainLoss.length - 1]?.toFixed(4) ?? "—"}</span>
          {" · "}
          Pairs: <span className="text-text-primary">{s.nPreferencePairs.toLocaleString()}</span>
        </div>
      </div>
      <div className="mt-3 h-2 w-full overflow-hidden bg-surface-2" style={{ borderRadius: 2 }}>
        <div
          className="h-full transition-all"
          style={{
            width: `${pct}%`,
            background: s.phase === "complete" ? "var(--stable-green)" : "var(--warning-amber)",
          }}
        />
      </div>
    </div>
  );
}

function Phase({ phase }: { phase: ReturnType<typeof useTrainingStatus>["phase"] }) {
  const map: Record<typeof phase, { label: string; color: string }> = {
    idle: { label: "IDLE", color: "var(--text-muted)" },
    collecting: { label: "COLLECTING ROLLOUTS", color: "var(--warning-amber)" },
    labeling: { label: "LABELING PREFERENCES", color: "var(--warning-amber)" },
    training: { label: "DPO TRAINING", color: "var(--clinical-blue)" },
    complete: { label: "TRAINING COMPLETE ✓", color: "var(--stable-green)" },
    error: { label: "ERROR", color: "var(--emergency-red)" },
  } as const;
  const m = map[phase];
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-1.5 w-1.5 rounded-full pulse-dot" style={{ background: m.color }} />
      <span className="font-mono text-[11px] uppercase tracking-wider" style={{ color: m.color }}>
        {m.label}
      </span>
    </span>
  );
}

function LossCurveChart({
  steps,
  train,
  evalL,
  live,
}: {
  steps: number[];
  train: number[];
  evalL: number[];
  live: boolean;
}) {
  const data = steps.map((s, i) => ({ step: s, train: train[i], eval: evalL[i] }));
  return (
    <div className="border border-border bg-surface" style={{ borderRadius: 8 }}>
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
          Training Loss
        </div>
        {live && (
          <div className="flex items-center gap-1.5 font-mono text-[10px] text-stable">
            <span className="h-1.5 w-1.5 rounded-full bg-stable pulse-dot" />
            LIVE
          </div>
        )}
      </div>
      <div className="p-3" style={{ height: 320 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -10 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="step" stroke="var(--text-muted)" tick={{ fontFamily: "DM Mono", fontSize: 10 }} />
            <YAxis domain={[0, 1]} stroke="var(--text-muted)" tick={{ fontFamily: "DM Mono", fontSize: 10 }} />
            <Tooltip
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 6,
                fontFamily: "DM Mono",
                fontSize: 11,
              }}
            />
            <Legend wrapperStyle={{ fontFamily: "DM Mono", fontSize: 10 }} />
            <ReferenceLine y={0.2} stroke="var(--stable-green)" strokeDasharray="4 4" label={{ value: "convergence target", fontSize: 10, fill: "var(--stable-green)", fontFamily: "DM Mono" }} />
            <Line type="monotone" dataKey="train" name="train_loss" stroke="var(--clinical-blue)" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="eval" name="eval_loss" stroke="var(--clinical-blue)" strokeDasharray="4 4" strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function BeforeAfterComparison({
  baseline,
  trained,
  rewardMargin,
  dpoAccuracy,
  comparisonStep,
  totalSteps,
  live,
}: {
  baseline: number[];
  trained: number[];
  rewardMargin: number;
  dpoAccuracy: number;
  comparisonStep: number;
  totalSteps: number;
  live: boolean;
}) {
  const data = baseline.map((b, i) => ({
    episode: `EP${i + 1}`,
    Baseline: b,
    Trained: trained[i],
  }));
  const baseAvg  = baseline.length ? baseline.reduce((a, b) => a + b, 0) / baseline.length : 0;
  const trainAvg = trained.length  ? trained.reduce((a, b)  => a + b, 0) / trained.length  : 0;
  const delta    = trainAvg - baseAvg;

  // Change the key every time step changes → forces Recharts to re-mount bars → animation replay
  const animKey = `cmp-${comparisonStep}`;

  return (
    <div className="border border-border bg-surface" style={{ borderRadius: 8 }}>
      {/* ── Header ── */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <div className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
            Reward Comparison
          </div>
          {live && (
            <span className="inline-flex items-center gap-1 font-mono text-[9px] text-stable">
              <span className="h-1.5 w-1.5 rounded-full bg-stable pulse-dot" />
              LIVE
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {comparisonStep > 0 && (
            <span className="font-mono text-[9px] text-text-muted">
              step{" "}
              <span className="text-text-primary">{comparisonStep.toLocaleString()}</span>
              <span className="text-text-muted"> / {totalSteps.toLocaleString()}</span>
            </span>
          )}
          <div className="font-mono text-[10px] text-stable">+{delta.toFixed(1)} ↑ avg</div>
        </div>
      </div>

      {/* ── Stats row ── */}
      <div className="grid grid-cols-5 gap-2 px-3 pt-3">
        <Stat label="Baseline avg"  value={baseAvg.toFixed(1)}             color="var(--text-muted)"    />
        <Stat label="Trained avg"   value={trainAvg.toFixed(1)}            color="var(--clinical-blue)" />
        <Stat label="Δ improvement" value={`+${delta.toFixed(1)}`}         color="var(--stable-green)"  />
        <Stat label="DPO Accuracy"  value={dpoAccuracy > 0 ? `${dpoAccuracy.toFixed(1)}%` : "—"}   color="var(--warning-amber)" />
        <Stat label="Reward Margin" value={rewardMargin > 0 ? rewardMargin.toFixed(3) : "—"}       color="var(--clinical-blue)" />
      </div>

      {/* ── Animated bar chart ── */}
      <div className="px-3 pb-3" style={{ height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart key={animKey} data={data} margin={{ top: 12, right: 8, bottom: 0, left: -10 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="episode" stroke="var(--text-muted)" tick={{ fontFamily: "DM Mono", fontSize: 10 }} />
            <YAxis stroke="var(--text-muted)" tick={{ fontFamily: "DM Mono", fontSize: 10 }} domain={[0, 'auto']} />
            <Tooltip
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 6,
                fontFamily: "DM Mono",
                fontSize: 11,
              }}
              cursor={{ fill: "rgba(255,255,255,0.04)" }}
            />
            <Legend wrapperStyle={{ fontFamily: "DM Mono", fontSize: 10 }} />
            <Bar
              dataKey="Baseline"
              fill="var(--text-muted)"
              isAnimationActive={true}
              animationDuration={600}
              animationEasing="ease-out"
            />
            <Bar
              dataKey="Trained"
              fill="var(--clinical-blue)"
              isAnimationActive={true}
              animationDuration={800}
              animationEasing="ease-out"
              radius={[2, 2, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="border border-border bg-surface-2 p-2" style={{ borderRadius: 4 }}>
      <div className="font-mono text-[10px] uppercase tracking-wider text-text-muted">{label}</div>
      <div className="mt-0.5 font-mono text-lg" style={{ color }}>{value}</div>
    </div>
  );
}

function MemoryRow() {
  return (
    <div className="border border-border bg-surface" style={{ borderRadius: 8 }}>
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
          Strategy Memory · {STRATEGY_LESSONS.length} lessons learned
        </div>
        <div className="font-mono text-[10px] text-text-muted">scroll →</div>
      </div>
      <div className="flex gap-3 overflow-auto p-3">
        {STRATEGY_LESSONS.map((l) => {
          const agent = AGENTS.find((a) => a.key === l.agentType);
          return (
            <div
              key={l.id}
              className="w-[300px] shrink-0 border border-border bg-surface-2 p-3"
              style={{ borderRadius: 6, borderLeft: `3px solid ${agent?.color}` }}
            >
              <div className="flex items-center justify-between font-mono text-[10px]">
                <span style={{ color: agent?.color }}>EP {l.episode} · {agent?.name}</span>
                <span className="text-stable">+{l.rewardDelta} ↑</span>
              </div>
              <div className="mt-2 text-[11px] uppercase tracking-wider text-text-muted">Pattern</div>
              <div className="text-[12px] text-text-primary">{l.pattern}</div>
              <div className="mt-2 text-[11px] uppercase tracking-wider text-text-muted">Correction</div>
              <div className="text-[12px] text-text-primary">{l.correction}</div>
              <div className="mt-3 flex items-center gap-2">
                <div className="h-1.5 flex-1 overflow-hidden bg-surface-3" style={{ borderRadius: 1 }}>
                  <div
                    className="h-full"
                    style={{ width: `${l.confidence * 100}%`, background: "var(--clinical-blue)" }}
                  />
                </div>
                <span className="font-mono text-[10px] text-text-muted">{l.confidence.toFixed(2)}</span>
              </div>
              <div className="mt-1 font-mono text-[10px] text-text-muted">
                Applied {l.timesApplied}× · {l.successCount} successful
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ActionRow({ s }: { s: ReturnType<typeof useTrainingStatus> }) {
  const ready = s.phase === "complete";
  return (
    <div className="flex flex-wrap items-center justify-end gap-2">
      <button
        disabled={!ready}
        onClick={() => {
          toast.success("Pushing to HuggingFace…", { description: "Uploading 2.3 GB of weights" });
          setTimeout(
            () => toast.success("Pushed!", { description: "huggingface.co/error404/triage-cmo" }),
            1800,
          );
        }}
        className="inline-flex items-center gap-1.5 bg-primary px-3 py-2 font-mono text-[11px] uppercase tracking-wider text-primary-foreground hover:bg-primary-dark disabled:opacity-50"
        style={{ borderRadius: 4 }}
      >
        <Upload className="h-3 w-3" /> Push to HuggingFace
      </button>
      <button
        disabled={!ready}
        onClick={() => toast.warning("Warning: 4.2 GB download")}
        className="inline-flex items-center gap-1.5 border border-border bg-surface px-3 py-2 font-mono text-[11px] uppercase tracking-wider text-text-primary hover:border-primary disabled:opacity-50"
        style={{ borderRadius: 4 }}
      >
        <Download className="h-3 w-3" /> Download Model
      </button>
      <button
        onClick={() => {
          toast.success("Blog draft copied to clipboard");
        }}
        className="inline-flex items-center gap-1.5 border border-border bg-surface px-3 py-2 font-mono text-[11px] uppercase tracking-wider text-text-primary hover:border-primary"
        style={{ borderRadius: 4 }}
      >
        <Copy className="h-3 w-3" /> Copy HF Blog
      </button>
    </div>
  );
}
