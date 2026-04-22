import type { TrainingStatus } from "./types";

const TOTAL_STEPS = 1200;
const TOTAL_EPOCHS = 3;

function lossAt(step: number): number {
  // smooth exponential decay 0.82 -> 0.14 with mild noise
  const t = step / TOTAL_STEPS;
  const base = 0.82 * Math.exp(-2.6 * t) + 0.13;
  const noise = (Math.sin(step * 0.37) + Math.cos(step * 0.19)) * 0.012;
  return Math.max(0.08, base + noise);
}

export function makeTrainingStatus(currentStep: number): TrainingStatus {
  const step = Math.min(currentStep, TOTAL_STEPS);
  const sample = 25;
  const lossSteps: number[] = [];
  const trainLoss: number[] = [];
  const evalLoss: number[] = [];
  for (let s = 0; s <= step; s += sample) {
    lossSteps.push(s);
    trainLoss.push(+lossAt(s).toFixed(4));
    evalLoss.push(+(lossAt(s) + 0.04 + Math.sin(s * 0.05) * 0.01).toFixed(4));
  }
  const epoch = Math.min(TOTAL_EPOCHS, Math.floor((step / TOTAL_STEPS) * TOTAL_EPOCHS) + 1);
  const remaining = TOTAL_STEPS - step;
  const phase: TrainingStatus["phase"] =
    step === 0
      ? "idle"
      : step < 100
        ? "collecting"
        : step < 220
          ? "labeling"
          : step >= TOTAL_STEPS
            ? "complete"
            : "training";
  return {
    isTraining: step > 0 && step < TOTAL_STEPS,
    phase,
    epoch,
    totalEpochs: TOTAL_EPOCHS,
    step,
    totalSteps: TOTAL_STEPS,
    trainLoss,
    evalLoss,
    lossSteps,
    etaMinutes: Math.max(0, Math.round((remaining / TOTAL_STEPS) * 32)),
    gpuUtilization: 78 + Math.round(Math.sin(step * 0.07) * 8),
    baselineRewards: [42, 44, 41, 45, 43, 46, 44, 45, 43, 47],
    trainedRewards: [45, 52, 58, 63, 69, 74, 78, 82, 85, 87],
    rewardMargin: 17.4,
    dpoAccuracy: 97.5,
    comparisonStep: step,
    nPreferencePairs: 4720,
    modelPushedToHub: step >= TOTAL_STEPS,
    hubUrl: step >= TOTAL_STEPS ? "https://huggingface.co/error404/triage-cmo" : null,
  };
}
