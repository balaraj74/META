import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def detect_repo_root() -> Path:
    candidates = [
        Path("/content/META final"),
        Path("/content/drive/MyDrive/META final"),
        Path("/content/drive/MyDrive/META_final"),
        Path.cwd(),
        Path.cwd().parent,
    ]
    for candidate in candidates:
        if (candidate / "triage-backend").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the repo root. Clone or copy the repo into Colab first, "
        "or update detect_repo_root() with the correct path."
    )


REPO_ROOT = detect_repo_root()
DATASET_PATHS = [
    REPO_ROOT / "triage-backend/data/full_training/hf_dpo_pairs.jsonl",
    REPO_ROOT / "triage-backend/data/full_training/dpo_pairs.jsonl",
    REPO_ROOT / "triage-backend/data/full_training/healthcare_dpo.jsonl",
    REPO_ROOT / "triage-backend/data/demo/dpo_pairs.jsonl",
]
MODEL_ID = "google/gemma-4-8b-it"
OUTPUT_DIR = Path("/content/triage-dpo-gemma")
MAX_PROMPT_CHARS = 1600
MAX_RESPONSE_CHARS = 900
MAX_SAMPLES = None
SEED = 42


def normalize_text(value, max_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = " ".join(text.split()).strip()
    return text[:max_chars]


def load_all_datasets(paths: list[Path], max_samples: int | None = None) -> tuple[Dataset, dict[str, int]]:
    rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    seen: set[tuple[str, str, str]] = set()

    for path in paths:
        if not path.exists():
            print(f"Skipping missing dataset: {path}")
            continue

        added = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not all(key in row for key in ("prompt", "chosen", "rejected")):
                    continue

                prompt = normalize_text(row["prompt"], MAX_PROMPT_CHARS)
                chosen = normalize_text(row["chosen"], MAX_RESPONSE_CHARS)
                rejected = normalize_text(row["rejected"], MAX_RESPONSE_CHARS)

                if not prompt or not chosen or not rejected or chosen == rejected:
                    continue

                key = (prompt, chosen, rejected)
                if key in seen:
                    continue
                seen.add(key)

                rows.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source_file": path.name,
                })
                added += 1

        counts[path.name] = added

    if not rows:
        raise RuntimeError("No valid DPO rows found. Check the dataset paths above.")

    random.Random(SEED).shuffle(rows)
    if max_samples is not None:
        rows = rows[:max_samples]

    return Dataset.from_list(rows), counts


dataset, dataset_counts = load_all_datasets(DATASET_PATHS, max_samples=MAX_SAMPLES)
dataset_split = dataset.train_test_split(test_size=0.05, seed=SEED)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

print("Loaded combined DPO dataset:")
for name, count in dataset_counts.items():
    print(f"  - {name}: {count:,} pairs")
print(f"  - train: {len(train_dataset):,}")
print(f"  - eval:  {len(eval_dataset):,}")

compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    else torch.float16
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

print(f"Loading {MODEL_ID} in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=compute_dtype,
    use_cache=False,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = DPOConfig(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    max_length=768,
    max_prompt_length=384,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False,
    fp16=compute_dtype == torch.float16,
    bf16=compute_dtype == torch.bfloat16,
)

trainer_kwargs = {
    "model": model,
    "ref_model": None,
    "args": training_args,
    "beta": 0.1,
    "train_dataset": train_dataset,
    "eval_dataset": eval_dataset,
    "max_length": 768,
    "max_prompt_length": 384,
}

try:
    trainer = DPOTrainer(
        processing_class=tokenizer,
        **trainer_kwargs,
    )
except TypeError:
    trainer = DPOTrainer(
        tokenizer=tokenizer,
        **trainer_kwargs,
    )

print("Initialization complete. Starting DPO training...")
trainer.train()

final_dir = OUTPUT_DIR / "final_adapter"
trainer.model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Training complete. Adapter saved to {final_dir}")
