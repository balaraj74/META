import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def detect_repo_root() -> Path:
    env_root = os.environ.get("TRIAGE_REPO_ROOT")
    if env_root and (Path(env_root) / "triage-backend").exists():
        return Path(env_root)

    candidates = [
        Path("/content/META final"),
        Path("/content/drive/MyDrive/META final"),
        Path("/content/drive/MyDrive/META_final"),
        Path("/content/drive/MyDrive"),
        Path("/content"),
        Path.cwd(),
        Path.cwd().parent,
    ]
    for candidate in candidates:
        if (candidate / "triage-backend").exists():
            return candidate

    search_roots = [Path("/content"), Path("/content/drive/MyDrive"), Path.cwd()]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for match in search_root.rglob("triage-backend"):
            if match.is_dir():
                return match.parent

    raise FileNotFoundError("Could not find the TRIAGE repo root.")


def resolve_dataset_paths(repo_root: Path | None) -> list[Path]:
    preferred: list[Path] = []
    if repo_root is not None:
        preferred.extend([
            repo_root / "triage-backend/data/full_training/hf_dpo_pairs.jsonl",
            repo_root / "triage-backend/data/full_training/dpo_pairs.jsonl",
            repo_root / "triage-backend/data/full_training/healthcare_dpo.jsonl",
            repo_root / "triage-backend/data/demo/dpo_pairs.jsonl",
        ])

    env_data_root = os.environ.get("TRIAGE_DATA_ROOT")
    if env_data_root:
        data_root = Path(env_data_root)
        preferred.extend([
            data_root / "full_training/hf_dpo_pairs.jsonl",
            data_root / "full_training/dpo_pairs.jsonl",
            data_root / "full_training/healthcare_dpo.jsonl",
            data_root / "demo/dpo_pairs.jsonl",
        ])

    resolved: list[Path] = [path for path in preferred if path.exists()]
    if resolved:
        return resolved

    filenames = {
        "hf_dpo_pairs.jsonl",
        "dpo_pairs.jsonl",
        "healthcare_dpo.jsonl",
    }
    fallback_matches: list[Path] = []
    search_roots = [Path("/content"), Path("/content/drive/MyDrive"), Path.cwd()]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for match in search_root.rglob("*.jsonl"):
            if match.name in filenames or (
                match.name == "dpo_pairs.jsonl"
                and any(part in {"demo", "full_training"} for part in match.parts)
            ):
                fallback_matches.append(match)

    deduped: list[Path] = []
    seen = set()
    for match in fallback_matches:
        if match in seen:
            continue
        seen.add(match)
        deduped.append(match)
    return deduped


def maybe_upload_datasets(existing_paths: list[Path]) -> list[Path]:
    if existing_paths:
        return existing_paths

    try:
        from google.colab import files  # type: ignore[import-not-found]
    except ImportError:
        return existing_paths

    print(
        "No repo or dataset files were found automatically.\n"
        "Upload one or more DPO JSONL files now:\n"
        "  - hf_dpo_pairs.jsonl\n"
        "  - dpo_pairs.jsonl\n"
        "  - healthcare_dpo.jsonl\n"
        "  - any other JSONL with prompt/chosen/rejected fields"
    )
    uploaded = files.upload()
    if not uploaded:
        return existing_paths

    upload_dir = Path("/content/triage_uploaded_datasets")
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for name, data in uploaded.items():
        target = upload_dir / name
        target.write_bytes(data)
        saved_paths.append(target)

    print("Uploaded dataset files:")
    for path in saved_paths:
        print(f"  - {path}")
    return saved_paths


try:
    REPO_ROOT = detect_repo_root()
    print(f"Detected repo root: {REPO_ROOT}")
except FileNotFoundError:
    REPO_ROOT = None
    print("Repo root not found. Falling back to dataset file discovery.")

DATASET_PATHS = resolve_dataset_paths(REPO_ROOT)
DATASET_PATHS = maybe_upload_datasets(DATASET_PATHS)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
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
        available = "\n".join(f"  - {path}" for path in paths) or "  - none"
        raise RuntimeError(
            "No valid DPO rows found.\n"
            "Looked at:\n"
            f"{available}\n\n"
            "Fix by either:\n"
            "1. cloning/copying the whole repo into /content or Drive, or\n"
            "2. setting TRIAGE_REPO_ROOT or TRIAGE_DATA_ROOT, or\n"
            "3. uploading the JSONL files directly into Colab."
        )

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
tokenizer.padding_side = "left"

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
    beta=0.1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    max_length=768,
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
        **trainer_kwargs,
    )

print("Initialization complete. Starting DPO training...")
trainer.train()

final_dir = OUTPUT_DIR / "final_adapter"
trainer.model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Training complete. Adapter saved to {final_dir}")
