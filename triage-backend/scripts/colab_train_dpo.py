"""
Colab DPO Trainer (Free Tier Edition - T4 GPU)
-----------------------------------------------
Copy and paste this into Colab Cell 3!
This takes full advantage of the free 15 GB VRAM available on the Colab T4 GPU.
"""

import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model

def load_data(file_path):
    print("Loading large dataset...")
    data = {"prompt": [], "chosen": [], "rejected": []}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            data["prompt"].append(row["prompt"])
            data["chosen"].append(row["chosen"])
            data["rejected"].append(row["rejected"])
    return Dataset.from_dict(data)

def main():
    # Load the massive dataset we generated in Cell 2
    dataset = load_data("large_triage_dpo.jsonl")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # We can use a smarter model now because we have 15 GB VRAM!
    model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
    print(f"Loading {model_name}...")

    # Load Base Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # LoRA Config (Injects trainable weights)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # DPO Config specifically tuned for Colab T4 GPU (15GB)
    dpo_config = DPOConfig(
        output_dir="./triage_dpo_colab_output",
        num_train_epochs=3,                     # 3 rounds over thousands of examples
        per_device_train_batch_size=4,          # Huge jump! (Laptop was 1)
        gradient_accumulation_steps=4,          # Effective batch size = 16
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=0.1,
        max_length=512,                         # More context window for bigger cases
        fp16=True,                              # T4 completely supports fp16 natively
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=256
    )

    print("\n🚀 Starting High-Speed Colab Training...")
    trainer.train()

    print("\n✅ Training Complete. Merging weights...")
    
    # Merge and save the final model right in Colab
    # You can zip this folder and download it back to your laptop!
    trainer.model.save_pretrained("./triage_dpo_colab_output/final_adapter")
    tokenizer.save_pretrained("./triage_dpo_colab_output/final_adapter")
    print("FINISHED! Files saved to ./triage_dpo_colab_output/final_adapter")

if __name__ == "__main__":
    main()
