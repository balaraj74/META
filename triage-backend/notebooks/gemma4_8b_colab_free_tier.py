import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
from datasets import load_dataset # Assuming dataset loading

# 1. Choose Gemma 4 8B (or equivalent like Gemma 2 9B depending on Exact HuggingFace Repo Name)
model_id = "google/gemma-4-8b-it" 

# 2. Crucial: The 4-Bit Quantization Config (Saves ~10GB of VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # Uses newer memory efficient types
)

print(f"Loading {model_id} in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Gemma models usually require adding EOS token explicitly
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False # Crucial for saving memory during training
)

# Prepare model for 4-bit training
model = prepare_model_for_kbit_training(model)

# 3. LoRA Config: Only train ~2% of the parameters
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. Load your dataset (assuming it has 'prompt', 'chosen', and 'rejected' columns)
# dataset = load_dataset("your_dpo_triage_dataset")

# 5. Training Arguments designed explicitly to heavily restrict GPU Usage
training_args = TrainingArguments(
    output_dir="./triage-dpo-gemma",
    per_device_train_batch_size=1,       # MUST BE 1 
    gradient_accumulation_steps=4,       # Simulates a batch size of 4
    optim="paged_adamw_32bit",           # Offloads optimizer states to CPU RAM if needed
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-5,
    bf16=True,                           # Faster and lighter than FP32
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# 6. Initialize DPO Trainer
# Ensure dataset is loaded before trainer initialization 
# trainer = DPOTrainer(
#     model=model,
#     ref_model=None, # Peft automatically handles reference when target is quantized
#     args=training_args,
#     beta=0.1,
#     train_dataset=dataset['train'], 
#     tokenizer=tokenizer,
#     max_length=1024,        # Force short sequences to prevent memory spiking
#     max_prompt_length=512,  # Keep prompts concise
# )

print("Initialization complete! Ready to start DPO training on Colab Free Tier.")
# trainer.train()
