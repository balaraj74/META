FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy training script
COPY train_grpo_hf.py .

# Entry — train on startup
CMD ["python3", "train_grpo_hf.py", \
     "--model", "unsloth/Qwen3.6-27B-bnb-4bit", \
     "--dataset", "balarajr/triage-grpo", \
     "--hub-model", "balarajr/triage-agent-27b", \
     "--push", \
     "--augment-hf", \
     "--augment-max", "200", \
     "--epochs", "3", \
     "--batch-size", "1", \
     "--grad-accum", "4", \
     "--num-gen", "4", \
     "--lr", "5e-6"]
