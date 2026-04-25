FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System deps — build-essential provides gcc which Triton JIT requires
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Triton JIT needs CC set explicitly
ENV CC=gcc
# Disable torch.compile/dynamo — Unsloth's manual patches already provide speedup
ENV TORCHDYNAMO_DISABLE=1
# Reduce CUDA memory fragmentation
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

# Install core deps first (cached layer)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install unsloth from PyPI (latest stable)
RUN pip3 install --no-cache-dir "unsloth==2026.4.8"

# Copy training script
COPY train_grpo_hf.py .

# Entry — train on startup
# Memory budget: L40S 44GB
#   9B 4-bit ≈ 6GB, LoRA ≈ 0.2GB, KV-cache + GRPO gen ≈ rest
#   num-gen=4 (Unsloth sets batch=num_gen), grad-accum=8 → effective batch=32
CMD ["python3", "train_grpo_hf.py", \
     "--model", "unsloth/Qwen3.5-9B", \
     "--dataset", "balarajr/triage-grpo", \
     "--hub-model", "balarajr/triage-agent-9b", \
     "--push", \
     "--augment-hf", \
     "--augment-max", "200", \
     "--epochs", "3", \
     "--batch-size", "2", \
     "--grad-accum", "8", \
     "--num-gen", "4", \
     "--lr", "5e-6"]
