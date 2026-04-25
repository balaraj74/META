#!/usr/bin/env python3
"""Convert kaggle_grpo_full.py into a proper .ipynb notebook."""
import json, re

with open("/home/balaraj/META final/triage-backend/notebooks/kaggle_grpo_full.py") as f:
    py = f.read()

# Split on Cell separators
parts = re.split(r'# ═+\n# Cell \d+: (.+?)\n# ═+\n', py)

cells = [{"cell_type":"markdown","metadata":{},"source":[
    "# 🏥 TRIAGE — GRPO Training (Full Dataset Edition)\n",
    "**Model:** Qwen3.5-4B | **Method:** GRPO with 9 reward verifiers | **Data:** 21 HF sources\n\n",
    "Compatible with Kaggle T4/P100 GPUs (no Unsloth required).\n",
    "Uses `peft` + `bitsandbytes` for 4-bit QLoRA training."
]}]

# parts[0] = header before first cell
# parts[1] = "Install Dependencies", parts[2] = code for it
# parts[3] = "Imports + Config", parts[4] = code for it, etc.

i = 1
while i < len(parts):
    title = parts[i].strip() if i < len(parts) else ""
    code = parts[i+1].strip() if i+1 < len(parts) else ""
    i += 2
    if not code:
        continue

    # Add markdown header
    cells.append({"cell_type":"markdown","metadata":{},"source":[f"## {title}\n"]})

    # Handle pip install line
    lines = []
    for line in code.split("\n"):
        if line.startswith("# !pip"):
            lines.append(line[2:])  # uncomment
        else:
            lines.append(line)

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["\n".join(lines) + "\n"]
    })

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name":"Python 3","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out = "/home/balaraj/META final/triage-backend/notebooks/triage_grpo_kaggle_full.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Generated {out}")
print(f"   {len(cells)} cells")
