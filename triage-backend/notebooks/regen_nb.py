import json, re

cells = []

def add_md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [src]})

def add_code(src):
    cells.append({"cell_type": "code", "metadata": {}, "source": src.split("\n"), "outputs": [], "execution_count": None})

add_md("# TRIAGE GRPO Benchmark - Kaggle GPU Edition\n\nQwen3.5-4B + GRPO LoRA, 9 reward verifiers, real inference.")
add_code("!pip install -q git+https://github.com/huggingface/transformers.git peft accelerate huggingface_hub")

with open("kaggle_benchmark.py") as f:
    lines = f.readlines()

code = "".join(l for l in lines if not l.startswith("#!/") and not l.startswith("# !pip"))
parts = re.split(r"# Cell \d+:", code)

for part in parts:
    s = part.strip()
    s = "\n".join(l for l in s.split("\n") if not re.match(r"^# [=]+", l))
    s = s.strip()
    if not s:
        continue
    fl = s.split("\n")
    title = fl[0].strip().strip("# ").strip()
    body = "\n".join(fl[1:]).strip() if len(fl) > 1 else s
    if body:
        add_md("## " + title)
        add_code(body)

add_md("## Execute Benchmark")
run_cell = (
    'import time as _t\n'
    'start = _t.time()\n'
    'from pathlib import Path\n'
    'adapter = CFG["adapter"]\n'
    'if not Path(adapter).exists():\n'
    '    alt = Path("/kaggle/working/grpo_output")\n'
    '    adapter = str(alt) if alt.exists() else None\n'
    '    if not adapter: print("No adapter - benchmarking BASE model")\n'
    'model, tokenizer = load_model(adapter_path=adapter)\n'
    'results = run_benchmark(model, tokenizer, num_scenarios=CFG["num_scenarios"])\n'
    'print(f"Total: {(_t.time()-start)/60:.1f} min")'
)
add_code(run_cell)

nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {"accelerator": "gpu", "dockerImageVersionId": 30919, "isGpuEnabled": True, "isInternetEnabled": True}
    },
    "cells": cells
}

with open("kaggle_benchmark.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"Done. Cells: {len(cells)}")
