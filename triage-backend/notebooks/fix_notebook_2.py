import json

with open('/home/balaraj/META final/triage-backend/notebooks/kaggle_grpo_full.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if '"wtsheng/synthetic_reasoning_natural"' in line and '(' not in line:
                source[i] = line.replace('"wtsheng/synthetic_reasoning_natural"', '("wtsheng/synthetic_reasoning_natural", None, ["prompt", "question", "text"])')
            if '"open-thought/OpenThought-89K"' in line and '(' not in line:
                source[i] = line.replace('"open-thought/OpenThought-89K"', '("open-thought/OpenThought-89K", None, ["prompt", "question", "text", "instruction"])')
            if '"Replete-AI/rStar-Math"' in line and '(' not in line:
                source[i] = line.replace('"Replete-AI/rStar-Math"', '("Replete-AI/rStar-Math", None, ["question", "prompt"])')
            if '"ajibola16/reasoning-data"' in line and '(' not in line:
                source[i] = line.replace('"ajibola16/reasoning-data"', '("ajibola16/reasoning-data", None, ["question", "prompt"])')
            if '"Jiayi-Pan/Tiny-GSM8k"' in line and '(' not in line:
                source[i] = line.replace('"Jiayi-Pan/Tiny-GSM8k"', '("Jiayi-Pan/Tiny-GSM8k", None, ["question", "prompt"])')

        cell['source'] = source

with open('/home/balaraj/META final/triage-backend/notebooks/kaggle_grpo_full.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
print("Notebook patched successfully again!")
