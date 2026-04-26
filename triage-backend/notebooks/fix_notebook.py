import json

with open('/home/balaraj/META final/triage-backend/notebooks/kaggle_grpo_full.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'wtsheng/synthetic_reasoning_natural' in line and '(' not in line:
                source[i] = '    ("wtsheng/synthetic_reasoning_natural", None, ["prompt", "question", "text"]),\n'
            elif 'open-thought/OpenThought-89K' in line and '(' not in line:
                source[i] = '    ("open-thought/OpenThought-89K", None, ["prompt", "question", "text", "instruction"]),\n'
            elif 'Replete-AI/rStar-Math' in line and '(' not in line:
                source[i] = '    ("Replete-AI/rStar-Math", None, ["question", "prompt"]),\n'
            elif 'ajibola16/reasoning-data' in line and '(' not in line:
                source[i] = '    ("ajibola16/reasoning-data", None, ["question", "prompt"]),\n'

        cell['source'] = source

with open('/home/balaraj/META final/triage-backend/notebooks/kaggle_grpo_full.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
print("Notebook patched successfully!")
