import json

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🏥 Triage DPO Trainer (Google Colab Edition)\n",
    "This notebook will automatically download huge datasets from Kaggle, build 6,000+ training scenarios, and train the massive `Qwen2.5-1.5B` model using the free 15GB T4 GPU!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q kagglehub pandas transformers trl peft\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    read_file("scripts/colab_dpo_builder.py")
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    read_file("scripts/colab_train_dpo.py")
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}

with open("notebooks/TRIAGE_Training_Colab.ipynb", "w") as f:
    json.dump(notebook_content, f, indent=1)

print("Successfully generated notebooks/TRIAGE_Training_Colab.ipynb")
