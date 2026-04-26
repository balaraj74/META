#!/usr/bin/env python3
"""Download completed kernel output and source."""
import os
os.chmod('/home/balaraj/.kaggle/kaggle.json', 0o600)
os.environ['KAGGLE_USERNAME'] = 'balarajr'
os.environ['KAGGLE_KEY'] = '249e64769da8831a4d34030104ccf3b7'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

slug = 'balarajr/notebook3df73e1b02'

# Pull the notebook source
src_dir = '/home/balaraj/META final/triage-backend/results/kaggle_completed_kernel'
os.makedirs(src_dir, exist_ok=True)
print(f"Pulling notebook source for {slug}...")
try:
    api.kernels_pull(slug, path=src_dir, metadata=True)
    for f in os.listdir(src_dir):
        print(f"  {f} ({os.path.getsize(os.path.join(src_dir, f)):,} bytes)")
except Exception as e:
    print(f"Pull error: {e}")

# Download output
out_dir = '/home/balaraj/META final/triage-backend/results/kaggle_completed_output'
os.makedirs(out_dir, exist_ok=True)
print(f"\nDownloading output for {slug}...")
try:
    api.kernels_output(slug, path=out_dir)
    print("Output files:")
    for f in os.listdir(out_dir):
        sz = os.path.getsize(os.path.join(out_dir, f))
        print(f"  {f} ({sz:,} bytes)")
    if not os.listdir(out_dir):
        print("  (no output files)")
except Exception as e:
    print(f"Output error: {e}")

# Also download the running kernel's output (it should have partial logs)
run_out = '/home/balaraj/META final/triage-backend/results/kaggle_training_logs'
os.makedirs(run_out, exist_ok=True)
print(f"\nDownloading partial output for running kernel...")
try:
    api.kernels_output('balarajr/notebook583d8fffed', path=run_out)
    print("Running kernel output files:")
    for f in os.listdir(run_out):
        if f.startswith('notebook') or f == 'kernel-metadata.json':
            continue  # skip source files from earlier pull
        sz = os.path.getsize(os.path.join(run_out, f))
        print(f"  {f} ({sz:,} bytes)")
except Exception as e:
    print(f"Running kernel output error: {e}")
