#!/usr/bin/env python3
"""Check current training status and progress."""
import os, json
os.chmod('/home/balaraj/.kaggle/kaggle.json', 0o600)
os.environ['KAGGLE_USERNAME'] = 'balarajr'
os.environ['KAGGLE_KEY'] = '249e64769da8831a4d34030104ccf3b7'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

slug = 'balarajr/notebook583d8fffed'

# 1. Status
print(f"Kernel: {slug}")
status = api.kernels_status(slug)
print(f"Status: {status}")

# 2. Try to get output/log
out_dir = '/tmp/kaggle_status_check'
os.makedirs(out_dir, exist_ok=True)
try:
    result = api.kernels_output(slug, path=out_dir)
    files = os.listdir(out_dir)
    if files:
        print(f"\nOutput files: {files}")
        for f in files:
            fp = os.path.join(out_dir, f)
            sz = os.path.getsize(fp)
            print(f"  {f}: {sz:,} bytes")
            if f.endswith('.log') and sz > 0:
                with open(fp) as fh:
                    content = fh.read()
                print(f"\n--- LOG (last 3000 chars) ---")
                print(content[-3000:])
    else:
        print("No output files yet (still running, no output saved)")
except Exception as e:
    print(f"Output error: {e}")

# 3. Calculate elapsed time
# Training started at 2026-04-26T06:39:55Z (cell 6 execute_input)
from datetime import datetime, timezone
start = datetime(2026, 4, 26, 6, 39, 55, tzinfo=timezone.utc)
now = datetime.now(timezone.utc)
elapsed = now - start
hours = elapsed.total_seconds() / 3600
print(f"\n⏱️  Training started: {start.isoformat()}")
print(f"⏱️  Current time:     {now.isoformat()}")
print(f"⏱️  Elapsed:          {hours:.1f} hours ({elapsed})")
print(f"\n⚠️  Note: Kaggle T4 sessions have a 12-hour limit.")
print(f"   Remaining: ~{max(0, 12-hours):.1f} hours")
