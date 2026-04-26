#!/usr/bin/env python3
"""List Kaggle kernels for balarajr."""
import os
os.environ['KAGGLE_USERNAME'] = 'balarajr'
os.environ['KAGGLE_KEY'] = 'd7b4e11c15ccf6f0339f23f5c028e6e3'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

print("Listing kernels for balarajr...")
kernels = api.kernels_list(user='balarajr', page_size=20)
for k in kernels:
    print(f"  slug: {k.ref}  |  title: {k.title}  |  status: {k.status}")

if not kernels:
    print("  (no kernels found or all are private)")
