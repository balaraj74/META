#!/usr/bin/env python3
"""Check status of all Kaggle kernels."""
import os
os.chmod('/home/balaraj/.kaggle/kaggle.json', 0o600)
os.environ['KAGGLE_USERNAME'] = 'balarajr'
os.environ['KAGGLE_KEY'] = '249e64769da8831a4d34030104ccf3b7'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

kernels = [
    'balarajr/notebook583d8fffed',
    'balarajr/notebook4b3fb31527',
    'balarajr/notebook3f440f7cde',
    'balarajr/notebook0024103197',
    'balarajr/notebook5c063f5aa8',
    'balarajr/notebookb9fed3ce63',
    'balarajr/notebook3df73e1b02',
    'balarajr/notebookccb19d1e62',
    'balarajr/notebook387908d9e0',
    'balarajr/notebook7b6d2c3f96',
]

for slug in kernels:
    try:
        status = api.kernels_status(slug)
        print(f"{slug}: {status}")
    except Exception as e:
        print(f"{slug}: ERROR - {e}")
