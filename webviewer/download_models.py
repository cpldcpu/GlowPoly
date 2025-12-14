import os
import json
import urllib.request
import sys
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://api.github.com/repos/tesseralis/polyhedra-viewer/contents/src/data/polyhedra?ref=canon"
TARGET_DIR = "models"

def download_file(url, filename):
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, os.path.join(TARGET_DIR, filename))
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

def main():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"Created directory: {TARGET_DIR}")

    print("Fetching file list...")
    try:
        with urllib.request.urlopen(BASE_URL) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to fetch file list: {e}")
        sys.exit(1)

    json_files = [item for item in data if item['name'].endswith('.json')]
    print(f"Found {len(json_files)} model files.")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for item in json_files:
            futures.append(executor.submit(download_file, item['download_url'], item['name']))
        
        results = [f.result() for f in futures]

    success_count = sum(results)
    print(f"Successfully downloaded {success_count} files.")

if __name__ == "__main__":
    main()
