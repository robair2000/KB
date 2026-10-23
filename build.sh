#!/usr/bin/env bash
# Render build step: deps, Kronos source (pinned), and model weights baked into the build.
set -euo pipefail
KRONOS_COMMIT="${KRONOS_COMMIT:-67b630e67f6a18c9e9be918d9b4337c960db1e9a}"

pip install --upgrade pip
pip install -r requirements.txt

rm -rf kronos_repo
git clone --quiet https://github.com/shiyu-coder/Kronos kronos_repo
git -C kronos_repo checkout --quiet "$KRONOS_COMMIT"

python download_models.py
