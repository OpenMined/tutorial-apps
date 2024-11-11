#!/bin/sh

set -e

if [ ! -d .venv ]; then
  uv venv
fi

. .venv/bin/activate

uv pip install --upgrade -r requirements.txt

echo "Running 'model_local_training' with $(python3 --version) at '$(which python3)'"
python3 main.py

deactivate
