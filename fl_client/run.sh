#!/bin/sh

echo "Running FL client app"

uv venv

uv pip install syftbox

uv run main.py
