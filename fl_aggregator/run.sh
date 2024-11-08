#!/bin/sh

echo "Running FL aggregator app"

uv venv

uv pip install syftbox

uv run main.py
