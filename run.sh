#!/bin/sh

# this will create venv from python version defined in .python-version
uv venv

uv pip install syftbox

# run app using python from venv
uv run main.py
