#!/bin/bash

if [ -f "./pyproject.toml" ]; then
  # Download and run the Poetry installation script
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="~/.local/bin:$PATH"
  # Install the pypi dependencies using poetry
  poetry install
else
  echo "No pypi dependencies defined with poetry."
fi
