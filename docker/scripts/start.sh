#!/bin/bash
set -e

echo "Iniciando ambiente de desenvolvimento..."

poetry new project || true && \
cd project && \
poetry config virtualenvs.create false --local && \
poetry install --no-root || true

echo "Container Iniciado!"

tail -f /dev/null
