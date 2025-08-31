# Justfile for common project tasks using `uv` for venv and deps
# Install `just`: https://github.com/casey/just
# Install `uv`  : https://docs.astral.sh/uv/ (or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

set dotenv-load := true
set shell := ["bash", "-c"]

# Print help for all recipes
help:
    @just --list

# Ensure a local virtual environment exists (managed by uv)
venv:
    uv --version
    uv venv

# Install Python dependencies from requirements.txt using uv
install: venv
    uv pip install -U pip
    uv pip install -r requirements.txt

# One-shot ask with a positional message argument
# Example: just prompt "Hello there"
prompt message:
    uv run -- python chat.py --prompt "{{message}}"

# Interactive chat session with streaming output (maintains context until you exit)
chat:
    uv run -- python chat.py

# Interactive chat session (maintains context until you exit)
chat-ns:
    uv run -- python chat.py --no-stream

# Lint (basic) - pyflakes is included in requirements.txt
lint:
    uv run -- pyflakes chat.py || true

# Copy the example env if .env is missing
init-env:
    if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example"; else echo ".env already exists"; fi

# Show resolved environment variables used by the client
env:
    @echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
    @echo "OPENAI_DEPLOYMENT=$OPENAI_DEPLOYMENT"
    @echo "OPENAI_MODEL=$OPENAI_MODEL"
    @echo "OPENAI_ORG=$OPENAI_ORG"
    @echo "OPENAI_API_KEY=<hidden>"
    @echo "ASSISTANT_NAME=$ASSISTANT_NAME"
