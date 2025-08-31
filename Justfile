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

# Run tests
test:
    uv run -- pytest -q

# Copy the example env if .env is missing
init-env:
    @if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example"; else echo ".env already exists"; fi

# Show resolved environment variables used by the client
env:
    @echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
    @echo "OPENAI_DEPLOYMENT=$OPENAI_DEPLOYMENT"
    @echo "OPENAI_MODEL=$OPENAI_MODEL"
    @echo "OPENAI_ORG=$OPENAI_ORG"
    @echo "OPENAI_API_KEY=<hidden>"
    @echo "ASSISTANT_NAME=$ASSISTANT_NAME"
    @echo "USER_NAME=$USER_NAME"
    @echo "CHAT_COLOR=$CHAT_COLOR"
    @echo "NO_COLOR=$NO_COLOR"

# Simple health check that sends a fixed prompt and exits
ping:
    uv run -- python chat.py --prompt "ping"

# Record a short terminal cast of interactive chat (you will type)
# Output: assets/chat-demo.cast
record-cast:
    @mkdir -p assets
    @rm -f assets/chat-demo.cast
    asciinema rec --overwrite --yes --quiet --cols 160 --rows 30 -c 'bash -c "just chat"' assets/chat-demo.cast

# Convert cast to GIF using asciinema/agg container (requires Docker/Podman)
# Output: assets/chat-demo.gif
cast-to-gif:
    @test -f assets/chat-demo.cast || (echo "Missing assets/chat-demo.cast. Run: just record-cast" && exit 1)
    @rm -f assets/chat-demo.gif
    AGG_FONT_FAMILY="${AGG_FONT_FAMILY:-DejaVu Sans Mono, Noto Color Emoji, Noto Emoji}" docker run --rm -v "$(pwd)":/data ghcr.io/asciinema/agg:latest assets/chat-demo.cast assets/chat-demo.gif

# Optimize GIF size (optional; requires gifsicle)
optimize-gif:
    @test -f assets/chat-demo.gif || (echo "Missing assets/chat-demo.gif. Run: just cast-to-gif" && exit 1)
    gifsicle -O3 assets/chat-demo.gif -o assets/chat-demo.gif

