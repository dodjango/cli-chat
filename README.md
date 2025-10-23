# CLI Chat for Azure/OpenAI-compatible Endpoints

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![OpenAI SDK](https://img.shields.io/badge/OpenAI-SDK-412991?logo=openai&logoColor=white)
![uv](https://img.shields.io/badge/Env-uv-000000)
![just](https://img.shields.io/badge/Tasks-just-00ADD8?logo=gnubash&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-pytest-0A9EDC)

An interactive command-line chat client for Azure OpenAI or any OpenAI-compatible endpoint. It supports streaming responses, keeps conversation context in interactive mode, and reads configuration from `.env`.

![CLI chat demo](assets/chat-demo.gif)

_Interactive streaming demo: start chat, send a message, exit._

## Highlights ✨

- Azure/OpenAI-compatible via the official `openai` Python SDK
- Streaming by default (disable with `--no-stream`)
- Stateful interactive chat and one-shot mode
- Simple automation with `just`; environments managed by `uv`

## Requirements 📦

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [just](https://github.com/casey/just) command runner

The Justfile explicitly runs recipes under bash.

## Quick start 🚀

1. Create your `.env` from the template and fill values

```bash
just init-env
```

At minimum set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and one of `OPENAI_DEPLOYMENT` (Azure) or `OPENAI_MODEL` (non-Azure).

1. Install dependencies (via uv)

```bash
just install
```

1. Use the chat

One-shot question (prints only the answer):

```bash
just prompt "What is the weather in Paris?"
just prompt "Explain Kafka in 2 lines"
```

Interactive session (maintains history). Commands: `/exit`, `/quit`, `/clear`.

```bash
just chat
just chat-ns
```

Run directly without just:

```bash
uv run -- python chat.py --prompt "Hello"
uv run -- python chat.py
```

## Configuration ⚙️

Environment variables (loaded via `.env`):

- `OPENAI_API_KEY`       – API key (Azure or OpenAI)
- `OPENAI_BASE_URL`      – Endpoint base URL, e.g. `https://YOUR-AZURE.openai.azure.com/openai/v1/`
- `OPENAI_DEPLOYMENT`    – Azure deployment name (Azure)
- `OPENAI_MODEL`         – Non-Azure model id (alternative to `OPENAI_DEPLOYMENT`); model selection is via env, not CLI
- `OPENAI_ORG`           – Optional organization (OpenAI only)
- `OPENAI_SYSTEM_PROMPT` – Optional default system prompt (configure only via `.env`)
- `ASSISTANT_NAME`       – Optional display name used in interactive mode (defaults to `Assistant`)
- `USER_NAME`            – Optional display name for your prompts (defaults to `You`)

Color control (optional):

- `NO_COLOR`             – If set, disables colorized output
- `CHAT_COLOR`           – Set to `off|0|false|no` to disable colors; anything else enables
- Theme overrides (Rich styles): `USER_PREFIX_COLOR`, `ASSISTANT_PREFIX_COLOR`, `ASSISTANT_TEXT_COLOR`, `SYSTEM_PREFIX_COLOR`, `META_INFO_COLOR`

Tips:

- For Azure, set `OPENAI_DEPLOYMENT` and your service’s `OPENAI_BASE_URL`.
- For non-Azure/OpenAI-compatible services, set `OPENAI_MODEL` and the appropriate `OPENAI_BASE_URL`.

## Just recipes (cheat sheet) 🧰

- `just install`     – Create venv (uv) and install deps
- `just prompt "..."` – One-shot with positional message
- `just chat`        – Interactive (streaming)
- `just chat-ns`     – Interactive (non-streaming)
- `just env`         – Print important environment variables
- `just ping`        – Quick health check that sends a fixed prompt (`ping`) and exits
- `just lint`        – Basic lint (pyflakes)
- `just typecheck`   – Type checking with mypy
- `just check`       – Run all quality checks (lint + typecheck)
- `just test`        – Run smoke tests
- `just venv`        – Ensure uv-managed venv exists
- `just init-env`    – Copy `.env.example` to `.env` if missing

## Development 🛠️

### Code Quality

The project includes several tools for maintaining code quality:

```bash
just lint        # Run pyflakes
just typecheck   # Run mypy type checking
just check       # Run both lint and typecheck
just test        # Run pytest
```

### Pre-commit Hooks

Pre-commit hooks are configured to automatically check code quality before commits:

```bash
# Install pre-commit hooks (one-time setup)
uv run -- pre-commit install

# Run hooks manually on all files
uv run -- pre-commit run --all-files
```

The hooks include:
- Code formatting (black)
- Linting (flake8)
- Type checking (mypy)
- General file checks (trailing whitespace, line endings, etc.)

### CI/CD

GitHub Actions automatically runs tests, linting, and type checking on:
- All pushes to `main` and `claude/*` branches
- All pull requests to `main`

The CI pipeline tests against Python 3.10, 3.11, and 3.12.

The repo includes `.editorconfig` to keep indentation consistent (tabs for `Justfile`, 4 spaces for Python).

## Troubleshooting 🧪

- Missing environment variables: ensure `.env` exists and values are set.

```bash
just env
```

- `uv` not found: install `uv` and re-run `just install`.

## Azure vs OpenAI examples 🌐

Azure (use your resource’s base URL and deployment name):

```bash
# .env
OPENAI_API_KEY=... # Azure key
OPENAI_BASE_URL=https://<your-azure-openai>.openai.azure.com/openai/v1/
OPENAI_DEPLOYMENT=<your-deployment-name>
# optional
# OPENAI_SYSTEM_PROMPT=You are a helpful assistant.
# ASSISTANT_NAME=Computer
# USER_NAME=You
```

OpenAI (official):

```bash
# .env
OPENAI_API_KEY=... # OpenAI key
OPENAI_BASE_URL=https://api.openai.com/v1/
OPENAI_MODEL=gpt-4o-mini
# optional
# OPENAI_ORG=org_...
# OPENAI_SYSTEM_PROMPT=You are a helpful assistant.
```

## Security 🔐

- `.env` is gitignored. Keep your API keys secret and rotate them if they leak.

## License 📄

MIT © 2025 dodjango. See [LICENSE](./LICENSE).
