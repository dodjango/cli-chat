# CLI Chat for Azure/OpenAI-compatible Endpoints

An interactive command-line chat client for Azure OpenAI or any OpenAI-compatible endpoint. It supports streaming responses, keeps conversation context in interactive mode, and reads configuration from `.env`.

## Highlights

- Azure/OpenAI-compatible via the official `openai` Python SDK
- Streaming by default (disable with `--no-stream`)
- Stateful interactive chat and one-shot mode
- Simple automation with `just`; environments managed by `uv`

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [just](https://github.com/casey/just) command runner

The Justfile explicitly runs recipes under bash.

## Quick start

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

## Configuration

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

## Just recipes (cheat sheet)

- `just install`     – Create venv (uv) and install deps
- `just prompt "..."` – One-shot with positional message
- `just chat`        – Interactive (streaming)
- `just chat-ns`     – Interactive (non-streaming)
- `just env`         – Print important environment variables
- `just ping`        – Quick health check that sends a fixed prompt (`ping`) and exits
- `just lint`        – Basic lint (pyflakes)
- `just test`        – Run smoke tests
- `just venv`        – Ensure uv-managed venv exists
- `just init-env`    – Copy `.env.example` to `.env` if missing

## Development

```bash
just lint
```

The repo includes `.editorconfig` to keep indentation consistent (tabs for `Justfile`, 4 spaces for Python).

## Troubleshooting

- Missing environment variables: ensure `.env` exists and values are set.

```bash
just env
```

- `uv` not found: install `uv` and re-run `just install`.

## Azure vs OpenAI examples

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

## Security

- `.env` is gitignored. Keep your API keys secret and rotate them if they leak.

## License

MIT © 2025 dodjango. See [LICENSE](./LICENSE).
