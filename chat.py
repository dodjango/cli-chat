"""
Interactive CLI chat for Azure OpenAI / OpenAI-compatible endpoints.

Configuration via environment variables (optionally loaded from .env):
  - OPENAI_API_KEY      (required)
  - OPENAI_BASE_URL     (required)
  - OPENAI_DEPLOYMENT   (Azure) or OPENAI_MODEL (non-Azure)
  - OPENAI_ORG          (optional for OpenAI)
  - OPENAI_SYSTEM_PROMPT (optional default system prompt)

Usage:
  Interactive chat:
    uv run -- python chat.py

        One-shot:
            uv run -- python chat.py --prompt "Your question"

Commands during interactive mode:
  /exit, /quit  -> exit the chat
  /clear        -> clear conversation history (keeps system prompt)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Dict, Any, cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# Optional rich-based coloring for interactive output
try:
    from rich.console import Console  # type: ignore
    from rich.theme import Theme  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Console = None  # type: ignore[assignment]
    Theme = None  # type: ignore[assignment]


def getenv_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Missing required environment variable: {name}", file=sys.stderr)
        raise SystemExit(2)
    return value


def get_client() -> OpenAI:
    load_dotenv()
    base_url = getenv_required("OPENAI_BASE_URL")
    api_key = getenv_required("OPENAI_API_KEY")
    org = os.getenv("OPENAI_ORG")
    return OpenAI(base_url=base_url, api_key=api_key, organization=org or None)


def get_assistant_name() -> str:
    """Assistant display name used in interactive mode."""
    load_dotenv()
    return os.getenv("ASSISTANT_NAME", "Assistant")


def build_console() -> Optional[Any]:
    """Return a Rich Console with a simple theme, or None if rich is unavailable."""
    if Console is None or Theme is None:
        return None
    theme = Theme(
        {
            "meta.info": "bold dim",
            "user.prefix": "bold cyan",
            "assistant.prefix": "bold green",
            "assistant.text": "green",
            "system.prefix": "bold magenta",
        }
    )
    return Console(theme=theme)


def resolve_model() -> str:
    load_dotenv()
    model = os.getenv("OPENAI_DEPLOYMENT") or os.getenv("OPENAI_MODEL")
    if not model:
        print(
            "Missing OPENAI_DEPLOYMENT (Azure) or OPENAI_MODEL environment variable.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return model


def system_prompt_from_env() -> Optional[str]:
    return os.getenv("OPENAI_SYSTEM_PROMPT")


def chat_once(
    client: OpenAI,
    model: str,
    messages: List[ChatCompletionMessageParam],
    prompt: str,
    stream: bool,
    *,
    console: Optional[Any] = None,
    assistant_style: str = "assistant.text",
) -> str:
    messages.append(cast(ChatCompletionMessageParam, {"role": "user", "content": prompt}))

    if stream:
        try:
            resp_stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            collected: List[str] = []
            for event in resp_stream:
                try:
                    delta = event.choices[0].delta
                    text = getattr(delta, "content", None)
                    if text:
                        collected.append(text)
                        if console is not None:
                            console.print(text, end="", style=assistant_style)
                        else:
                            print(text, end="", flush=True)
                except Exception:
                    # Be resilient to any shape differences
                    pass
            if console is not None:
                console.print("")
            else:
                print()
            assistant_text = "".join(collected)
        except Exception:
            # Fallback to non-streaming on error
            stream = False
        else:
            messages.append(cast(ChatCompletionMessageParam, {"role": "assistant", "content": assistant_text}))
            return assistant_text

    # Non-streaming path
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = completion.choices[0].message.content or ""
    if console is not None:
        console.print(content, style=assistant_style)
    else:
        print(content)
    messages.append(cast(ChatCompletionMessageParam, {"role": "assistant", "content": content}))
    return content


def interactive_chat(model: str, system_prompt: Optional[str], stream: bool) -> int:
    client = get_client()
    messages: List[ChatCompletionMessageParam] = []
    if system_prompt:
        messages.append(cast(ChatCompletionMessageParam, {"role": "system", "content": system_prompt}))

    assistant_name = get_assistant_name()
    console = build_console()
    if console is not None:
        console.print("Type your message. Commands: /exit, /quit, /clear", style="meta.info")
    else:
        print("Type your message. Commands: /exit, /quit, /clear")
    try:
        while True:
            try:
                user = input("You: ").strip()
            except EOFError:
                print()
                break

            if not user:
                continue

            if user in {"/exit", "/quit"}:
                break
            if user == "/clear":
                messages = [cast(ChatCompletionMessageParam, {"role": "system", "content": system_prompt})] if system_prompt else []
                if console is not None:
                    console.print("History cleared.", style="meta.info")
                else:
                    print("History cleared.")
                continue

            if console is not None:
                console.print(f"{assistant_name}:", style="assistant.prefix", end=" ")
            else:
                print(f"{assistant_name}: ", end="", flush=True)
            chat_once(
                client,
                model,
                messages,
                user,
                stream=stream,
                console=console,
                assistant_style="assistant.text",
            )
    except KeyboardInterrupt:
        if console is not None:
            console.print("\nInterrupted.", style="meta.info")
        else:
            print("\nInterrupted.")
    return 0


def one_shot(model: str, system_prompt: Optional[str], prompt: str, stream: bool) -> int:
    client = get_client()
    messages: List[ChatCompletionMessageParam] = []
    if system_prompt:
        messages.append(cast(ChatCompletionMessageParam, {"role": "system", "content": system_prompt}))
    chat_once(client, model, messages, prompt, stream=stream)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="CLI chat for Azure/OpenAI-compatible endpoints")
    parser.add_argument("--prompt", help="One-shot prompt; if omitted, starts interactive chat")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")

    args = parser.parse_args(argv)
    model = resolve_model()
    system_prompt = system_prompt_from_env()
    stream = not args.no_stream

    if args.prompt:
        return one_shot(model, system_prompt, args.prompt, stream)
    return interactive_chat(model, system_prompt, stream)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
