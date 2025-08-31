"""
Interactive CLI chat for Azure OpenAI / any OpenAI-compatible endpoints.

This script provides:
    - An interactive REPL with slash-commands ("/exit", "/quit", "/clear").
    - A one-shot mode via --prompt.
    - Optional streaming output with a graceful fallback to non-streaming if
        the server or client does not support streaming.
    - Optional colorized output powered by "rich" (auto-disabled when unavailable
        or when NO_COLOR is set or CHAT_COLOR=off).

Configuration (reads environment variables and optionally .env via python-dotenv):
    - OPENAI_API_KEY        (required) API key for your provider
    - OPENAI_BASE_URL       (required) Base URL of the compatible API endpoint
                                                     e.g. Azure: https://<resource>.openai.azure.com
                                                                OpenAI: https://api.openai.com/v1
    - OPENAI_DEPLOYMENT     (preferred for Azure) model deployment name
    - OPENAI_MODEL          (fallback for non-Azure) model name
    - OPENAI_ORG            (optional, OpenAI only) organization id
    - OPENAI_SYSTEM_PROMPT  (optional) default system prompt for all chats
    - ASSISTANT_NAME        (optional) label shown for the assistant (default: "Assistant")
    - USER_NAME             (optional) label shown for you (default: "You")

Usage examples:
    - Interactive chat:
            uv run -- python chat.py

    - One-shot:
            uv run -- python chat.py --prompt "Your question"

Interactive commands:
    /exit, /quit  -> exit the chat
    /clear        -> clear conversation history (keeps system prompt if set)
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
    """
    Retrieve the value of a required environment variable.

    Args:
        name: Environment variable to read.

    Returns:
        The non-empty string value of the variable.

    Raises:
        SystemExit: If the variable is unset or empty (exit code 2).
    """
    value = os.getenv(name)
    if not value:
        print(f"Missing required environment variable: {name}", file=sys.stderr)
        raise SystemExit(2)
    return value


def get_client() -> OpenAI:
    """
        Create and return an OpenAI client configured from the environment.

        Notes:
            - load_dotenv() is idempotent; calling here ensures .env is considered
                even if callers forget to preload it.
            - OPENAI_BASE_URL and OPENAI_API_KEY are required.
            - OPENAI_ORG is optional and only used by OpenAI.
    """
    load_dotenv()
    base_url = getenv_required("OPENAI_BASE_URL")
    api_key = getenv_required("OPENAI_API_KEY")
    org = os.getenv("OPENAI_ORG")
    return OpenAI(base_url=base_url, api_key=api_key, organization=org or None)


def get_assistant_name() -> str:
    """
    Get the assistant's display name used in interactive mode.

    Env:
        ASSISTANT_NAME (default: "Assistant")
    """
    load_dotenv()
    return os.getenv("ASSISTANT_NAME", "Assistant")


def get_user_name() -> str:
    """
    Get the user's display name (input prompt prefix) used in interactive mode.

    Env:
        USER_NAME (default: "You")
    """
    load_dotenv()
    return os.getenv("USER_NAME", "You")


def build_console() -> Optional[Any]:
    """
        Return a Rich Console with a custom theme for colorized output, or None.

        Behavior and overrides:
            - If "rich" is not installed, returns None silently.
            - Respects NO_COLOR or CHAT_COLOR=off to disable colors.
            - Colors can be customized via:
                    USER_PREFIX_COLOR, ASSISTANT_PREFIX_COLOR, ASSISTANT_TEXT_COLOR,
                    SYSTEM_PREFIX_COLOR, META_INFO_COLOR.
    """
    if Console is None or Theme is None:
        return None

    # Color disable toggles
    if os.getenv("NO_COLOR") is not None:
        return None
    chat_color = (os.getenv("CHAT_COLOR") or "").strip().lower()
    if chat_color in {"off", "0", "false", "no"}:
        return None

    # Defaults
    theme_map: Dict[str, str] = {
        "meta.info": "bold dim",
        "user.prefix": os.getenv("USER_PREFIX_COLOR", "bold cyan"),
        "assistant.prefix": os.getenv("ASSISTANT_PREFIX_COLOR", "bold green"),
        "assistant.text": os.getenv("ASSISTANT_TEXT_COLOR", "green"),
        "system.prefix": os.getenv("SYSTEM_PREFIX_COLOR", "bold magenta"),
    }
    # Optional override for meta color
    if os.getenv("META_INFO_COLOR"):
        theme_map["meta.info"] = os.getenv("META_INFO_COLOR", theme_map["meta.info"])  # type: ignore[index]

    theme = Theme(theme_map)
    return Console(theme=theme)


def resolve_model() -> str:
    """
        Resolve and return the model/deployment identifier.

        Precedence:
            1) OPENAI_DEPLOYMENT (Azure)
            2) OPENAI_MODEL (generic / OpenAI)

        Raises:
                SystemExit: If neither environment variable is set (exit code 2).
    """
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
    """
    Return the system prompt from OPENAI_SYSTEM_PROMPT if set, else None.
    """
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
    """
        Send a single user prompt to the chat model and return the assistant reply.

        Behavior:
            - Appends the user message to ``messages`` and, after completion,
                appends the assistant message as well (mutates the list in-place).
            - If ``stream`` is True, attempts to stream tokens and prints them as
                they arrive; if streaming fails for any reason, automatically falls
                back to a non-streaming request.

        Args:
                client: Configured OpenAI-compatible client.
                model: Deployment/model identifier.
                messages: Mutable history of role/content dicts.
                prompt: The user message to send.
                stream: Whether to request a streaming response.
                console: Optional Rich console for styled printing; when None, uses print().
                assistant_style: Rich style name used for assistant tokens.

        Returns:
                The assistant's final response text (possibly empty string).
    """
    messages.append(cast(ChatCompletionMessageParam, {"role": "user", "content": prompt}))

    if stream:
        try:
            resp_stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            collected: List[str] = []
            # Iterate server-sent events, collecting partial deltas.
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
                    # Be resilient to any schema/shape differences from providers.
                    pass
            if console is not None:
                console.print("")
            else:
                print()
            assistant_text = "".join(collected)
        except Exception:
        # Any error while streaming -> fallback to non-streaming mode.
            stream = False
        else:
            messages.append(cast(ChatCompletionMessageParam, {"role": "assistant", "content": assistant_text}))
            return assistant_text

    # Non-streaming path (also used as a fallback when streaming fails)
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
    """
        Run an interactive chat session with the assistant in the terminal.

        Features:
            - Prompt loop with basic slash commands:
                    /exit, /quit -> leave the session
                    /clear       -> reset conversation history (preserves system prompt)
            - Optional token streaming for responses.

        Returns:
                0 on normal exit.
    """
    client = get_client()
    messages: List[ChatCompletionMessageParam] = []  # entire conversation state
    if system_prompt:
        # Seed the conversation with a system message if provided.
        messages.append(cast(ChatCompletionMessageParam, {"role": "system", "content": system_prompt}))

    assistant_name = get_assistant_name()
    user_name = get_user_name()
    console = build_console()
    if console is not None:
        console.print("Type your message. Commands: /exit, /quit, /clear", style="meta.info")
    else:
        print("Type your message. Commands: /exit, /quit, /clear")
    try:
        while True:
            try:
                user = input(f"{user_name}: ").strip()
            except EOFError:
                print()
                break

            if not user:
                continue

            if user in {"/exit", "/quit"}:
                break
            if user == "/clear":
                # Reset history; keep system message if one was set.
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
    """
    Run a single-turn chat completion and print the response.

    Args:
        model: Deployment/model identifier.
        system_prompt: Optional system message to seed the request.
        prompt: The user question.
        stream: Whether to request streaming output.

    Returns:
        0 on success.
    """
    client = get_client()
    messages: List[ChatCompletionMessageParam] = []
    if system_prompt:
        messages.append(cast(ChatCompletionMessageParam, {"role": "system", "content": system_prompt}))
    chat_once(client, model, messages, prompt, stream=stream)
    return 0


def main(argv: list[str]) -> int:
    """
        Parse CLI arguments and run either one-shot or interactive mode.

        Contract:
            - Exit code 0 on success; 2 for missing required env vars.
    """
    parser = argparse.ArgumentParser(description="CLI chat for Azure/OpenAI-compatible endpoints")
    parser.add_argument("--prompt", help="One-shot prompt; if omitted, starts interactive chat")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument(
        "--mode",
        choices=["completions", "agents"],
        default="completions",
        help="Backend mode: standard Chat Completions (default) or experimental OpenAI Agents (requires OpenAI platform)",
    )

    args = parser.parse_args(argv)
    model = resolve_model()
    system_prompt = system_prompt_from_env()
    stream = not args.no_stream

    if args.mode == "agents":
        # Lazy import so default mode has no new dependencies
        try:
            from agents_mode import run_agents_mode  # type: ignore
        except Exception:
            print(
                "Agents mode is not available. This experimental path targets the OpenAI platform and may require additional dependencies.",
                file=sys.stderr,
            )
            return 2
        return run_agents_mode(system_prompt=system_prompt, prompt=args.prompt, stream=stream)

    if args.prompt:
        return one_shot(model, system_prompt, args.prompt, stream)
    return interactive_chat(model, system_prompt, stream)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
