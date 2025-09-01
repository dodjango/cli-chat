"""
Prototype: OpenAI Agents SDK + MCP (via LiteLLM for Azure models)

This module wires an experimental "agents" mode, using the OpenAI Agents SDK and
its MCP integration while driving models through LiteLLM so Azure-hosted models
can be used without an OpenAI account.

Environment variables (expected):
  - AGENTS_MODEL                Optional. If set, used as the Litellm model id.
                Example for Azure: "azure/<deployment-name>".
                If not set and OPENAI_DEPLOYMENT exists, we'll
                build "azure/<OPENAI_DEPLOYMENT>" automatically.
  - OPENAI_API_KEY              Used as AZURE_API_KEY when base URL is Azure.
  - OPENAI_BASE_URL             If Azure URL, mapped to AZURE_API_BASE.
                If ends with "/openai/v1" or "/openai/v1/",
                the suffix is removed for AZURE_API_BASE.
  - AGENTS_AZURE_API_VERSION    Required for Azure unless environment already
                defines AZURE_API_VERSION (or litellm config).

MCP configuration (optional):
  - AGENTS_MCP_CONFIG           Path to a JSON file describing servers.
                Default: ./mcp_servers.json if present.
  Format:
    {
      "servers": [
      {"name":"web","transport":"stdio","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","."]}
      ]
    }

Notes:
  - This is a prototype. Error handling and transport options are minimal.
  - Streaming is supported for final assistant text only (no partials per token
  are printed during tool calls).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict

try:
  from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional but recommended
  load_dotenv = None  # type: ignore


def _strip_azure_suffix(base_url: str) -> str:
  # Azure base often includes /openai/v1; litellm expects the resource base
  # e.g., https://<resource>.openai.azure.com
  for suffix in ("/openai/v1", "/openai/v1/"):
    if base_url.endswith(suffix):
      return base_url[: -len(suffix)]
  return base_url.rstrip("/")


def _configure_env_for_azure() -> None:
  base_url = os.getenv("OPENAI_BASE_URL", "")
  if "openai.azure.com" not in base_url:
    return  # Not Azure

  # Map credentials for litellm's Azure provider
  if os.getenv("AZURE_API_KEY") is None and os.getenv("OPENAI_API_KEY"):
    os.environ["AZURE_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

  if os.getenv("AZURE_API_BASE") is None:
    os.environ["AZURE_API_BASE"] = _strip_azure_suffix(base_url)

  # Require an API version one way or another
  if os.getenv("AZURE_API_VERSION") is None:
    version = os.getenv("AGENTS_AZURE_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")
    if not version:
      print(
        "Missing Azure API version. Set AGENTS_AZURE_API_VERSION or AZURE_API_VERSION.",
        file=sys.stderr,
      )
      raise SystemExit(2)
    os.environ["AZURE_API_VERSION"] = version


def _resolve_litellm_model() -> str:
  # Prefer explicit agents model
  model = os.getenv("AGENTS_MODEL")
  if model:
    return model

  # Try infer Azure
  deployment = os.getenv("OPENAI_DEPLOYMENT")
  base_url = os.getenv("OPENAI_BASE_URL", "")
  if deployment and "openai.azure.com" in base_url:
    return f"azure/{deployment}"

  print(
    "AGENTS_MODEL is required unless using Azure with OPENAI_DEPLOYMENT set.",
    file=sys.stderr,
  )
  raise SystemExit(2)


def _install_loop_exception_handler() -> None:
  """Install an asyncio loop exception handler to suppress noisy shutdown errors.

  Specifically filters AnyIO/Trio-related RuntimeError:
    "Attempted to exit cancel scope in a different task than it was entered in"
  which can surface during asyncio.run() shutdown when MCP stdio clients close.
  """
  try:
    loop = asyncio.get_running_loop()
  except RuntimeError:
    return
  default = loop.get_exception_handler()

  def handler(loop, context):  # type: ignore[override]
    exc = context.get("exception")
    msg = context.get("message")
    text = str(exc or msg or "")
    if "Attempted to exit cancel scope in a different task than it was entered in" in text:
      return
    if default:
      default(loop, context)
    else:
      loop.default_exception_handler(context)

  try:
    loop.set_exception_handler(handler)
  except Exception:
    pass


@dataclass
class MCPServerSpec:
  name: str
  transport: str
  command: Optional[str] = None
  args: Optional[List[str]] = None
  env: Optional[Dict[str, str]] = None


def _load_mcp_servers_from_config() -> List[MCPServerSpec]:
  path = os.getenv("AGENTS_MCP_CONFIG", "mcp_servers.json")
  if not os.path.exists(path):
    return []
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
  servers = []
  for item in data.get("servers", []):
    servers.append(
      MCPServerSpec(
        name=item.get("name") or item.get("id") or "server",
        transport=item.get("transport", "stdio"),
        command=item.get("command"),
        args=item.get("args"),
  env=item.get("env"),
      )
    )
  return servers


async def _build_agent(system_prompt: Optional[str]):
  # Dynamic imports so default mode doesn't require these deps at import time
  import importlib
  agents_mod = importlib.import_module("agents")
  LitellmModel = importlib.import_module("agents.extensions.models.litellm_model").LitellmModel  # type: ignore[attr-defined]
  MCPServerStdio = importlib.import_module("agents.mcp.server").MCPServerStdio  # type: ignore[attr-defined]

  _configure_env_for_azure()
  model_id = _resolve_litellm_model()

  # LitellmModel will read AZURE_* envs for Azure
  model = LitellmModel(model=model_id, api_key=os.getenv("AZURE_API_KEY") or os.getenv("OPENAI_API_KEY", ""))

  # Optional MCP servers
  mcp_servers_specs = _load_mcp_servers_from_config()
  mcp_servers = []
  # Resolve a working directory for subprocesses (e.g., Git MCP)
  workdir = os.getenv("AGENTS_WORKDIR") or os.getcwd()
  for spec in mcp_servers_specs:
    if spec.transport != "stdio":
      # Prototype supports stdio only
      continue
    if not spec.command:
      continue
    # Allow slower servers (e.g., first-run npx installs) to start up.
    # Default to 60s, configurable via AGENTS_MCP_CONNECT_TIMEOUT (seconds).
    try:
      timeout = float(os.getenv("AGENTS_MCP_CONNECT_TIMEOUT", "60"))
    except Exception:
      timeout = 60.0
    # Build environment: inherit current env and overlay per-server env (with ${VAR} expansion)
    env_overlay: Dict[str, str] = {}
    if spec.env:
      for k, v in spec.env.items():
        raw = str(v)
        try:
          expanded = os.path.expandvars(raw)
        except Exception:
          expanded = raw
        # Treat unresolved ${VAR} as missing
        if "${" in raw and "}" in raw and expanded == raw:
          expanded = ""
        env_overlay[k] = expanded
    proc_env = dict(os.environ)
    proc_env.update(env_overlay)

    # If the server declares env and any value expands to empty, skip starting it.
    # This prevents runtime tool errors (e.g., Bing without BING_SEARCH_API_KEY).
    if spec.env:
      missing_keys = [k for k, v in env_overlay.items() if (v or "").strip() == ""]
      if missing_keys:
        print(
          f"[mcp] Skipping server '{spec.name}' due to missing required env: {', '.join(missing_keys)}",
          file=sys.stderr,
        )
        continue

    server = MCPServerStdio(
      params={
        "command": spec.command,
        "args": spec.args or [],
        # Some Agents SDK builds use 'timeout'; others use 'connect_timeout'. Provide both.
        "timeout": timeout,
        "connect_timeout": timeout,
        # Provide environment and working directory to the subprocess.
        # Many MCP servers (e.g., Git) respect cwd for relative operations.
        "env": proc_env,
        "cwd": workdir,
      }
    )
    try:
      await server.connect()
    except Exception as e:
      print(f"[mcp] Failed to start server '{spec.name}': {e}", file=sys.stderr)
      # Best-effort cleanup if the process started partially
      try:
        maybe_disc = getattr(server, "disconnect", None)
        if callable(maybe_disc):
          res = maybe_disc()
          if asyncio.iscoroutine(res):
            await res
      except Exception:
        pass
      continue
    mcp_servers.append(server)

  # Augment instructions with MCP usage guidance to improve tool reliability
  # and ensure Git working directory is set correctly.
  default_instructions = "You are a helpful assistant."
  bing_key_present = bool(os.getenv("BING_SEARCH_API_KEY"))
  if bing_key_present:
    search_pref = "Prefer Bing Search MCP (API-based)."
  else:
    search_pref = "Use available search MCPs; if scraping is rate-limited, wait and retry."
  mcp_guidance = (
    f"\n\nMCP tools may be available (filesystem, git, search, weather, time). "
    f"For any Git-related tasks, first ensure the working directory is set by calling the 'git_set_working_dir' tool "
    f"with path: {workdir}. Prefer MCP tools for Git operations. {search_pref}"
  )
  instructions = (system_prompt or default_instructions) + mcp_guidance
  agent = agents_mod.Agent(
    name="Assistant",
    instructions=instructions,
    model=model,
    mcp_servers=mcp_servers,
  )
  return agent


async def _run_one_shot(system_prompt: Optional[str], prompt: str, stream: bool) -> int:
  import importlib
  Runner = importlib.import_module("agents").Runner  # type: ignore[attr-defined]
  from openai.types.responses import ResponseTextDeltaEvent  # type: ignore

  _install_loop_exception_handler()
  agent = await _build_agent(system_prompt)
  try:
    # Optional Git bootstrap: ask the agent to set working dir via MCP before the main run
    if os.getenv("AGENTS_GIT_BOOTSTRAP", "1") not in {"0", "false", "False"}:
      try:
        cwd = os.getenv("AGENTS_WORKDIR") or os.getcwd()
        bootstrap_msg = (
          "Before anything else, use the MCP tool 'git_set_working_dir' to set the Git working directory to "
          f"'{cwd}'. Respond only with 'OK' when done."
        )
        # Run a quick, non-streamed bootstrap step with a timeout
        await asyncio.wait_for(Runner.run(agent, bootstrap_msg), timeout=12.0)
      except Exception:
        # Non-fatal; continue even if bootstrap fails
        pass
    if stream:
      # Stream raw token deltas from the model
      result_stream = Runner.run_streamed(agent, input=prompt)
      try:
        async for event in result_stream.stream_events():
          if getattr(event, "type", None) == "raw_response_event" and isinstance(getattr(event, "data", None), ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
        print()
      finally:
        # Ensure the stream is closed
        for meth_name in ("aclose", "close", "cancel", "stop"):
          meth = getattr(result_stream, meth_name, None)
          if callable(meth):
            try:
              res = meth()
              if asyncio.iscoroutine(res):
                await res
            except Exception:
              pass
    else:
      result = await Runner.run(agent, prompt)
      print(result.final_output)
    return 0
  finally:
    # Best-effort graceful shutdown to avoid lingering tasks/cancel-scope errors
    await _cleanup_agent(agent)
    # Give background tasks a brief moment to settle
    try:
      await asyncio.sleep(0.05)
    except Exception:
      pass


async def _run_interactive(system_prompt: Optional[str], stream: bool) -> int:
  import importlib
  Runner = importlib.import_module("agents").Runner  # type: ignore[attr-defined]
  from openai.types.responses import ResponseTextDeltaEvent  # type: ignore

  _install_loop_exception_handler()
  agent = await _build_agent(system_prompt)
  # Attempt a Git bootstrap once at session start to set working directory
  if os.getenv("AGENTS_GIT_BOOTSTRAP", "1") not in {"0", "false", "False"}:
    try:
      cwd = os.getenv("AGENTS_WORKDIR") or os.getcwd()
      bootstrap_msg = (
        "Before anything else, use the MCP tool 'git_set_working_dir' to set the Git working directory to "
        f"'{cwd}'. Respond only with 'OK' when done."
      )
      await asyncio.wait_for(Runner.run(agent, bootstrap_msg), timeout=12.0)
    except Exception:
      pass
  user_name = os.getenv("USER_NAME", "You")
  assistant_name = os.getenv("ASSISTANT_NAME", "Assistant")
  print("Type your message. Commands: /exit, /quit, /clear")
  messages: List[str] = []  # For simple local context; Agent holds its own state per run
  try:
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
          messages.clear()
          print("History cleared.")
          continue

        # Join local history + latest input (Agent maintains internal state too)
        messages.append(user)
        query = "\n\n".join(messages)
        print(f"{assistant_name}: ", end="", flush=True)
        if stream:
          result_stream = Runner.run_streamed(agent, input=query)
          try:
            async for event in result_stream.stream_events():
              if getattr(event, "type", None) == "raw_response_event" and isinstance(getattr(event, "data", None), ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
            print()
          finally:
            for meth_name in ("aclose", "close", "cancel", "stop"):
              meth = getattr(result_stream, meth_name, None)
              if callable(meth):
                try:
                  res = meth()
                  if asyncio.iscoroutine(res):
                    await res
                except Exception:
                  pass
        else:
          result = await Runner.run(agent, query)
          print(result.final_output)
    except KeyboardInterrupt:
      print("\nInterrupted.")
    return 0
  finally:
    await _cleanup_agent(agent)
    try:
      await asyncio.sleep(0.05)
    except Exception:
      pass


async def _cleanup_agent(agent) -> None:
  """Best-effort cleanup of agent and attached MCP servers to avoid shutdown errors."""
  # Close agent if it exposes a suitable API
  for meth_name in ("aclose", "close", "shutdown", "disconnect", "stop"):
    meth = getattr(agent, meth_name, None)
    if callable(meth):
      try:
        res = meth()
        if asyncio.iscoroutine(res):
          await res
      except Exception:
        pass
  # Close any attached servers if accessible via attribute
  for attr in ("mcp_servers", "servers", "tools"):
    servers = getattr(agent, attr, None)
    if not servers:
      continue
    for srv in servers:
      for meth_name in ("disconnect", "aclose", "close", "shutdown", "stop"):
        meth = getattr(srv, meth_name, None)
        if callable(meth):
          try:
            res = meth()
            if asyncio.iscoroutine(res):
              await res
          except Exception:
            pass


def run_agents_mode(*, system_prompt: Optional[str], prompt: Optional[str], stream: bool) -> int:
  # Ensure .env is loaded so Azure/OpenAI settings are available.
  if load_dotenv is not None:
    try:
      load_dotenv()
    except Exception:
      pass
  try:
    import importlib
    importlib.import_module("agents")
  except Exception as e:
    print(
      "Agents mode requires the OpenAI Agents SDK. Install with: pip install 'openai-agents[litellm]'.\n"
      f"Import error: {e}",
      file=sys.stderr,
    )
    return 2

  # Disable tracing by default to avoid OpenAI tracing API calls in Azure-only setups
  try:
    import importlib
    set_tracing_disabled = getattr(importlib.import_module("agents"), "set_tracing_disabled", None)  # type: ignore[attr-defined]
    if callable(set_tracing_disabled):
      set_tracing_disabled(True)  # type: ignore[misc]
  except Exception:
    pass

  if prompt:
    return asyncio.run(_run_one_shot(system_prompt, prompt, stream))
  return asyncio.run(_run_interactive(system_prompt, stream))
