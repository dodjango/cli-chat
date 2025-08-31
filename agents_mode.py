"""
Experimental Agents mode scaffold.

This module is a placeholder for a migration toward OpenAI Agents / Responses APIs
with MCP support. It is provided for users who want to experiment with the
OpenAI platform's native MCP integrations.

Important constraints for this repository:
- The current project is designed to be provider-agnostic and supports Azure
  OpenAI via environment variables and OPENAI_BASE_URL. The OpenAI Agents SDK
  and platform-native MCP integrations are not designed to target arbitrary base
  URLs or Azure endpoints.
- Because the user indicated they do not have an OpenAI account and their model
  is hosted on Azure (e.g., GPT-5 on Azure), this module only prints a helpful
  message and exits. A full migration would couple this CLI to OpenAI's
  platform-only features and break Azure compatibility.

Strategy going forward:
- Keep the default "completions" mode unchanged to continue supporting Azure and
  other OpenAI-compatible endpoints.
- If at some point you decide to move to the OpenAI platform, this module can be
  replaced with a concrete implementation that uses the OpenAI Agents SDK and
  configures MCP servers as documented at:
    https://platform.openai.com/docs/guides/tools-connectors-mcp
    https://openai.github.io/openai-agents-python/mcp/

"""
from __future__ import annotations

from typing import Optional
import sys


def run_agents_mode(*, system_prompt: Optional[str], prompt: Optional[str], stream: bool) -> int:
    """
    Entry point for the experimental Agents mode.

    Since this repository targets Azure/OpenAI-compatible endpoints (via
    OPENAI_BASE_URL) and the user does not have an OpenAI account, we do not
    attempt to run the OpenAI Agents SDK here. Instead, provide a clear
    explanation and exit with code 2.
    """
    _ = stream  # unused in this stub

    msg = (
        "Agents mode (OpenAI platform native MCP) is not available in this setup.\n"
        "Reason: the OpenAI Agents SDK targets the OpenAI platform, while this\n"
        "project is configured to use Azure/OpenAI-compatible endpoints via\n"
        "OPENAI_BASE_URL.\n\n"
        "What you can do now:\n"
        "  - Continue using the default 'completions' mode.\n"
        "  - Or, if you want MCP today with Azure, implement the local MCP bridge\n"
        "    as discussed: connect to MCP servers, map their tools to function-\n"
        "    calling, execute tool calls, and feed results back to the model.\n\n"
        "If you later switch to OpenAI's platform, replace this module with an\n"
        "Agents SDK-based implementation. See: https://platform.openai.com/docs/guides/tools-connectors-mcp\n"
    )
    print(msg, file=sys.stderr)
    return 2
