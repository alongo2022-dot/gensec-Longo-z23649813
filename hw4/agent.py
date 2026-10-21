"""LangChain agent that connects to local MCP servers and answers questions with tool calls."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.mcp import MCPAdapter


HW4_DIR = Path(__file__).resolve().parent
PYTHON_COMMAND = sys.executable


def _require_env() -> tuple[str, str]:
    """Load required Google credentials from environment variables and exit if missing."""
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GOOGLE_MODEL")
    if not api_key or not model_name:
        print("Error: GOOGLE_API_KEY and GOOGLE_MODEL environment variables must be set.")
        raise SystemExit(1)
    return api_key, model_name


def _mcp_config() -> dict[str, dict[str, dict[str, str | list[str]]]]:
    """Create the stdio MCP server config for the glossary and time servers."""
    return {
        "mcpServers": {
            "security-glossary": {
                "command": PYTHON_COMMAND,
                "args": [str(HW4_DIR / "mcp-server.py")],
                "cwd": str(HW4_DIR),
            },
            "mcp-server-time": {
                "command": PYTHON_COMMAND,
                "args": ["-m", "mcp_server_time", "--local-timezone", "America/New_York"],
                "cwd": str(HW4_DIR),
            },
        }
    }


async def _print_tools_by_server(adapter: MCPAdapter) -> list:
    """Load tools from the adapter, print each tool name and description, and return the tool list."""
    tools = await adapter.list_tools()
    config = _mcp_config()["mcpServers"]
    print("\nLoaded MCP tools:")
    for server_name in config:
        server_tools = [tool for tool in tools if tool.name.startswith(f"{server_name}_")]
        print(f"\n=== {server_name} ===")
        if not server_tools:
            print("No tools available.")
            continue
        for tool in server_tools:
            description = tool.description or "No description provided."
            print(f"- {tool.name}: {description}")
    return tools


def _build_agent(api_key: str, model_name: str, tool_list: list) -> object:
    """Create the LangChain agent configured with the Google model and MCP tools."""
    model = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
    system_prompt = """You are a cybersecurity assistant with access to local MCP tools.

Use the available server-prefixed tools, not your general knowledge, when answering questions.

- For current date or timezone questions, call the prefixed time tool from the mcp-server-time server.
- For glossary definitions, call the security-glossary-prefixed lookup/list tools.
- For study plans, call the security-glossary-prefixed make_study_plan tool.
- Example prefixed tools include: mcp-server-time_get_current_time, security-glossary_list_terms, security-glossary_lookup_term, security-glossary_make_study_plan.
- Always call the relevant tool before answering.
- If a tool result is missing or unclear, say so plainly.
- Do not invent facts.
"""
    return create_agent(model=model, tools=tool_list, system_prompt=system_prompt)


async def _interactive_loop(agent: object) -> None:
    """Run the interactive chat loop and print tool calls and results as the agent works."""
    print("\nType a question and press Enter. Press Enter on an empty line to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                print("\nGoodbye!")
                break

            print(f"\nProcessing: {user_input}\n")
            async for step in agent.astream(
                {"messages": [{"role": "user", "content": user_input}]}
            ):
                if "model" in step:
                    for message in step["model"]["messages"]:
                        if getattr(message, "tool_calls", None):
                            for tool_call in message.tool_calls:
                                print(f"Tool call: {tool_call['name']}")
                                print(f"  Arguments: {tool_call['args']}")
                        if getattr(message, "content", None):
                            content = message.content
                            if isinstance(content, str):
                                print(f"Agent response: {content}")
                if "tools" in step:
                    for message in step["tools"]["messages"]:
                        result = getattr(message, "content", "")
                        print(f"Tool result: {result}")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as exc:
            print(f"\nError: {type(exc).__name__}: {exc}")


async def main() -> None:
    """Validate environment variables, load the MCP tools, create the agent, and run the interactive loop."""
    api_key, model_name = _require_env()
    async with MCPAdapter(_mcp_config()) as adapter:
        tools = await _print_tools_by_server(adapter)
        agent = _build_agent(api_key, model_name, tools)
        await _interactive_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())
