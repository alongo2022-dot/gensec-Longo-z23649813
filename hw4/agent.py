"""Start local MCP servers and print their tool metadata."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


HW4_DIR = Path(__file__).resolve().parent
PYTHON_COMMAND = sys.executable


def _server_parameters(command: str, args: list[str]) -> StdioServerParameters:
    """Build stdio parameters for an MCP server process."""
    return StdioServerParameters(command=command, args=args, cwd=str(HW4_DIR))


async def print_tools_for_server(label: str, server: StdioServerParameters) -> None:
    """Connect to a server, initialize it, and print each tool name and description."""
    try:
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                print(f"\n=== {label} ===")
                if not result.tools:
                    print("No tools available.")
                    return
                for tool in result.tools:
                    description = tool.description or "No description provided."
                    print(f"- {tool.name}: {description}")
    except Exception as exc:
        print(f"Error starting {label}: {exc}")
        raise


async def main() -> None:
    """Start the glossary server and the time server, then print their tool metadata."""
    my_server = _server_parameters(PYTHON_COMMAND, [str(HW4_DIR / "mcp-server.py")])
    time_server = _server_parameters(
        PYTHON_COMMAND,
        ["-m", "mcp_server_time", "--local-timezone", "America/New_York"],
    )

    for label, server in [
        ("security-glossary", my_server),
        ("mcp-server-time", time_server),
    ]:
        try:
            await print_tools_for_server(label, server)
        except Exception:
            return


if __name__ == "__main__":
    asyncio.run(main())
