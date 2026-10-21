"""Security glossary MCP server for cybersecurity vocabulary and study planning."""

from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # pragma: no cover - compatibility with mcp v2.x
    from mcp.server.mcpserver import MCPServer as FastMCP

server = FastMCP("security-glossary")

GLOSSARY = {
    "phishing": (
        "Phishing is a cyber attack in which attackers impersonate trusted entities "
        "to trick people into revealing sensitive information or performing unsafe actions."
    ),
    "malware": (
        "Malware is malicious software designed to damage, disrupt, or gain unauthorized "
        "access to devices, networks, or data."
    ),
    "firewall": (
        "A firewall is a security system that monitors and controls incoming and outgoing "
        "network traffic based on security rules."
    ),
    "encryption": (
        "Encryption is the process of converting data into a coded form so unauthorized "
        "parties cannot read it without the correct key."
    ),
    "mfa": (
        "Multi-factor authentication (MFA) is a security method that requires more than one "
        "form of verification before granting access."
    ),
}


@server.tool()
def list_terms() -> list[str]:
    """List all available security glossary terms.

    Args: None.
    Returns: A list of all glossary term names as strings in the order they are defined.
    """
    return list(GLOSSARY.keys())


@server.tool()
def lookup_term(term: str) -> str:
    """Look up the definition for a specific security glossary term.

    Args:
        term (str): The term to look up, such as "phishing" or "firewall".
    Returns:
        A string containing the matching definition, or a clear message if the term is missing.
    """
    normalized_term = (term or "").strip().lower()
    definition = GLOSSARY.get(normalized_term)
    if definition is None:
        return f"Term '{term}' is not in the security glossary. Try one of: {', '.join(GLOSSARY)}."
    return definition


@server.tool()
def make_study_plan(topic: str, days: int) -> list[str]:
    """Create a day-by-day study checklist for a cybersecurity topic.

    Args:
        topic (str): The security topic to study, such as "phishing" or "network security".
        days (int): The number of study days to plan for.
    Returns:
        A list of strings representing a day-by-day checklist for the requested topic.
    """
    if topic is None or not str(topic).strip():
        topic = "the selected topic"

    day_count = max(1, int(days))
    plan: list[str] = []
    for day in range(1, day_count + 1):
        plan.append(
            f"Day {day}: Review {topic}; identify key concepts; practice examples; summarize risks and defenses; and test yourself."
        )
    return plan


if __name__ == "__main__":
    server.run(transport="stdio")
