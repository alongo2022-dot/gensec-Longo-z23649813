"""
LangChain Agent Application

This module sets up a LangChain agent powered by Google's Generative AI model.
The agent uses environment variables for configuration to avoid hardcoding sensitive data.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import load_tools
from langchain_experimental.tools import PythonREPLTool

# Read configuration from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
google_model = os.getenv("GOOGLE_MODEL")

# Initialize the ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(
    model=google_model,
    api_key=google_api_key
)

# Load and configure agent tools
# Built-in tools: arxiv for academic paper search, requests_all for HTTP requests
builtin_tools = load_tools(
    ["arxiv", "requests_all"],
    llm=model,
    allow_dangerous_tools=True
)

# Python REPL tool for executing Python code within the agent
python_repl_tool = PythonREPLTool()

# Combine all tools into a single list
tools = builtin_tools + [python_repl_tool]

# Display available tools at startup
print("=" * 60)
print("Agent Tools Loaded:")
print("=" * 60)
for tool in tools:
    print(f"\n📌 {tool.name}")
    print(f"   Description: {tool.description}")
print("\n" + "=" * 60)
