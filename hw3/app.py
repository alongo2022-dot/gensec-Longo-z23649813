"""
LangChain Agent Application

This module sets up a LangChain agent powered by Google's Generative AI model.
The agent uses environment variables for configuration to avoid hardcoding sensitive data.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import load_tools
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent

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

# Create the agent with a system prompt
system_prompt = """You are a helpful AI assistant with access to various tools.
You can search academic papers using arxiv, make HTTP requests, and execute Python code.
Please use these tools to help answer user questions accurately and thoroughly."""

agent = create_react_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)


def run_agent_interactive():
    """
    Run an interactive agent loop that processes user queries.
    
    This function:
    - Reads user prompts from standard input
    - Streams agent steps and tool calls
    - Displays tool names, arguments, and results
    - Exits on empty input
    - Handles errors gracefully with try/except
    """
    print("\n🤖 Agent Ready. Enter your query (or press Enter to exit):\n")
    
    while True:
        try:
            # Read user prompt
            user_input = input("You: ").strip()
            
            # Exit on empty input
            if not user_input:
                print("\n👋 Goodbye!")
                break
            
            print(f"\n🔄 Processing: {user_input}\n")
            
            # Stream agent steps and tool calls
            for step in agent.stream({"messages": [("user", user_input)]}):
                # Detect and print tool calls from the agent
                if "agent" in step:
                    agent_step = step["agent"]
                    if "messages" in agent_step:
                        for msg in agent_step["messages"]:
                            # Check for tool calls in message content
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    print(f"🔧 Tool Call: {tool_call['name']}")
                                    print(f"   Arguments: {tool_call['args']}")
                
                # Detect and print tool results
                if "tools" in step:
                    tool_step = step["tools"]
                    if "messages" in tool_step:
                        for msg in tool_step["messages"]:
                            print(f"📋 Tool Result: {msg.content[:200]}...")
                
                # Print final agent response
                if "agent" in step:
                    agent_step = step["agent"]
                    if "messages" in agent_step:
                        for msg in agent_step["messages"]:
                            if hasattr(msg, "content") and not hasattr(msg, "tool_calls"):
                                print(f"\n🤖 Agent: {msg.content}\n")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {type(e).__name__}: {e}")
            print("(Continuing...)\n")


# Run the agent if this script is executed directly
if __name__ == "__main__":
    run_agent_interactive()
