"""
LangChain Agent Application

This module sets up a LangChain agent powered by Google's Generative AI model.
The agent uses environment variables for configuration to avoid hardcoding sensitive data.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_agent

# Read configuration from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
google_model = os.getenv("GOOGLE_MODEL")

# Initialize the ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(
    model=google_model,
    api_key=google_api_key
)

# Load and configure agent tools
# Built-in tool: requests_all provides HTTP request tools (GET/POST/PUT/PATCH/DELETE)
builtin_tools = load_tools(
    ["requests_all"],
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
system_prompt = """You are a helpful AI research assistant with access to powerful tools. Your role is to answer user questions accurately and thoroughly by leveraging available resources.

AVAILABLE TOOLS:
1. requests: Make HTTP requests to fetch web content. Use this to get real-time information, check current data, or retrieve web-based resources.
2. python: Execute Python code for calculations, data analysis, and complex computations. Use this for mathematical problems, data processing, or when you need to verify results.

INSTRUCTIONS:
- Always use the appropriate tool to answer the user's question instead of relying solely on your training data.
- Show your work by making explicit tool calls. Explain what tool you're using and why.
- For questions about live web content, use the requests tool to fetch the relevant page.
- When multiple tools might help, use them in combination to provide comprehensive answers.
- If you cannot answer the question even after using available tools, clearly state "I don't know" rather than guessing.
- Be transparent about tool results - share both successful findings and any limitations encountered.
- Always cite sources when referencing papers or web content found through tools."""

agent = create_agent(
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
            for step in agent.stream({"messages": [{"role": "user", "content": user_input}]}):
                # Detect and print model output and tool calls
                if "model" in step:
                    for msg in step["model"]["messages"]:
                        # Print any textual content (may be the final answer)
                        if getattr(msg, "content", None):
                            print(f"\n🤖 Agent: {msg.content}\n")
                        # Print any tool calls the model requested
                        if getattr(msg, "tool_calls", None):
                            for tool_call in msg.tool_calls:
                                print(f"🔧 Tool Call: {tool_call['name']}")
                                print(f"   Arguments: {tool_call['args']}")

                # Detect and print tool results
                if "tools" in step:
                    for msg in step["tools"]["messages"]:
                        print(f"📋 Tool Result: {str(msg.content)[:200]}...")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {type(e).__name__}: {e}")
            print("(Continuing...)\n")


# Run the agent if this script is executed directly
if __name__ == "__main__":
    run_agent_interactive()
