"""
LangChain Agent Application

This module sets up a LangChain agent powered by Google's Generative AI model.
The agent uses environment variables for configuration to avoid hardcoding sensitive data.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Read configuration from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
google_model = os.getenv("GOOGLE_MODEL")

# Initialize the ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(
    model=google_model,
    api_key=google_api_key
)
