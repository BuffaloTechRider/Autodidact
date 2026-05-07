"""Autodidact — the self-learning AI agent.

An AI that learns like a human. It thinks first, asks when uncertain,
remembers what it learned, and gets more independent over time.

Quick start:
    from autodidact import Agent

    agent = Agent(
        local_model="ollama/qwen2.5:7b",
        cloud_model="openai/gpt-4o",
    )
    response = agent.query("What is the capital of France?")
    print(response.answer)
"""

__version__ = "1.0.0"

from autodidact.agent import Agent, QueryResponse, SavingsReport

__all__ = ["Agent", "QueryResponse", "SavingsReport"]
