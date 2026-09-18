"""
Simple connectivity test for Groq.
"""

from config.llm import llm

response = llm.invoke("Say hello in one sentence.")

print(response.content)