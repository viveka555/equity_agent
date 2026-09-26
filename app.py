"""
app.py

Run the Institutional Equity Research Agent.
"""

from langchain_core.messages import HumanMessage

from config.logging_config import setup_logging
from graph import build_graph

# Configure logging once.
setup_logging()

graph = build_graph()

initial_state = {
    "messages": [HumanMessage(content="Latest news of BEL")],
    "company": "",
    "task": "",
    "news": "",
    "final_report": "",
}

result = graph.invoke(initial_state)

print("\n==============================")
print(result["final_report"])
print("==============================")