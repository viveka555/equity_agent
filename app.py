"""
app.py

Run the Institutional Equity Research Agent with laptop-local memory.
"""

from config.logging_config import setup_logging
from graph import build_graph
from memory.session import invoke_with_memory

# Configure logging once.
setup_logging()

def main() -> None:
    """Run an interactive session backed by local conversation history."""
    graph = build_graph()
    conversation_id = input(
        "Conversation ID [local]: "
    ).strip() or "local"
    print("Enter a question, or type 'exit' to end the session.")

    while True:
        user_message = input("\nYou: ").strip()
        if user_message.lower() in {"exit", "quit"}:
            break
        if not user_message:
            continue

        result = invoke_with_memory(
            graph=graph,
            user_message=user_message,
            conversation_id=conversation_id,
        )
        print(f"\nAgent:\n{result['final_report']}")


if __name__ == "__main__":
    setup_logging()
    main()
