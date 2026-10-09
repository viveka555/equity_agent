"""
app.py

Run the Institutional Equity Research Agent with laptop-local memory.
"""

import json
import logging
from pathlib import Path
import sys
from typing import Any

from config.logging_config import setup_logging
from graph import build_graph
from memory.session import RESEARCH_INPUT_KEYS, invoke_with_memory

logger = logging.getLogger(__name__)

# Configure logging once.
setup_logging()

def main() -> None:
    """Run an interactive session with local memory and optional evidence."""
    _configure_console_encoding()
    graph = build_graph()
    conversation_id = input(
        "Conversation ID [local]: "
    ).strip() or "local"
    research_inputs = _load_research_inputs()
    print("Enter a question, or type 'exit' to end the session.")

    while True:
        user_message = input("\nYou: ").strip()
        if user_message.lower() in {"exit", "quit"}:
            break
        if not user_message:
            continue

        try:
            result = invoke_with_memory(
                graph=graph,
                user_message=user_message,
                conversation_id=conversation_id,
                research_inputs=research_inputs,
            )
        except Exception as exc:
            logger.error(
                "Request failed for conversation %s: %s",
                conversation_id,
                exc,
            )
            print(f"\nAgent:\nI couldn't complete this request: {exc}")
            continue

        print(f"\nAgent:\n{result['final_report']}")


def _configure_console_encoding() -> None:
    """Use UTF-8 console output so report punctuation cannot crash the app."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _load_research_inputs() -> dict[str, Any] | None:
    """Read optional local evidence JSON for specialist/report workflows.

    Returns:
        Validated top-level research inputs, or ``None`` when omitted.

    Raises:
        ValueError: If the file is invalid JSON or contains unsupported keys.
        OSError: If the requested file cannot be read.
    """
    file_name = input("Research evidence JSON path [none]: ").strip()
    if not file_name:
        return None

    path = Path(file_name).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in research input file: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Research input JSON must contain an object.")

    unsupported_keys = set(payload) - RESEARCH_INPUT_KEYS
    if unsupported_keys:
        raise ValueError(
            "Unsupported research input keys: "
            + ", ".join(sorted(unsupported_keys))
        )

    logger.info("Loaded local research inputs from %s", path)
    return payload


if __name__ == "__main__":
    setup_logging()
    main()
