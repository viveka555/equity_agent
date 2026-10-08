"""Optional live connectivity check for the Groq LLM."""

from __future__ import annotations

import os

import pytest


def test_groq_connectivity() -> None:
    """Check Groq only when the explicit integration-test flag is enabled.

    Live API checks are opt-in so normal pytest collection does not make
    network calls or incur API usage.
    """
    if os.getenv("RUN_LLM_INTEGRATION") != "1":
        pytest.skip(
            "Set RUN_LLM_INTEGRATION=1 to run the live Groq check."
        )

    from config.llm import llm

    response = llm.invoke("Say hello in one sentence.")

    assert response.content.strip()
