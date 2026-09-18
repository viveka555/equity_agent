"""
llm.py

Centralized Groq LLM configuration.

Responsibilities
----------------
- Load environment variables
- Validate API key
- Create a reusable ChatGroq client

Used by
--------
planner.py
news.py
rag.py
report.py
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from config.settings import (
    GROQ_MODEL,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# Load environment variables from the project's .env file.
load_dotenv()


def create_llm() -> ChatGroq:
    """
    Create and validate the Groq LLM client.

    Returns
    -------
    ChatGroq
        Configured Groq chat model.

    Raises
    ------
    ValueError
        If GROQ_API_KEY is missing.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.error("GROQ_API_KEY was not found.")
        raise ValueError(
            "Missing GROQ_API_KEY. Create a .env file."
        )

    logger.info("Groq LLM initialized successfully.")

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=api_key,
    )


# Singleton LLM instance reused across the application.
llm = create_llm()