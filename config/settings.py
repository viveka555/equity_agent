"""
settings.py

Central configuration for the Institutional Equity Research Agent.

Responsibilities
----------------
- Store application constants
- Store model configuration
- Avoid hardcoded values across the project

Used by
--------
config/llm.py
planner.py
future agents
"""

from __future__ import annotations
from pathlib import Path

# --------------------------------------------------------------------
# LLM Configuration
# --------------------------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_TEMPERATURE = 0

# --------------------------------------------------------------------
# Planner Tasks
# --------------------------------------------------------------------

TASK_NEWS = "news"
TASK_RAG = "rag"
TASK_RATIO = "ratio"
TASK_DCF = "dcf"
TASK_REPORT = "report"
TASK_UNKNOWN = "unknown"

# --------------------------------------------------------------------
# Search Configuration
# --------------------------------------------------------------------

MAX_NEWS_RESULTS = 10

#--------------------------------------------------------------------
# Persistent database location
#--------------------------------------------------------------------
CHROMA_PATH = Path("vector_db/chroma")