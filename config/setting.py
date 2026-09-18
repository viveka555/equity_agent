"""
settings.py

Application configuration.

All configurable values are stored here instead of hardcoding them
inside business logic.
"""
MODEL_NAME= "llama-3.3-70b-versatile"
MAX_NEWS_RESULT = 10

#planner task names
TASK_NEWS = "news"
TASK_RAG = "rag"
TASK_RATIO = "ratio"
TASK_DCF = "dcf"
TASK_REPORT = "report"
TASK_UNKNOWN = "unknown"