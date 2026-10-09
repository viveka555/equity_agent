"""Tests for interactive app request handling."""

from unittest.mock import patch

import app


def test_app_recovers_from_workflow_error(capsys) -> None:
    """A failed workflow should not terminate the interactive session."""
    responses = iter(
        [
            ValueError("No DCF financial inputs available."),
            {"final_report": "The next request completed."},
        ]
    )
    with (
        patch("app.build_graph", return_value=object()),
        patch("app._load_research_inputs", return_value=None),
        patch("app.invoke_with_memory", side_effect=responses),
        patch(
            "builtins.input",
            side_effect=["test-session", "First request", "Second request", "exit"],
        ),
    ):
        app.main()

    output = capsys.readouterr().out
    assert "No DCF financial inputs available." in output
    assert "The next request completed." in output


def test_console_output_uses_utf8() -> None:
    """Configure UTF-8 encoding for Unicode report characters."""
    with (
        patch("app.sys.stdout") as stdout,
        patch("app.sys.stderr") as stderr,
    ):
        app._configure_console_encoding()

    stdout.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")
    stderr.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")
