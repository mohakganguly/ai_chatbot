"""
Tool/Input guardrails.

Validates a selected tool and its arguments before
the tool is allowed to execute.

Designed to prevent:
- dangerous shell commands
- destructive file operations
- path traversal
- access to sensitive files
- credential exposure
"""

from __future__ import annotations

import re
from typing import Any

from guardrails.schemas import GuardrailResult


class ToolGuard:
    """
    Validate tool calls before execution.
    """

    # ==========================================================
    # Dangerous command patterns
    # ==========================================================

    DANGEROUS_COMMAND_PATTERNS = [
        r"\brm\s+-rf\s+/",
        r"\brm\s+-rf\s+\*",
        r"\bdel\s+/[fqs]+\s+.*",
        r"\bformat\s+[a-z]:",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bpoweroff\b",
        r"\binit\s+[06]\b",
        r"\bmkfs\b",
        r"\bdd\s+if=",
        r"\bchmod\s+-R\s+777\b",
        r"\bcurl\b.*\|\s*(sh|bash)",
        r"\bwget\b.*\|\s*(sh|bash)",
        r"\b:()\s*\{\s*:\|\:&\s*\};:",
    ]

    # ==========================================================
    # Sensitive file patterns
    # ==========================================================

    SENSITIVE_PATH_PATTERNS = [
        r"\.env\b",
        r"\bid_rsa\b",
        r"\bid_ed25519\b",
        r"\.ssh[/\\]",
        r"\.aws[/\\]",
        r"\.kube[/\\]",
        r"\bcredentials\.json\b",
        r"\bsecrets?\.json\b",
        r"/etc/passwd",
        r"/etc/shadow",
        r"windows[/\\]system32",
    ]

    # ==========================================================
    # Path traversal patterns
    # ==========================================================

    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
    ]

    # ==========================================================
    # Sensitive argument names
    # ==========================================================

    SENSITIVE_ARGUMENT_NAMES = {
        "password",
        "passwd",
        "secret",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "private_key",
        "credential",
        "credentials",
    }

    def __init__(self):

        self._dangerous_command_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.DANGEROUS_COMMAND_PATTERNS
        ]

        self._sensitive_path_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SENSITIVE_PATH_PATTERNS
        ]

        self._path_traversal_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.PATH_TRAVERSAL_PATTERNS
        ]

    # ==========================================================
    # Main Validation
    # ==========================================================

    def validate(
        self,
        tool_name: str,
        arguments: Any,
    ) -> GuardrailResult:
        """
        Validate a tool call before execution.
        """

        # ------------------------------------------------------
        # No tool
        # ------------------------------------------------------

        if not tool_name:
            return GuardrailResult(
                allowed=False,
                reason="No tool was selected.",
                category="invalid_tool",
            )

        # ------------------------------------------------------
        # Normalize arguments
        # ------------------------------------------------------

        arguments_dict = self._normalize_arguments(
            arguments
        )

        # ------------------------------------------------------
        # Check sensitive argument names
        # ------------------------------------------------------

        for key in arguments_dict:

            normalized_key = str(key).lower().strip()

            if normalized_key in self.SENSITIVE_ARGUMENT_NAMES:

                return GuardrailResult(
                    allowed=False,
                    reason=(
                        "Sensitive credentials must not be "
                        "passed through tool arguments."
                    ),
                    category="sensitive_data",
                )

        # ------------------------------------------------------
        # Convert arguments to searchable text
        # ------------------------------------------------------

        arguments_text = self._arguments_to_text(
            arguments_dict
        )

        # ------------------------------------------------------
        # Dangerous command detection
        # ------------------------------------------------------

        for pattern in self._dangerous_command_patterns:

            if pattern.search(arguments_text):

                return GuardrailResult(
                    allowed=False,
                    reason=(
                        "Potentially dangerous system command "
                        "detected."
                    ),
                    category="dangerous_command",
                )

        # ------------------------------------------------------
        # Sensitive file detection
        # ------------------------------------------------------

        for pattern in self._sensitive_path_patterns:

            if pattern.search(arguments_text):

                return GuardrailResult(
                    allowed=False,
                    reason=(
                        "Attempt to access a sensitive file "
                        "or credential location detected."
                    ),
                    category="sensitive_path",
                )

        # ------------------------------------------------------
        # Path traversal detection
        # ------------------------------------------------------

        for pattern in self._path_traversal_patterns:

            if pattern.search(arguments_text):

                return GuardrailResult(
                    allowed=False,
                    reason=(
                        "Potential path traversal attempt "
                        "detected."
                    ),
                    category="path_traversal",
                )

        # ------------------------------------------------------
        # Tool call allowed
        # ------------------------------------------------------

        return GuardrailResult(
            allowed=True,
        )

    # ==========================================================
    # Helpers
    # ==========================================================

    @staticmethod
    def _normalize_arguments(
        arguments: Any,
    ) -> dict:

        if arguments is None:
            return {}

        if isinstance(arguments, dict):
            return arguments

        # Pydantic model
        if hasattr(arguments, "model_dump"):
            return arguments.model_dump()

        # Generic object
        if hasattr(arguments, "__dict__"):
            return vars(arguments)

        return {
            "value": str(arguments)
        }

    @staticmethod
    def _arguments_to_text(
        arguments: dict,
    ) -> str:

        return " ".join(
            str(value)
            for value in arguments.values()
        )


# ==========================================================
# Shared Instance
# ==========================================================

_tool_guard = ToolGuard()


def get_tool_guard() -> ToolGuard:
    """
    Return the shared ToolGuard instance.
    """

    return _tool_guard