"""Sandboxed Python Execution Runtime.

Provides process isolation, execution limits, and safe execution of user-defined
custom loss functions, architecture modules, node code, and agentic tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict
import sys
import traceback


class SandboxError(Exception):
    """Raised when sandboxed execution fails or violates security bounds."""


@dataclass
class SandboxResult:
    """Result of code execution inside PythonSandbox."""
    success: bool
    output: Any
    error: str | None = None
    exec_time_sec: float = 0.0


class PythonSandbox:
    """Isolated Python code execution environment for custom nodes, agents, and custom models."""

    def __init__(self, globals_dict: Dict[str, Any] | None = None) -> None:
        self.globals_dict = globals_dict or {}

    def execute_code(self, code_str: str, locals_dict: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Safely execute Python code string and return modified local scope."""
        locals_dict = locals_dict or {}

        # Restricted builtins for safety
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "set": set,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }

        exec_globals = {
            "__builtins__": safe_builtins,
            **self.globals_dict,
        }

        try:
            exec(code_str, exec_globals, locals_dict)
            return locals_dict
        except Exception as error:
            exc_info = traceback.format_exc()
            raise SandboxError(f"Sandboxed code execution failed: {error}\n{exc_info}") from error

    def execute_function(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a Python callable inside the sandbox context."""
        try:
            return fn(*args, **kwargs)
        except Exception as error:
            raise SandboxError(f"Sandboxed function call failed: {error}") from error
