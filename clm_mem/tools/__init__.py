"""Agent tools for clm-mem: web search, sandboxed commands, file work.

The package keeps the tool layer next to the memory layer on purpose: tool
output is fed back into the model as short-term context, so what an agent does
and what it remembers follow the same budget and eviction rules as chat.
"""
from __future__ import annotations

from .sandbox import ExecResult, Sandbox
from .spec import (CATEGORIES, CATEGORY_EXEC, CATEGORY_FS, CATEGORY_WEB,
                   ToolCall, ToolResult, ToolSpec, parse_llm_tool_calls,
                   render_call)
from .manager import Approval, ApprovalBroker, ToolManager
from .web import PROVIDERS, WebError, fetch_page, run_search

__all__ = ["Approval", "ApprovalBroker", "ToolManager", "ExecResult",
           "Sandbox", "ToolCall", "ToolResult", "ToolSpec", "WebError",
           "CATEGORIES", "CATEGORY_WEB", "CATEGORY_EXEC", "CATEGORY_FS",
           "parse_llm_tool_calls", "render_call", "run_search", "fetch_page",
           "PROVIDERS"]
