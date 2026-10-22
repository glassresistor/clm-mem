"""Tool specs: the contract shared by the LLM tool loop, the MCP surface and
the web/CLI UX.

A :class:`ToolSpec` is a name + JSON schema + a *category* (``web``, ``exec``
or ``fs``) plus a ``read_only`` flag. Categories drive the UX: the web studio
composer shows one toggle per category, the CLI offers ``/tools on web``, and
the approval policy decides whether a call may run without asking ("read-only
tools never need approval; mutating ones may, depending on session policy").
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

CATEGORY_WEB = "web"
CATEGORY_EXEC = "exec"
CATEGORY_FS = "fs"
CATEGORIES = (CATEGORY_WEB, CATEGORY_EXEC, CATEGORY_FS)


@dataclass
class ToolCall:
    """One request to run a tool, as emitted by the model or the UX."""
    id: str
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)   # provider payload
    origin: str = "model"             # model | user | mcp


@dataclass
class ToolResult:
    """Outcome of one tool call; ``detail`` is what the cards/CLI expand to."""
    ok: bool
    summary: str                       # one line, shown collapsed
    detail: str = ""                   # full output, shown expanded
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_text(self, limit: int = 6000) -> str:
        """Render for the model's context (what it reads back)."""
        head = self.summary if self.ok else f"ERROR: {self.error or self.summary}"
        body = self.detail or ""
        if len(body) > limit:
            body = body[:limit] + f"\n...[truncated {len(body) - limit} chars]"
        return f"{head}\n{body}".strip() if body else head


@dataclass
class ToolSpec:
    name: str
    description: str
    params: Dict[str, Any]                       # JSON schema (object)
    category: str = CATEGORY_EXEC
    read_only: bool = False
    handler: Optional[Callable[[Dict[str, Any]], ToolResult]] = None

    def __post_init__(self) -> None:
        if not NAME_RE.match(self.name):
            raise ValueError(f"invalid tool name {self.name!r}")

    # -- provider shapes -------------------------------------------------
    def openai_schema(self) -> Dict[str, Any]:
        """Function-call spec for OpenAI-compatible ``tools=[...]``."""
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.params}}

    def mcp_schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "inputSchema": self.params}


def parse_llm_tool_calls(message: Dict[str, Any]) -> List[ToolCall]:
    """Normalize ``message.tool_calls`` (OpenAI shape) into ToolCalls."""
    out: List[ToolCall] = []
    for i, c in enumerate(message.get("tool_calls") or []):
        fn = c.get("function") or {}
        args = fn.get("arguments") or "{}"
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except Exception:
                parsed = {"_raw": args}
        elif not isinstance(args, dict):
            parsed = {"_raw": str(args)}
        else:
            parsed = args
        out.append(ToolCall(id=c.get("id") or f"call-{i}",
                            name=str(fn.get("name", "")),
                            args=dict(parsed), raw=dict(c), origin="model"))
    return out


def render_call(name: str, args: Dict[str, Any], width: int = 90) -> str:
    """Compact human label for a call: ``web_search "qwen 3"``."""
    parts = []
    for k, v in args.items():
        if k == "_raw":
            parts.append(str(v))
            continue
        s = v if isinstance(v, str) else json.dumps(v)
        if len(s) > width:
            s = s[:width] + "..."
        parts.append(f"{k}={s}")
    return f"{name}({', '.join(parts)})"


def clamp_int(value, default: int, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))
