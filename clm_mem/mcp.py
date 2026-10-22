"""MCP (Model Context Protocol) server embedded in `clm-mem serve`.

Implements the streamable-HTTP transport subset that matters for tool
clients: JSON-RPC over POST /mcp with ``initialize``, ``tools/list`` and
``tools/call``. The exposed tools mirror exactly what the web studio and
``clm-mem chat`` can do -- they all drive the same in-process library, so an
MCP client (Unsloth Studio's integrations panel, Claude/Cline-style agents)
gets identical long/short-term memory per workspace sandbox.
"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from .server import VERSION as SERVER_VERSION
from .tools import ToolCall

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "clm-mem",
               "title": "clm-mem memory harness",
               "version": SERVER_VERSION}

_WS_DEFAULT = "default"


def _sid(args: Dict[str, Any]) -> str:
    ws = str(args.get("workspace") or _WS_DEFAULT).strip()
    return f"ws:{ws}"


def _obj(props: Dict[str, Any], required: List[str]) -> Dict:
    return {"type": "object", "properties": props, "required": required}


_STR = {"type": "string"}
_WS = {"type": "string",
       "description": "workspace/project sandbox name (default: 'default')"}

TOOLS: List[Dict] = [
    {"name": "clm_chat",
     "description": ("Send a message to the clm-mem memory-managed chat for a "
                     "workspace. The harness classifies context as longterm / "
                     "shortterm / transient and enforces the token budget."),
     "inputSchema": _obj({"message": _STR, "workspace": _WS}, ["message"])},
    {"name": "clm_status",
     "description": ("Inspect a workspace session: turn count, live tokens vs "
                     "budget, per-tier block counts, archive size."),
     "inputSchema": _obj({"workspace": _WS}, [])},
    {"name": "clm_blocks",
     "description": ("List the live context blocks of a workspace session with "
                     "their memory tags and token counts (previews unless "
                     "full=true)."),
     "inputSchema": _obj({"workspace": _WS,
                          "tag": {"type": "string",
                                  "enum": ["longterm", "shortterm",
                                          "transient"]},
                          "full": {"type": "boolean"}}, [])},
    {"name": "clm_recall",
     "description": ("Substring-search the eviction archive of a workspace -- "
                     "memory that was dropped/offloaded is still retrievable."),
     "inputSchema": _obj({"query": _STR, "workspace": _WS,
                          "limit": {"type": "integer", "default": 50}},
                         ["query"])},
    {"name": "clm_reinject",
     "description": ("Pull an archived block back into the live context of a "
                     "workspace as shortterm."),
     "inputSchema": _obj({"block_id": {"type": "integer"},
                          "workspace": _WS}, ["block_id"])},
    {"name": "clm_set_keep_pct",
     "description": ("Tune the memory knob (fraction of shortterm kept per "
                     "FIFO eviction round) live, in [0, 1)."),
     "inputSchema": _obj({"pct": {"type": "number"}, "workspace": _WS},
                         ["pct"])},
    {"name": "clm_set_forget",
     "description": ("Enable/disable the forgetting strategies live: decay "
                     "(forgetting-curve demotion of stale longterm plus "
                     "value-ranked offload) and supersede (updates overwrite "
                     "older longterm, duplicates dropped at write time). "
                     "Omitted flags leave the current state untouched."),
     "inputSchema": _obj({"decay": {"type": "boolean"},
                          "supersede": {"type": "boolean"}, "workspace": _WS},
                         [])},
    {"name": "clm_workspaces",
     "description": ("List project workspaces (sandboxes) with their active "
                     "session stats."),
     "inputSchema": _obj({}, [])},

    # -- agent tools: identical to the web studio toggles and CLI slash cmds --
    {"name": "clm_tools",
     "description": ("Inspect a workspace session's agent tools: which are "
                     "enabled (web / exec / fs categories), the approval "
                     "policy, sandbox isolation level and recent activity."),
     "inputSchema": _obj({"workspace": _WS,
                          "activity": {"type": "integer", "default": 10}},
                         [])},
    {"name": "clm_tool_on",
     "description": ("Enable/disable agent tools live: target is a category "
                     "(web, exec, fs) or a tool name (all toggles every one). "
                     "Also accepts approval_mode=auto|ask|deny."),
     "inputSchema": _obj({"target": _STR,
                          "on": {"type": "boolean"},
                          "approval_mode": {"type": "string",
                                           "enum": ["auto", "ask", "deny"]},
                          "workspace": _WS}, ["target"])},
    {"name": "clm_tool_call",
     "description": ("Run one clm-mem tool directly (web_search, web_fetch, "
                     "run_command, run_python, write_file, read_file, "
                     "list_files). Subject to the session's approval policy; "
                     "read-only tools always run."),
     "inputSchema": _obj({"tool": _STR,
                          "args": {"type": "object", "description":
                                   "tool arguments"},
                          "workspace": _WS}, ["tool"])},
    {"name": "clm_sandbox_exec",
     "description": ("Run a shell command inside a workspace's sandbox "
                     "(bounded process; cwd is the sandbox root). Human-"
                     "initiated, so it bypasses approval prompts."),
     "inputSchema": _obj({"command": _STR,
                          "cwd": _STR,
                          "timeout": {"type": "number"},
                          "workspace": _WS}, ["command"])},
    {"name": "clm_web_search",
     "description": ("Search the web with the session's configured provider "
                     "(stub/ddg/brave/tavily/exa) without a model turn."),
     "inputSchema": _obj({"query": _STR,
                          "max_results": {"type": "integer", "default": 5},
                          "workspace": _WS}, ["query"])}
,
]


def _text(msg: str, is_error: bool = False) -> Dict:
    return {"content": [{"type": "text", "text": msg}], "isError": is_error}


def _agent_result(sup, sid: str, name: str, args: Dict[str, Any]) -> Dict:
    """MCP handlers for the agent surface (same library as the web studio)."""
    h = sup.harness_for(sid)
    if name == "clm_tools":
        tail = int(args.get("activity", 10) or 10)
        return _text(json.dumps({**h.tools.state(),
                                 "tools": h.tools.describe(),
                                 "activity": h.tools.activity(limit=tail)}))
    if name == "clm_tool_on":
        if args.get("approval_mode"):
            h.tools.set_policy(str(args["approval_mode"]))
        on = args.get("on")
        out = h.tools.toggle(str(args.get("target", "all")),
                             None if on is None else bool(on))
        return _text(json.dumps(out))
    if name == "clm_tool_call":
        call = ToolCall(id="mcp-call",
                        name=str(args.get("tool", "")).strip(),
                        args=dict(args.get("args") or {}), origin="mcp")
        res = h.tools.execute(call)
        return _text(f"{res.summary}\n{res.detail[:4000]}".strip(),
                     is_error=not res.ok)
    if name == "clm_sandbox_exec":
        call = ToolCall(id="mcp-exec", name="run_command",
                        args={"command": str(args.get("command", "")).strip(),
                              "cwd": str(args.get("cwd", "")),
                              "timeout": args.get("timeout")},
                        origin="user")        # human-initiated: never asks
        res = h.tools.execute(call)
        return _text(f"{res.summary}\n{res.detail[:4000]}".strip(),
                     is_error=not res.ok)
    if name == "clm_web_search":
        call = ToolCall(id="mcp-web", name="web_search",
                        args={"query": str(args.get("query", "")).strip(),
                              "max_results": int(args.get("max_results", 5))},
                        origin="user")
        res = h.tools.execute(call)
        return _text(f"{res.summary}\n{res.detail[:4000]}".strip(),
                     is_error=not res.ok)
    return _text(f"unknown agent tool {name!r}", is_error=True)


def _tool_result(sup, name: str, args: Dict[str, Any]) -> Dict:
    try:
        if name == "clm_workspaces":
            return _text(json.dumps({"workspaces": sup.list_workspaces()}))

        sid = _sid(args)
        # -- agent tools -------------------------------------------------
        if name in ("clm_tools", "clm_tool_on", "clm_tool_call",
                    "clm_sandbox_exec", "clm_web_search"):
            return _agent_result(sup, sid, name, args)
        if name == "clm_chat":
            h = sup.harness_for(sid)
            reply = h.step(str(args.get("message", "")))
            return _text(f"{reply}\n\n[context: {h.status()}]")
        if name == "clm_status":
            h = sup.harness_for(sid)
            return _text(json.dumps(h.snapshot(preview_chars=0,
                                               events_tail=5)))
        if name == "clm_blocks":
            h = sup.harness_for(sid)
            snap = h.snapshot(include_full=bool(args.get("full")))
            tag = args.get("tag")
            if tag:
                snap["blocks"] = [b for b in snap["blocks"]
                                  if b["tag"] == tag]
            return _text(json.dumps(snap))
        if name == "clm_recall":
            h = sup.harness_for(sid)
            hits = h.recall(str(args.get("query", "")),
                            limit=int(args.get("limit", 50)))
            return _text(json.dumps({"hits": hits}))
        if name == "clm_reinject":
            h = sup.harness_for(sid)
            ok = h.reinject(int(args.get("block_id", -1)))
            return _text(("re-injected as shortterm" if ok else
                          "block id not in archive"), is_error=not ok)
        if name == "clm_set_forget":
            h = sup.harness_for(sid)
            out = []
            if "decay" in args:
                h.cfg.decay_enabled = bool(args["decay"])
                out.append(f"decay_enabled={h.cfg.decay_enabled}")
            if "supersede" in args:
                h.cfg.supersede_enabled = bool(args["supersede"])
                out.append(f"supersede_enabled={h.cfg.supersede_enabled}")
            return _text("; ".join(out) or
                         f"decay_enabled={h.cfg.decay_enabled} "
                         f"supersede_enabled={h.cfg.supersede_enabled}")
        if name == "clm_set_keep_pct":
            pct = float(args.get("pct", -1))
            if not 0.0 <= pct < 1.0:
                return _text("pct must be in [0, 1)", is_error=True)
            h = sup.harness_for(sid)
            h.cfg.shortterm_keep_pct = pct
            return _text(f"shortterm_keep_pct = {pct}")
        return _text(f"unknown tool {name!r}", is_error=True)
    except Exception as exc:      # MCP tools report failures via isError
        return _text(f"{name} failed: {exc}", is_error=True)


def _err(rid, code: int, msg: str) -> Dict:
    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": code, "message": msg}}


def handle(sup, msg: Dict[str, Any]) -> Optional[Dict]:
    """Handle one JSON-RPC message; None means 'notification accepted'."""
    rid = msg.get("id")
    method = str(msg.get("method", ""))

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid,
                "result": {"protocolVersion": PROTOCOL_VERSION,
                           "capabilities": {"tools": {}},
                           "serverInfo": SERVER_INFO}}
    if method in ("notifications/initialized", "initialized"):
        return None                        # notification -> 202 no body
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        name = str(params.get("name", ""))
        args = params.get("arguments") or {}
        known = {t["name"] for t in TOOLS}
        if name not in known:
            return _err(rid, -32602, f"unknown tool {name!r}")
        result = _tool_result(sup, name, args)
        return {"jsonrpc": "2.0", "id": rid, "result": result}
    if method.startswith("notifications/"):
        return None
    return _err(rid, -32601, f"unsupported method {method!r}")
