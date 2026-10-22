"""ToolManager + ApprovalBroker: the agent side of one memory session.

Everything an agent can *do* lives here, next to everything it can *remember*:
the manager owns the sandbox, the web provider, the tool registry and a bounded
activity log; the broker decides whether a call may run without asking. The
harness feeds executed results back into the model as short-term context blocks,
so tool output participates in the same budget/eviction machinery as chat --
that is deliberate: an agent's scratchpad should be forgettable working state.

Policies (``approval_mode``):
  ``auto``   read-only tools always run; mutating tools run inside the sandbox
  ``ask``    mutating tools pause until a human approves in the web studio,
             the CLI prompt, or over REST/MCP
  ``deny``   mutating tools never run (web + read-only file tools only)
"""
from __future__ import annotations

import json
import os
import shlex
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .sandbox import Sandbox
from .spec import (CATEGORIES, CATEGORY_EXEC, CATEGORY_FS, CATEGORY_WEB, ToolCall,
                   ToolResult, ToolSpec, render_call)
from .web import fetch_page, run_search, WebError

#: categories shown in the UX (composer toggles, CLI /tools output)
CATEGORIES_ON = (CATEGORY_WEB, CATEGORY_EXEC, CATEGORY_FS)


@dataclass
class Approval:
    """A pending (or resolved) request to run a mutating tool."""
    id: str
    tool: str
    args: Dict[str] = field(default_factory=dict)
    origin: str = "model"
    created: float = field(default_factory=time.time)
    decided: Optional[bool] = None
    by: str = ""
    resolved: float = 0.0

    def to_dict(self) -> Dict:
        return {"id": self.id, "tool": self.tool, "args": self.args,
                "origin": self.origin, "created": round(self.created, 3),
                "decided": self.decided, "by": self.by,
                "age_s": round((self.resolved or time.time()) - self.created, 1)}


class ApprovalBroker:
    """Blocks a tool call until a human decides, or times out denying it."""

    def __init__(self, mode: str = "auto", timeout: float = 300.0):
        self.mode = mode if mode in ("auto", "ask", "deny") else "auto"
        self.timeout = float(timeout)
        self.pending: Dict[str, Approval] = {}
        self.resolved: List[Approval] = []
        self._lock = threading.Lock()
        self._seq = 0
        #: optional blocking responder (set by the CLI prompt loop)
        self.responder: Optional[Callable[[Approval], bool]] = None

    def needs_approval(self, spec: ToolSpec) -> bool:
        """Read-only tools never ask; mutating ones do under ask / deny."""
        return self.mode in ("ask", "deny") and not spec.read_only

    def request(self, call: ToolCall, spec: ToolSpec) -> bool:
        """Blocking decision request; True means run the tool."""
        if self.mode == "auto":
            return True
        if self.mode == "deny":
            return False
        with self._lock:
            self._seq += 1
            rec = Approval(id=f"a{self._seq}", tool=call.name,
                           args=dict(call.args), origin=call.origin)
            self.pending[rec.id] = rec
        waiter = threading.Event()
        rec._waiter = waiter                          # type: ignore[attr-defined]

        if self.responder is not None:
            try:
                ok = bool(self.responder(rec))
            except Exception:
                ok = False
            return self._finish(rec, ok, "responder")
        deadline = time.time() + max(5.0, self.timeout)
        while True:
            with self._lock:
                if rec.decided is not None:
                    break
            if time.time() > deadline:
                break
            time.sleep(0.25)
        return self._finish(rec, bool(rec.decided), rec.by or "timeout")

    def _finish(self, rec: Approval, ok: bool, by: str) -> bool:
        with self._lock:
            rec = self.pending.get(rec.id, rec)
            if rec.decided is None:
                rec.decided = ok
                rec.by = by
                rec.resolved = time.time()
            else:
                ok = bool(rec.decided)
            self.pending.pop(rec.id, None)
            self.resolved.append(rec)
            self.resolved[:] = self.resolved[-24:]
            w = getattr(rec, "_waiter", None)
        if w is not None:
            w.set()
        return ok

    def decide(self, approval_id: str, approve: bool, by: str = "web") -> Dict:
        """Resolve a pending approval from the UI / CLI / REST side."""
        with self._lock:
            rec = self.pending.get(approval_id)
            if rec is None:
                for r in reversed(self.resolved):
                    if r.id == approval_id:
                        return {"id": approval_id, "decided": r.decided,
                                "by": r.by, "note": "already decided"}
                return {"error": f"unknown approval {approval_id!r}"}
        ok = self._finish(rec, bool(approve), by)
        return rec.to_dict() | {"ok": ok}

    def pending_list(self) -> List[Dict]:
        with self._lock:
            return [a.to_dict() for a in self.pending.values()]


class ToolManager:
    """Per-session agent tools: registry, sandbox, approvals and activity log."""

    def __init__(self, cfg, root: str = "", responder=None):
        """
        Args:
            cfg: the session's HarnessConfig (sandbox/policy/search knobs).
            root: sandbox directory (``<workdir>/sandbox`` for local sessions).
            responder: optional blocking approval callback (CLI prompt).
        """
        self.cfg = cfg
        self.sandbox = Sandbox(
            root=root, mode=getattr(cfg, "sandbox_mode", "jail"),
            timeout=float(getattr(cfg, "tool_timeout", 30.0)),
            output_limit=int(getattr(cfg, "exec_output_limit", 8000)),
            disk_mb=int(getattr(cfg, "sandbox_disk_mb", 256)),
            mem_mb=int(getattr(cfg, "sandbox_mem_mb", 512)),
            image=str(getattr(cfg, "sandbox_image", "")))
        self.enabled: Dict[str, bool] = {}
        for s in self._specs():
            self.enabled[s.name] = True
        self.broker = ApprovalBroker(
            getattr(cfg, "approval_mode", "auto"),
            float(getattr(cfg, "approval_timeout", 300.0)))
        self.broker.responder = responder
        self.log: List[Dict] = []
        self._lock = threading.Lock()
        self._seq = 0
        self.registry: Dict[str, ToolSpec] = {}
        for s in self._specs():
            self.registry[s.name] = s

    # -- registry ---------------------------------------------------------
    def _specs(self) -> List[ToolSpec]:
        return [
            ToolSpec(
                name="web_search", category=CATEGORY_WEB, read_only=True,
                description=("Search the web. Returns ranked results with title, "
                             "url and snippet; use web_fetch on a result to read "
                             "the page. Never invent URLs you did not get here."),
                handler=self._h_web_search,
                params={"type": "object",
                        "properties": {
                            "query": {"type": "string",
                                      "description": "search query"},
                            "max_results": {"type": "integer", "default": 5,
                                            "minimum": 1, "maximum": 10}},
                        "required": ["query"]}),
            ToolSpec(
                name="web_fetch", category=CATEGORY_WEB, read_only=True,
                description="Fetch one URL and return its readable text.",
                handler=self._h_web_fetch,
                params={"type": "object",
                        "properties": {"url": {"type": "string"},
                                       "max_chars": {"type": "integer",
                                                     "default": 6000}},
                        "required": ["url"]}),
            ToolSpec(
                name="run_command", category=CATEGORY_EXEC,
                description=("Run a shell command inside the session sandbox. "
                             "cwd is the sandbox root; files written there are "
                             "the project files. Returns stdout+stderr and exit "
                             "code."),
                handler=self._h_exec,
                params={"type": "object",
                        "properties": {
                            "command": {"type": "string",
                                        "description": "shell command line"},
                            "cwd": {"type": "string",
                                    "description": "sandbox-relative directory"},
                            "timeout": {"type": "number"}},
                        "required": ["command"]}),
            ToolSpec(
                name="run_python", category=CATEGORY_EXEC,
                description=("Write and run a self-contained Python snippet in "
                             "the sandbox (preferred over run_command for code). "
                             "The snippet is saved as main.py-style scratch."),
                handler=self._h_run_python,
                params={"type": "object",
                        "properties": {"code": {"type": "string"},
                                       "timeout": {"type": "number"}},
                        "required": ["code"]}),
            ToolSpec(
                name="write_file", category=CATEGORY_FS,
                description=("Create or overwrite a file inside the sandbox "
                             "(paths are sandbox-relative)."),
                handler=self._h_write_file,
                params={"type": "object",
                        "properties": {"path": {"type": "string"},
                                       "content": {"type": "string"},
                                       "append": {"type": "boolean",
                                                  "default": False}},
                        "required": ["path", "content"]}),
            ToolSpec(
                name="read_file", category=CATEGORY_FS, read_only=True,
                description="Read a sandbox file (sandbox-relative path).",
                handler=self._h_read_file,
                params={"type": "object",
                        "properties": {"path": {"type": "string"},
                                       "max_chars": {"type": "integer",
                                                     "default": 20000}},
                        "required": ["path"]}),
            ToolSpec(
                name="list_files", category=CATEGORY_FS, read_only=True,
                description="List files in the sandbox (optionally a subdir).",
                handler=self._h_list_files,
                params={"type": "object",
                        "properties": {"path": {"type": "string"},
                                       "limit": {"type": "integer",
                                                 "default": 200}}}),
        ]

    # -- enablement ---------------------------------------------------------
    @property
    def active(self) -> bool:
        return bool(self.cfg.tools_enabled and any(self.enabled.values()))

    def category_state(self, cat: str) -> str:
        """``on`` / ``off`` / ``partial`` for one tool category."""
        names = [n for n, s in self.registry.items() if s.category == cat]
        if not names:
            return "off"
        on = sum(1 for n in names if self.enabled.get(n))
        return "on" if on == len(names) else ("off" if on == 0 else "partial")

    def toggle(self, target: str, on: Optional[bool] = None) -> Dict:
        """Enable/disable a category (``web``/``exec``/``fs``/``all``) or tool.

        Passing ``on=None`` flips the current state (a bare ``/tools web``).
        """
        if target == "all":
            value = any(not self.enabled.get(n) for n in self.registry) \
                if on is None else bool(on)
            self.enabled = {n: bool(value) for n in self.registry}
            if value:
                self.cfg.tools_enabled = True
            return self.state()
        if target in CATEGORIES_ON:
            names = [n for n, s in self.registry.items() if s.category == target]
            value = (self.category_state(target) != "on") if on is None \
                else bool(on)
            for n in names:
                self.enabled[n] = bool(value)
            if value:
                self.cfg.tools_enabled = True
            return self.state()
        spec = self.registry.get(target)
        if spec is None:
            raise ValueError(f"unknown tool or category {target!r} "
                             f"(categories: web, exec, fs, all)")
        value = (not self.enabled.get(target, False)) if on is None \
            else bool(on)
        self.enabled[target] = bool(value)
        if value:
            self.cfg.tools_enabled = True
        return self.state()

    def set_master(self, on: bool) -> Dict:
        """Master switch: the tool loop only runs when this (and a category)
        is on. Toggling any tool on also turns the master on."""
        self.cfg.tools_enabled = bool(on)
        if not on:
            self.enabled = {n: False for n in self.registry}
        else:
            self.enabled = {n: True for n in self.registry}
        return self.state()

    def set_policy(self, mode: str) -> Dict:
        if mode not in ("auto", "ask", "deny"):
            raise ValueError("approval_mode must be auto | ask | deny")
        self.broker.mode = mode
        return self.state()

    def state(self) -> Dict:
        enabled = [n for n in self.registry if self.enabled.get(n)]
        return {"tools_enabled": bool(self.cfg.tools_enabled),
                "enabled_categories": {c: self.category_state(c)
                                       for c in CATEGORIES_ON},
                "enabled_tools": enabled,
                "approval_mode": self.broker.mode,
                "sandbox": self.sandbox.info(),
                "search_provider": getattr(self.cfg, "search_provider", ""),
                "max_tool_calls": int(getattr(self.cfg, "max_tool_calls", 8)),
                "active": bool(self.active)}

    def llm_tools(self) -> List[Dict]:
        if not self.active:
            return []
        return [s.openai_schema() for s in self.registry.values()
                if self.enabled.get(s.name)]

    def describe(self, limit: int = 100) -> List[Dict]:
        out = []
        for s in self.registry.values():
            d = {"name": s.name, "description": s.description,
                 "category": s.category, "read_only": s.read_only,
                 "enabled": bool(self.enabled.get(s.name))}
            out.append(d)
        return out[:limit]

    def mcp_surface(self) -> List[Dict]:
        return [s.mcp_schema() for s in self.registry.values()]

    # -- execution -----------------------------------------------------------
    def _start_entry(self, call: ToolCall) -> Dict:
        with self._lock:
            self._seq += 1
            entry = {"seq": self._seq, "id": f"t{self._seq}",
                     "tool": call.name, "args": dict(call.args),
                     "origin": call.origin, "status": "running",
                     "started": time.time(), "finished": None}
            self.log.append(entry)
            self.log[:] = self.log[-60:]
        return entry

    def _finish_entry(self, entry: Dict, result: ToolResult) -> None:
        entry["status"] = "ok" if result.ok else "error"
        entry["summary"] = result.summary
        entry["detail"] = (result.detail or "")[:20_000]
        entry["data"] = result.data
        entry["error"] = result.error
        entry["finished"] = time.time()
        entry["duration_s"] = round(entry["finished"] - entry["started"], 3)

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call (with approval policy), logging both outcomes."""
        spec = self.registry.get(call.name)
        if spec is None:
            return ToolResult(False, f"unknown tool {call.name!r}",
                              error=f"unknown tool {call.name!r}")
        if not self.cfg.tools_enabled or not self.enabled.get(spec.name):
            return ToolResult(False, f"{call.name} is switched off",
                              error="tool disabled for this session")
        entry = self._start_entry(call)
        # human-initiated calls (composer buttons, CLI slash commands, MCP
        # sandbox exec) are approved by definition; only model-initiated ones
        # can be pending.
        if call.origin != "user" and self.broker.needs_approval(spec):
            ok = self.broker.request(call, spec)
            if not ok:
                res = ToolResult(False, f"{call.name}: approval denied",
                                 error="approval denied",
                                 data={"approval": "denied"})
                self._finish_entry(entry, res)
                entry["status"] = "denied"
                return res
        try:
            res = spec.handler(call.args)   # type: ignore[misc]
        except Exception as exc:            # tools never crash the session
            res = ToolResult(False, f"{call.name} failed",
                             error=f"{type(exc).__name__}: {exc}",
                             detail=str(exc))
        self._finish_entry(entry, res)
        return res

    def run_user(self, name: str, args: Dict[str, Any], origin: str = "user") -> ToolResult:
        """Direct invocation from the UX (composer buttons, CLI slash cmds)."""
        call = ToolCall(id=f"u{int(time.time() * 1000) % 100000}", name=name,
                        args=dict(args), origin=origin)
        return self.execute(call)

    # -- handlers -----------------------------------------------------------
    def _h_web_search(self, args: Dict) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(False, "web_search needs a query")
        limit = int(args.get("max_results") or 5)
        provider = getattr(self.cfg, "search_provider", "") or "ddg"
        try:
            hits = run_search(provider, query, limit=min(limit, 10))
        except WebError as exc:
            return ToolResult(False, f"web_search: {exc}", error=str(exc))
        except Exception as exc:
            return ToolResult(False, "web_search failed", error=str(exc),
                              detail=f"{type(exc).__name__}: {exc}")
        lines = []
        for i, h in enumerate(hits, 1):
            sn = (h.get("snippet") or "").replace("\n", " ").strip()
            if len(sn) > 240:
                sn = sn[:240] + "..."
            lines.append(f"{i}. {h.get('title', '')}\n   {h.get('url', '')}\n   {sn}")
        return ToolResult(
            ok=True, summary=f'web_search "{query}" -> {len(hits)} hits',
            detail="\n".join(lines), data={"hits": hits,
                                            "provider": provider})

    def _h_web_fetch(self, args: Dict) -> ToolResult:
        url = str(args.get("url", "")).strip()
        if not url:
            return ToolResult(False, "web_fetch needs a url")
        limit = int(args.get("max_chars") or getattr(self.cfg, "fetch_max_chars", 6000))
        try:
            page = fetch_page(url, limit=min(limit, 20_000))
        except Exception as exc:
            return ToolResult(False, f"web_fetch {url} failed",
                              error=str(exc)[:300])
        head = str(page.get("title") or "")[:120]
        return ToolResult(ok=True, summary=f"fetched {url} "
                        f"({page['chars']} chars){': ' + head if head else ''}",
                          detail=str(page.get("text", "")), data=page)

    def _h_exec(self, args: Dict) -> ToolResult:
        command = str(args.get("command", "")).strip()
        if not command:
            return ToolResult(False, "run_command needs a command")
        timeout = args.get("timeout")
        res = self.sandbox.exec(command, timeout=float(timeout) if timeout else None,
                                cwd=str(args.get("cwd", "")))
        detail = res.stdout.strip() or ("(no output) exit "
                                        f"{res.code}")
        label = command.replace("\n", " ").strip()
        if len(label) > 80:
            label = label[:80] + "…"
        return ToolResult(
            ok=res.ok, summary=f"$ {label} -> exit {res.code}"
            + (f" ({res.note})" if res.note else ""),
            detail=detail,
            data={"command": command, "exit": res.code,
                  "isolation": res.isolation, "duration_s": res.duration,
                  "note": res.note})

    def _h_run_python(self, args: Dict) -> ToolResult:
        code = str(args.get("code", ""))
        if not code.strip():
            return ToolResult(False, "run_python needs code")
        with self._lock:
            self._seq += 1
            n = self._seq
        script = f"scratch/snippet_{n}.py"
        self.sandbox.write_file(script, code)
        runner = "python3" if self.sandbox.mode == "docker" else sys.executable
        cmd = f"{shlex.quote(runner)} {shlex.quote(script)}"
        timeout = args.get("timeout")
        res = self.sandbox.exec(cmd, timeout=float(timeout) if timeout else None)
        detail = res.stdout.strip() or "(no output)"
        return ToolResult(
            ok=res.ok, summary=f"python {script} -> exit {res.code}",
            detail=f"[{script}]\n{code}\n--- output ---\n{detail}",
            data={"script": script, "exit": res.code,
                  "isolation": res.isolation, "duration_s": res.duration})

    def _h_write_file(self, args: Dict) -> ToolResult:
        path = str(args.get("path", "")).strip()
        content = str(args.get("content", ""))
        if not path:
            return ToolResult(False, "write_file needs a path")
        info = self.sandbox.write_file(path, content,
                                       append=bool(args.get("append")))
        return ToolResult(ok=True,
                          summary=f"wrote {info['bytes']} bytes -> {info['path']}",
                          detail=content[:2000], data=info)

    def _h_read_file(self, args: Dict) -> ToolResult:
        path = str(args.get("path", "")).strip()
        if not path:
            return ToolResult(False, "read_file needs a path")
        limit = int(args.get("max_chars") or 20_000)
        try:
            out = self.sandbox.read_file(path, limit=min(limit, 50_000))
        except FileNotFoundError as exc:
            return ToolResult(False, f"read_file: {exc}", error=str(exc))
        return ToolResult(ok=True, summary=f"read {path} ({out['chars']} chars)",
                          detail=out["text"], data=out)

    def _h_list_files(self, args: Dict) -> ToolResult:
        sub = str(args.get("path") or ".")
        try:
            files = self.sandbox.list_files(sub,
                                            max_entries=int(args.get("limit", 200)))
        except (OSError, ValueError) as exc:
            return ToolResult(False, f"list_files: {exc}", error=str(exc))
        listing = "\n".join(f"{f['bytes']:>8}  {f['path']}" for f in files)
        return ToolResult(ok=True, summary=f"{len(files)} file(s) in {sub}",
                          detail=listing or "(empty)", data={"files": files})

    # -- activity --------------------------------------------------------------
    def activity(self, limit: int = 24, full: bool = False) -> List[Dict]:
        with self._lock:
            tail = self.log[-limit:]
        out = []
        for e in tail[::-1]:
            d = dict(e)
            if not full:
                d["detail"] = (d.get("detail") or "")[:900]
            out.append(d)
        return out

    def approval_history(self, limit: int = 12) -> List[Dict]:
        with self.broker._lock:      # noqa: access via broker API
            hist = [a.to_dict() for a in self.broker.resolved[-limit:]]
        return hist

    def approvals(self) -> Dict:
        return {"pending": self.broker.pending_list(),
                "history": self.approval_history(),
                "mode": self.broker.mode}
