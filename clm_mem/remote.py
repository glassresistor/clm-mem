"""Client for a running "clm-mem serve" instance.

This is what makes chat and serve two ends of the same stick: run the
server (proxy + backends + web studio), then attach one or more CLI chat
sessions to it with:  clm-mem chat --connect http://host:port

Everything the web UI can do per workspace (status, tag listings, archive
search, reinject, live keep-pct) is available at the prompt through the same
REST API the studio's app.js uses.
"""
from __future__ import annotations
import json
import urllib.error
import urllib.request
from typing import Dict, List, Optional
from .server import SESSION_HEADER


def _ns(d: Dict):
    """Lightweight object from a snapshot block dict."""
    from types import SimpleNamespace
    return SimpleNamespace(**d)


class RemoteConfig:
    """Proxy for the live harness config of a remote session.

    Attribute writes are pushed to the server (POST .../settings), so
    remote.cfg.shortterm_keep_pct = 0.8 behaves exactly like it does on a
    local MemoryHarness.
    """

    _FIELDS = ("shortterm_keep_pct", "max_context_tokens",
               "systemone_evict_threshold", "decay_enabled",
               "supersede_enabled")

    _DEFAULTS = {"shortterm_keep_pct": 0.5, "max_context_tokens": 8192,
                 "systemone_evict_threshold": 0.7,
                 "decay_enabled": False, "supersede_enabled": False}

    def __init__(self, remote: "RemoteHarness", values: Dict):
        object.__setattr__(self, "_remote", remote)
        for f in self._FIELDS:
            v = values.get(f, self._DEFAULTS[f])
            object.__setattr__(self, f, v)

    def __setattr__(self, name: str, value) -> None:
        if name not in self._FIELDS:
            raise AttributeError(f"no remote knob {name!r}")
        object.__setattr__(self, name, value)
        self._remote.push_settings({name: value})

    def __getattr__(self, name: str):
        raise AttributeError(f"no remote knob {name!r}")


class RemoteHarness:
    """Talks to a live serve session over its REST API (same library underneath)."""

    def __init__(self, base_url: str, session: str = "ws:cli",
                 api_key: str = "EMPTY"):
        # base_url: the OpenAI-style public base, e.g. http://127.0.0.1:8080/v1
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.api_key = api_key
        root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1")             else self.base_url
        self.root = root
        self.cfg = RemoteConfig(self, {})

    # -- transport -----------------------------------------------------------
    def _req(self, method: str, path: str, body: Optional[dict] = None,
             extra_headers: Dict[str, str] | None = None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        req = urllib.request.Request(f"{self.root}{path}", data=data,
                                     method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                raw = r.read()
                return json.loads(raw.decode()) if raw else {}
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{e.code}: {e.read().decode()[:200]}") from e

    @property
    def _snap_path(self) -> str:
        from urllib.parse import quote
        return f"/v1/clm/sessions/{quote(self.session, '')}"

    def push_settings(self, settings: Dict) -> None:
        self._req("POST", f"{self._snap_path}/settings", settings)

    def prime(self) -> "RemoteHarness":
        """Mirror the server-side live knobs into .cfg (one snapshot call)."""
        try:
            snap = self.snapshot()
        except Exception:
            snap = {}
        self.cfg = RemoteConfig(self, snap)
        return self

    # -- harness-shaped surface -------------------------------------------------
    def step(self, user_text: str) -> str:
        resp = self._req("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": user_text}],
        }, extra_headers={SESSION_HEADER: self.session})
        reply = resp["choices"][0]["message"]["content"]
        # pull live settings/budget so local knobs mirror the server truth
        snap = self.snapshot()
        self.cfg = RemoteConfig(self, snap)
        return reply

    def snapshot(self, full: bool = False) -> Dict:
        return self._req("GET", f"{self._snap_path}/snapshot"
                         + ("?full=1" if full else ""))

    def status(self) -> str:
        try:
            s = self.snapshot()
        except Exception:
            return "turn=0 blocks=0 (new session -- send a message to create it)"
        n = s["counts"]
        return (f"turn={s['turn']} tokens={s['tokens']}/{s['budget']} "
                f"blocks={len(s['blocks'])} (long={n['longterm']} "
                f"short={n['shortterm']} transient={n['transient']})")

    def counts(self) -> Dict[str, int]:
        return self.snapshot()["counts"]

    def live_blocks(self, tag: str = ""):
        s = self.snapshot()
        return [_ns(b) for b in s["blocks"] if not tag or b["tag"] == tag]

    def recall(self, query: str, limit: int = 50) -> List[dict]:
        from urllib.parse import quote
        d = self._req("GET", f"{self._snap_path}/archive"
                      f"?q={quote(query)}&limit={limit}")
        return d.get("hits", [])

    def reinject(self, block_id: int) -> bool:
        try:
            self._req("POST", f"{self._snap_path}/reinject",
                      {"block_id": int(block_id)})
            return True
        except RuntimeError:
            return False

    # -- agent tools (same surface the web studio drives) -------------------
    def tools(self) -> Dict:
        return self._req("GET", f"{self._snap_path}/tools")

    def set_tools(self, target: str = "all", on: Optional[bool] = None,
                  approval_mode: str = "") -> Dict:
        body: Dict[str, object] = {}
        if approval_mode:
            body["approval_mode"] = approval_mode
        else:
            body["target"] = target
            if on is not None:
                body["on"] = bool(on)
        return self._req("POST", f"{self._snap_path}/tools", body)

    def activity(self, limit: int = 24, full: bool = False) -> List[Dict]:
        d = self._req("GET", f"{self._snap_path}/activity"
                      f"?limit={int(limit)}&full={'1' if full else '0'}")
        return d.get("activity", [])

    def approvals(self) -> Dict:
        return self._req("GET", f"{self._snap_path}/approvals")

    def approve(self, approval_id: str, ok: bool = True,
                by: str = "cli") -> Dict:
        return self._req("POST", f"{self._snap_path}/approvals",
                         {"id": approval_id, "approve": bool(ok), "by": by})

    def tool_call(self, name: str, args: Optional[Dict] = None) -> Dict:
        """Run any session tool directly (web_fetch, run_python, ...)."""
        return self._req("POST", f"{self._snap_path}/toolcall",
                         {"tool": name, "args": dict(args or {})})

    def web_search(self, query: str, limit: int = 5) -> Dict:
        return self._req("POST", f"{self._snap_path}/search",
                         {"query": query, "max_results": int(limit)})

    # -- sandbox -----------------------------------------------------------
    def sandbox_files(self, path: str = ".") -> Dict:
        from urllib.parse import quote as q
        return self._req("GET", f"{self._snap_path}/sandbox"
                         f"?path={q(path)}")

    def sandbox_read(self, path: str, limit: int = 20_000) -> Dict:
        from urllib.parse import quote as q
        return self._req("GET", f"{self._snap_path}/sandbox/file"
                         f"?path={q(path)}&max={int(limit)}")

    def sandbox_write(self, path: str, content: str,
                      append: bool = False) -> Dict:
        return self._req("POST", f"{self._snap_path}/sandbox/write",
                         {"path": path, "content": content,
                          "append": bool(append)})

    def sandbox_exec(self, command: str, cwd: str = "",
                     timeout: Optional[float] = None) -> Dict:
        body: Dict[str, object] = {"command": command}
        if cwd:
            body["cwd"] = cwd
        if timeout:
            body["timeout"] = float(timeout)
        return self._req("POST", f"{self._snap_path}/sandbox/exec", body)

    def sandbox_delete(self, path: str) -> Dict:
        return self._req("POST", f"{self._snap_path}/sandbox/delete",
                         {"path": path})

    def sandbox_reset(self) -> Dict:
        from urllib.parse import quote as q
        return self._req("DELETE", f"{self._snap_path}/sandbox")

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> None:
        self._req("POST", f"{self._snap_path}/reset")

    def ui_url(self) -> Optional[str]:
        try:
            h = self._req("GET", "/health")
            return self.root if h.get("ui") else None
        except Exception:
            return None


def connect(base_url: str, session: Optional[str] = None,
            workspace: Optional[str] = None) -> RemoteHarness:
    """Build a RemoteHarness; workspace sets a ws:<name> session."""
    if session is None:
        session = f"ws:{workspace}" if workspace else "ws:cli"
    return RemoteHarness(base_url, session=session)
