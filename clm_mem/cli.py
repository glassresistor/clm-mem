"""Command-line interface: clm-mem chat / demo / serve."""

from __future__ import annotations
import argparse
import sys
import textwrap
from typing import Dict, List, Optional
from .blocks import LONG, SHORT
from .config import DEFAULT_CTX_BUDGET, HarnessConfig
from .harness import MemoryHarness
from .llm import OpenAIChat, StubChat, SystemOneClient

HELP = (
    "commands:  /status   /long   /short   /transient   /recall <text>   "
    "/reinject <block-id>   /keep <pct>   /decay [on|off]   "
    "/supersede [on|off]   /quit\n"
    "       agent: /tools [on|off <what>]   /tools <policy>   "
    "/web <query>   /run <command>   /py <code>   /files [path]   "
    "/cat <path>   /rm <path>   /sandbox [info|reset]   "
    "/activity [n]   /tool <seq>   /approvals   /approve <id> | /deny <id>"
)


def _tools_view(h) -> Dict:
    """Tool state for either a local MemoryHarness or an attached remote one."""
    if hasattr(h, "tools") and callable(h.tools):  # RemoteHarness
        return h.tools()
    return h.tools.state()  # type: ignore[union-attr]


def _tools_toggle(h, target: str, on: Optional[bool]):
    if hasattr(h, "tools") and callable(h.tools):
        return h.set_tools(target=target, on=on)
    return h.tools.toggle(target, on)  # type: ignore[union-attr]


def _activity(h, limit: int = 12, full: bool = False):
    if hasattr(h, "tools") and callable(h.tools):
        return h.activity(limit=limit, full=full)
    return h.tools.activity(limit=limit, full=full)  # type: ignore[union-attr]


def _remote(h) -> bool:
    """True for an attached RemoteHarness (its .tools is a method, not a mgr)."""
    return hasattr(h, "tools") and callable(h.tools)


def _tool_run(h, name: str, args: Dict) -> Dict:
    """Invoke a tool directly; slash commands are human-initiated."""
    if _remote(h):
        return h.tool_call(name, args)
    from .tools import ToolCall

    res = h.tools.execute(
        ToolCall(id="cli", name=name, args=args, origin="user")
    )  # type: ignore
    out = {
        "ok": res.ok,
        "summary": res.summary,
        "detail": res.detail,
        "error": res.error,
    }
    out.update(res.data)
    return out


def _show_tool(out: Dict, detail_lines: int = 14) -> None:
    mark = "\u2705" if out.get("ok") else "\u274c"
    print(f"  {mark} {out.get('summary', '')}")
    detail = (out.get("detail") or "").strip()
    if not detail:
        return
    lines = detail.splitlines()
    shown = lines[:detail_lines]
    for ln in shown:
        print(f"     {ln}")
    if len(lines) > detail_lines:
        print(f"     ...[{len(lines) - detail_lines} more lines]")


def _print_activity(entries, seqs=None) -> None:
    icons = {
        "web_search": "\U0001f310",
        "web_fetch": "\U0001f517",
        "run_command": "\U0001f4bb",
        "run_python": "\U0001f40d",
        "write_file": "\u270f",
        "read_file": "\U0001f4c4",
        "list_files": "\U0001f4c1",
    }
    for a in entries:
        mark = icons.get(a.get("tool"), "*")
        status = a.get("status", "?")
        tail = "" if status == "ok" else f" [{status}]"
        print(f"   {mark} {a.get('summary') or render_tool(a)}{tail}")


def render_tool(entry: Dict) -> str:
    import json as _json

    args = entry.get("args") or {}
    parts = []
    for k, v in args.items():
        s = v if isinstance(v, str) else _json.dumps(v)
        parts.append(f"{k}={s[:60]}{'...' if len(s) > 60 else ''}")
    return f"{entry.get('tool')}({', '.join(parts)})"


def handle_agent_command(h, text: str, seen_seq: int = 0) -> int:
    """Slash commands for the agent surface (local or attached remote)."""
    cmd = text.split(maxsplit=1)[0]
    rest = text[len(cmd) :].strip()

    if cmd == "/tools":
        words = rest.split()
        if not words:
            view = _tools_view(h)
            cats = view.get("enabled_categories", {})
            sandbox = view.get("sandbox", {}) or {}
            print(
                f"  tools {'ON' if view.get('active') else 'off'} "
                f"| policy={view.get('approval_mode')} "
                f"| search={view.get('search_provider')} "
                f"| sandbox={sandbox.get('isolation')}"
            )
            hints = {
                "web": "web_search, web_fetch",
                "exec": "run_command, run_python",
                "fs": "write_file, read_file, list_files",
            }
            for c, hint in hints.items():
                icon = {"on": "\u2705", "partial": "\u2b06", "off": "·"}.get(
                    cats.get(c, "off"), "?"
                )
                print(f"    {icon} {c:5} {hint}")
            enabled = view.get("enabled_tools", [])
            if enabled:
                print("    live: " + ", ".join(enabled))
            print(
                "    /tools on|off [web|exec|fs|all|<tool>]   " "/tools auto|ask|deny"
            )
            return seen_seq
        if words[0].lower() in ("auto", "ask", "deny"):
            mode = words[0].lower()
            if _remote(h):
                h.set_tools(approval_mode=mode)
            else:
                h.tools.set_policy(mode)  # type: ignore
            print(f"  approval policy -> {mode}")
            return seen_seq
        verb, target = (words[0].lower(), words[1] if len(words) > 1 else "all")
        if verb not in ("on", "off"):
            verb, target = "toggle", words[0].lower()
        on = None if verb == "toggle" else (verb == "on")
        try:
            view = _tools_toggle(h, target, on)
        except ValueError as exc:
            print(f"  {exc}")
            return seen_seq
        cats = view.get("enabled_categories", {})
        print(
            f"  {target} -> {cats.get(target, 'set')} "
            f"| master={'on' if view.get('tools_enabled') else 'off'}"
        )
        return seen_seq

    if cmd in ("/web", "/fetch", "/run", "/py", "/cat", "/rm"):
        mapping = {
            "/web": ("web_search", lambda r: {"query": r, "max_results": 5}),
            "/fetch": ("web_fetch", lambda r: {"url": r}),
            "/run": ("run_command", lambda r: {"command": r}),
            "/py": ("run_python", lambda r: {"code": r}),
            "/cat": ("read_file", lambda r: {"path": r}),
        }
        if cmd == "/rm":
            if _remote(h):
                out = h.sandbox_delete(rest)
            else:
                try:
                    out = {
                        "ok": True,
                        "summary": f"deleted {rest}",
                        **h.tools.sandbox.delete(rest),
                    }
                except (OSError, ValueError) as exc:
                    print(f"  \u274c {exc}")
                    return seen_seq
            _show_tool(out)
            return seen_seq
        name, mk = mapping[cmd]
        if not rest:
            arg = {
                "/web": "<query>",
                "/fetch": "<url>",
                "/run": "<command>",
                "/py": "<code>",
                "/cat": "<path>",
            }[cmd]
            print(f"  usage: {cmd} {arg}")
            return seen_seq
        out = _tool_run(h, name, mk(rest))
        # search results get their own layout; everything else shows output
        _show_tool(out, detail_lines=0 if cmd in ("/web", "/fetch") else 14)
        for hit in (out.get("hits") or [])[:6]:
            print(f"     • {hit.get('title', '')}")
            print(f"       {hit.get('url', '')}")
            sn = (hit.get("snippet") or "").strip()
            if sn:
                print(f"       {textwrap.shorten(sn, 200)}")
        return seen_seq

    if cmd == "/files":
        sub = rest or "."
        if _remote(h):
            d = h.sandbox_files(sub)
            files = d.get("files", [])
        else:
            files = h.tools.sandbox.list_files(sub)  # type: ignore
        if not files:
            print(f"  (no files under {sub})")
        for f in files[:40]:
            print(f"    {f['bytes']:>8}  {f['path']}")
        return seen_seq

    if cmd == "/sandbox":
        if rest.lower() == "reset":
            out = h.sandbox_reset() if _remote(h) else h.tools.sandbox.reset()
            print(f"  sandbox reset: {out.get('root', '')}")
        else:
            view = _tools_view(h).get("sandbox", {})
            print(f"  root       : {view.get('root')}")
            print(f"  isolation  : {view.get('isolation')}")
            print(
                f"  files      : {view.get('files')} "
                f"({view.get('bytes')} bytes, quota "
                f"{view.get('quota_mb')} MB)"
            )
        return seen_seq

    if cmd in ("/activity", "/tool"):
        if cmd == "/tool":
            try:
                want = int(rest)
            except ValueError:
                print("  usage: /tool <seq>")
                return seen_seq
            entries = [a for a in _activity(h, 60, full=True) if a.get("seq") == want]
        else:
            try:
                n = int(rest) if rest else 12
            except ValueError:
                n = 12
            entries = _activity(h, n)
        if not entries:
            print("  (no tool activity yet)")
        for a in entries:
            print(
                f"    #{a.get('seq')} {a.get('tool')} "
                f"[{a.get('status')}] {a.get('summary', '')}"
            )
            det = (a.get("detail") or "").strip()
            if det and cmd == "/tool":
                for ln in det.splitlines()[:20]:
                    print(f"        {ln}")
        return seen_seq

    # approvals -----------------------------------------------------------
    if cmd == "/approvals":
        data = h.approvals() if _remote(h) else h.tools.approvals()
        pend = data.get("pending", [])
        if not pend:
            print("  (nothing pending)")
        for p in pend:
            print(f"    #{p['id']} {p['tool']}({p['args']}) " f"waiting {p['age_s']}s")
        return seen_seq
    if cmd in ("/approve", "/deny"):
        ok = cmd == "/approve"
        ids = rest.split() or []
        if not ids:
            print(f"  usage: {cmd} <approval-id>")
            return seen_seq
        for aid in ids:
            out = (
                h.approve(aid, ok)
                if _remote(h)
                else h.tools.broker.decide(aid, ok, "cli")
            )
            print(f"    {aid}: {out}")
        return seen_seq
    return seen_seq


def _install_prompt_responder(h) -> None:
    """Local (non-attached) sessions answer approval prompts on stdin."""
    if hasattr(h, "tools") and callable(h.tools):
        return  # remote resolves over REST
    mgr = h.tools  # type: ignore[union-attr]

    def ask(rec):
        print(
            f"\n  \u26a0 {rec.tool}({render_tool({'tool': rec.tool, 'args': rec.args})}) "
            f"wants to run"
        )
        answer = input("  allow? [y/n/a] ").strip().lower()
        if answer == "a":
            mgr.broker.mode = "auto"
            print("   policy -> auto for the rest of this session")
            return True
        return answer in ("y", "yes")

    mgr.broker.responder = ask


def chat_loop(h: MemoryHarness) -> None:
    print(HELP, flush=True)
    _install_prompt_responder(h)
    try:  # don't replay history from earlier runs
        seen_seq = max((a.get("seq", 0) for a in _activity(h, 60)), default=0)
    except Exception:
        seen_seq = 0
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/status":
            print(h.status())
            continue
        if text.split(maxsplit=1)[0] in ("/decay", "/supersede"):
            attr = {"/decay": "decay_enabled", "/supersede": "supersede_enabled"}[
                text.split()[0]
            ]
            parts = text.split()
            if len(parts) == 1:  # bare command: toggle
                setattr(h.cfg, attr, not getattr(h.cfg, attr))
            elif parts[1].lower() in ("on", "true", "1", "yes"):
                setattr(h.cfg, attr, True)
            elif parts[1].lower() in ("off", "false", "0", "no"):
                setattr(h.cfg, attr, False)
            else:
                print("  usage: /decay [on|off] or /supersede [on|off]")
                continue
            print(f"  {attr} = {getattr(h.cfg, attr)}")
            continue
        if text.startswith("/keep"):
            try:
                h.cfg.shortterm_keep_pct = float(text.split()[1])
            except (IndexError, ValueError):
                print(
                    f"  current shortterm_keep_pct = "
                    f"{h.cfg.shortterm_keep_pct:.0%}  "
                    f"(usage: /keep 0.75)"
                )
                continue
            print(f"  shortterm_keep_pct = {h.cfg.shortterm_keep_pct:.0%}")
            continue
        if text in ("/long", "/short", "/transient") or text.startswith(
            ("/long ", "/short ", "/transient ")
        ):
            from .blocks import TRANSIENT

            want = {"/long": LONG, "/short": SHORT, "/transient": TRANSIENT}[
                text.split()[0]
            ]
            hits = h.live_blocks(want)
            for b in hits:
                print(f"  #{b.id} t{b.turn}: " f"{textwrap.shorten(b.content, 120)}")
            if not hits:
                print(f"  (no live {want} blocks)")
            continue
        if text.startswith("/recall "):
            hits = h.recall(text[len("/recall ") :])
            for rec in hits:
                print(
                    f"  archive #{rec['id']} ({rec['tag']}, turn "
                    f"{rec['turn']}): {textwrap.shorten(rec['content'], 140)}"
                )
            if not hits:
                print("  (no archive hits)")
            continue
        if text.startswith("/reinject "):
            try:
                bid = int(text.split()[1])
            except (IndexError, ValueError):
                print("  usage: /reinject <block-id>")
                continue
            print(
                "  re-injected as shortterm"
                if h.reinject(bid)
                else "  id not in archive"
            )
            continue
        # ------------------------------------------------ agent slash commands
        head = text.split(maxsplit=1)[0]
        if head in (
            "/tools",
            "/web",
            "/fetch",
            "/run",
            "/py",
            "/files",
            "/cat",
            "/rm",
            "/sandbox",
            "/activity",
            "/tool",
            "/approvals",
            "/approve",
            "/deny",
        ):
            try:
                seen_seq = handle_agent_command(h, text, seen_seq)
            except Exception as exc:
                print(f"  error: {exc}")
            continue

        print("  ...working", flush=True)
        try:
            reply = h.step(text)
        except Exception as e:  # a backend hiccup must not kill the session
            print(f"ai> ERROR backend: {e}")
            continue
        entries = [a for a in _activity(h, 12) if a.get("seq", 0) > seen_seq]
        if entries:
            print("  \u26a1 tools used this turn:", flush=True)
            _print_activity(entries[:8])
            print("    (/tool <seq> expands one; /activity lists more)")
            seen_seq = max(a.get("seq", 0) for a in entries)
        print(f"ai> {reply}")
        print(f"    ({h.status()})", flush=True)


def _add_harness_flags(sub: argparse.ArgumentParser) -> None:
    """Harness knobs shared by chat / demo / serve (no endpoint args)."""
    sub.add_argument("--temperature", type=float, default=0.6)
    sub.add_argument(
        "--max-tokens", type=int, default=1024, help="generation cap per reply"
    )
    sub.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="live-window token budget for the harness; when "
        "unset it follows the backend's own context window "
        "(read from /props) so llama.cpp and clm-mem never "
        "drift apart",
    )
    sub.add_argument(
        "--shortterm-keep-pct",
        type=float,
        default=0.5,
        help="fraction of shortterm KEPT per FIFO eviction round "
        "(changeable mid-session with /keep <pct>)",
    )
    sub.add_argument("--keep-recent-turns", type=int, default=2)
    sub.add_argument(
        "--classify-mode", choices=["inline", "post", "both"], default="both"
    )
    sub.add_argument(
        "--tokenizer",
        default="",
        help="HF tokenizer id for exact token counts (optional)",
    )
    sub.add_argument(
        "--workdir",
        default=".",
        help="where context.md / context.jsonl / archive.jsonl live",
    )
    sub.add_argument(
        "--evict-threshold",
        type=float,
        default=None,
        help="SystemOne noul confidence needed to approve eviction "
        "(higher = more conservative, keep more memory). "
        "Default 0.7, or 0.55 under --use-mock so the "
        "bundled mock decisions visibly evict",
    )
    # --- forgetting strategies (toggles; default off) -----------------------
    sub.add_argument(
        "--decay",
        dest="decay",
        action="store_true",
        default=False,
        help="enable decay & reheating: long-term facts lose value "
        "on a forgetting curve unless accessed, demote to "
        "short-term below the floor, and phase-3 offload "
        "ranks by value instead of age",
    )
    sub.add_argument(
        "--no-decay",
        dest="decay",
        action="store_false",
        help="disable decay & reheating (default)",
    )
    sub.add_argument(
        "--decay-half-life",
        type=int,
        default=8,
        help="turns for importance to halve without access",
    )
    sub.add_argument(
        "--demote-threshold",
        type=float,
        default=0.45,
        help="effective importance below which a long-term block "
        "becomes short-term (working state again)",
    )
    sub.add_argument(
        "--supersede",
        dest="supersede",
        action="store_true",
        default=False,
        help="enable supersession: a new long-term fact that "
        "shares a subject slot (codename/port/id/...) with "
        "an older one supersedes it; exact duplicates are "
        "dropped at write time instead of admitted twice",
    )
    sub.add_argument(
        "--no-supersede",
        dest="supersede",
        action="store_false",
        help="disable supersession (default)",
    )
    # --- agent tools (opt-in) ------------------------------------------------
    sub.add_argument(
        "--tools",
        action="store_true",
        default=False,
        help="let the model call tools (web search, sandboxed "
        "commands, file work); results enter context as "
        "short-term memory. Toggle mid-session with /tools",
    )
    sub.add_argument(
        "--sandbox-mode",
        choices=["off", "jail", "docker"],
        default="jail",
        help="isolation level for run_command/run_python: 'jail' "
        "is a bounded local process (cwd, scrubbed env, "
        "time/rlimits, output caps); 'docker' runs each "
        "command in a container (--network none) when the "
        "docker CLI is available; 'off' removes exec/file "
        "tools entirely",
    )
    sub.add_argument(
        "--sandbox-image",
        default="python:3.12-slim",
        help="container image for --sandbox-mode docker",
    )
    sub.add_argument(
        "--sandbox-disk-mb",
        type=int,
        default=256,
        help="soft disk quota per workspace sandbox (MB)",
    )
    sub.add_argument(
        "--sandbox-mem-mb",
        type=int,
        default=512,
        help="address-space / container memory cap (MB)",
    )
    sub.add_argument(
        "--ask",
        action="store_true",
        help="require approval before mutating tools run "
        "(CLI prompts inline; web studio/REST approve via "
        "cards). Equivalent to --approval-mode ask",
    )
    sub.add_argument(
        "--approval-mode",
        choices=["auto", "ask", "deny"],
        default="",
        help="auto (default) | ask | deny for mutating tools",
    )
    sub.add_argument(
        "--max-tool-calls",
        type=int,
        default=8,
        help="agent-loop iterations allowed per turn",
    )
    sub.add_argument(
        "--tool-timeout",
        type=float,
        default=30.0,
        help="wall-clock cap per sandbox command / fetch (s)",
    )
    sub.add_argument(
        "--search-provider",
        choices=["ddg", "brave", "tavily", "exa", "stub", "off"],
        default="",
        help="web search backend; 'stub' is deterministic offline "
        "(default under --use-mock), ddg needs no API key",
    )
    sub.add_argument(
        "--search-results", type=int, default=5, help="results per web_search call"
    )
    sub.add_argument(
        "--exec-output-limit",
        type=int,
        default=8000,
        help="stdout/stderr cap per command (chars)",
    )


def _add_common(sub: argparse.ArgumentParser) -> None:
    """Backend endpoint args + harness flags (chat / demo)."""
    sub.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible endpoint (vLLM/Ollama/...)",
    )
    sub.add_argument("--model", default="Qwen/Qwen3-8B-Instruct")
    sub.add_argument(
        "--api-key",
        "--backend-api-key",
        help="bearer token for the chat backend's API (any "
        "OpenAI-compatible server; local llama.cpp/vLLM/"
        "Ollama do not check it, hosted services like "
        "Featherless or OpenRouter require one)",
    )
    _add_harness_flags(sub)
    sub.add_argument(
        "--systemone-url",
        default="http://127.0.0.1:8010",
        help="bare root URL of the System One Lite decision server "
        "(scheme://host[:port] only -- do not append endpoint paths like "
        "/v1/decisions or /v1/systemone; the client sets the full path to "
        "the decision endpoint itself). Used for typed decisions "
        "(pre-classification + eviction backstop). "
        "Set to 'none' to disable.",
    )
    sub.add_argument(
        "--systemone-model",
        default="",
        help="model id sent to the decision API (required; both "
        "the llama.cpp fork and Featherless reject bodies "
        "without it). e.g. featherless-ai/"
        "Qwen3.6-35B-A3B-classifier",
    )
    sub.add_argument(
        "--systemone-api-key",
        default="",
        help="bearer token for the decision API (defaults to " "$FEATHERLESS_API_KEY)",
    )
    sub.add_argument(
        "--connect",
        default="",
        help="attach to a running `clm-mem serve` at this base "
        "URL (e.g. http://127.0.0.1:8080/v1) instead of "
        "talking to a backend directly",
    )
    sub.add_argument(
        "--workspace",
        default="",
        help="workspace/project for --connect sessions " "(session becomes ws:<name>)",
    )
    sub.add_argument(
        "--session",
        default="",
        help="explicit session id for --connect " "(overrides ws:<workspace> naming)",
    )


def _cfg(a: argparse.Namespace, fallback_model: Optional[str] = None) -> HarnessConfig:
    fb = fallback_model if fallback_model is not None else getattr(a, "model", "")
    thr = (
        a.evict_threshold
        if a.evict_threshold is not None
        else (0.55 if getattr(a, "use_mock", False) else 0.7)
    )
    tools_on = bool(getattr(a, "tools", False))
    provider = getattr(a, "search_provider", "") or (
        "stub" if getattr(a, "use_mock", False) else "ddg"
    )
    return HarnessConfig(
        max_context_tokens=a.max_context_tokens or DEFAULT_CTX_BUDGET,
        shortterm_keep_pct=a.shortterm_keep_pct,
        keep_recent_turns=a.keep_recent_turns,
        classify_mode=a.classify_mode,
        workdir=a.workdir,
        tokenizer_name=a.tokenizer or fb,
        systemone_evict_threshold=thr,
        decay_enabled=a.decay,
        decay_half_life=a.decay_half_life,
        demote_threshold=a.demote_threshold,
        supersede_enabled=a.supersede,
        tools_enabled=tools_on,
        sandbox_mode=getattr(a, "sandbox_mode", "jail"),
        sandbox_image=getattr(a, "sandbox_image", "python:3.12-slim"),
        sandbox_disk_mb=getattr(a, "sandbox_disk_mb", 256),
        sandbox_mem_mb=getattr(a, "sandbox_mem_mb", 512),
        approval_mode=(
            "ask"
            if getattr(a, "ask", False)
            else getattr(a, "approval_mode", "auto") or "auto"
        ),
        max_tool_calls=getattr(a, "max_tool_calls", 8),
        tool_timeout=getattr(a, "tool_timeout", 30.0),
        search_provider=provider,
        max_search_results=getattr(a, "search_results", 5),
        exec_output_limit=getattr(a, "exec_output_limit", 8000),
    )


def cmd_chat(a: argparse.Namespace) -> None:
    if getattr(a, "connect", ""):
        # Attach this prompt to a running `clm-mem serve` (same library the
        # web studio uses; sessions are shared via X-Cllm-Mem-Session).
        from .remote import connect as open_remote

        h = open_remote(
            a.connect,
            session=getattr(a, "session", "") or None,
            workspace=getattr(a, "workspace", "") or None,
        )
        try:
            status = h.status()
        except Exception as exc:
            sys.exit(f"error: cannot attach to {a.connect}: {exc}")
        ui = h.ui_url()
        h.prime()
        print(f"clm-mem | attached to serve @ {a.connect} " f"(session {h.session})")
        if ui:
            print(f"web studio: {ui}")
        print(status)
        try:
            view = h.tools()
            print(
                f"tools   : {'ON' if view.get('active') else 'off'} "
                f"| policy={view.get('approval_mode')} "
                f"| sandbox={view.get('sandbox', {}).get('isolation')} "
                f"(/tools on, /web ..., /run ...)"
            )
            if view.get("approval_mode") == "ask":
                print(
                    "          mutating tool calls pause for approval: "
                    "resolve them in the web studio cards or from a second "
                    "terminal with /approvals + /approve <id>"
                )
        except Exception:
            pass  # older server without the agent surface
        chat_loop(h)
        return

    if a.max_context_tokens is None:
        # follow the served window: llama.cpp fixes its context at load time,
        # so mirror it instead of letting the two budgets drift.
        from .server import discover_backend_ctx, normalize_root

        ctx = discover_backend_ctx(normalize_root(a.base_url), api_key=a.api_key)
        a.max_context_tokens = ctx or DEFAULT_CTX_BUDGET
        print(
            f"context window: {a.max_context_tokens} tokens " f"(from backend)"
            if ctx
            else f"context window: {a.max_context_tokens} tokens (default)"
        )

    llm = OpenAIChat(
        base_url=a.base_url,
        model=a.model,
        api_key=a.api_key,
        temperature=a.temperature,
        max_tokens=a.max_tokens,
    )

    # Wire up SystemOne for typed decisions if configured. The probe goes
    # through the same client the harness will use later: --systemone-url is
    # the bare server root and the client sets the full path to the decision
    # endpoint itself (every provider answers the same request/response
    # contract there). A raw ping with no model always fails against both
    # the fork and Featherless.
    systemone = None
    if a.systemone_url and a.systemone_url.lower() != "none":
        try:
            systemone = SystemOneClient(
                a.systemone_url,
                model=a.systemone_model,
                api_key=getattr(a, "systemone_api_key", ""),
            )
        except ValueError as e:
            parser.error(str(e))
        try:
            probe = systemone.system_one(
                "test", {"q": {"type": "noul", "instructions": "is this a test?"}}
            )
            if "answers" not in probe:
                raise RuntimeError(f"unexpected reply: {probe}")
            print(
                f"SystemOne decision API: {systemone.base_url} (OK, "
                f"model={systemone.model})"
            )
        except Exception as e:
            print(f"WARNING: SystemOne API unreachable at {a.systemone_url}: {e}")
            print("         Pre-classification and eviction backstop will be skipped.")
            systemone = None

    h = MemoryHarness(llm, _cfg(a), systemone=systemone)
    print(f"clm-mem | {h.status()}\nbackend: {a.base_url} -> {a.model}")
    if h.tools.active:
        print(
            f"tools   : ON | policy={h.cfg.approval_mode} "
            f"| sandbox={h.tools.sandbox.isolation_label()} "
            f"| search={h.cfg.search_provider}"
        )
    chat_loop(h)


def cmd_demo(a: argparse.Namespace) -> None:
    """Offline simulation with StubChat -- exercises all three eviction phases."""
    a.use_mock = True  # mock-aware default evict threshold
    cfg = _cfg(a)
    cfg.max_context_tokens = min(cfg.max_context_tokens, 2048)

    # Use mock SystemOne to exercise typed decision code paths offline
    from .llm import MockSystemOne

    h = MemoryHarness(StubChat(), cfg, systemone=MockSystemOne())
    print("demo mode: using mock SystemOne (no HTTP calls)")
    filler = (
        "Here is a longish status update with plenty of incidental detail "
        "that eats tokens while we work. " * 3
    )
    msgs = [
        (
            f"Remember: project codename Nova-{i}, port 80{i}. {filler}"
            if i % 3 == 0
            else f"Quick temp note {i}: {filler}"
        )
        for i in range(24)
    ]
    for m in msgs:
        h.step(m)
        print(h.status())
    print("\n-- final live context --")
    for b in h.blocks:
        print(
            f"  #{b.id:<3} {b.tag:<9} t{b.turn:<2} "
            f"{textwrap.shorten(b.content, 90)}"
        )
    hits = h.recall("Nova")
    sample = textwrap.shorten(hits[-1]["content"], 80) if hits else ""
    print(f"\n/recall 'Nova' -> {len(hits)} archived hit(s); e.g. {sample}")


def _add_serve_flags(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--host",
        dest="public_host",
        default="127.0.0.1",
        help="address the public llama.cpp-compatible endpoint binds",
    )
    sub.add_argument("--port", dest="public_port", type=int, default=8080)
    sub.add_argument(
        "--served-model", default="clm-mem-model", help="model id advertised to clients"
    )
    sub.add_argument(
        "--use-mock",
        action="store_true",
        help="run the bundled mock backend + mock SystemOne "
        "(no GPU, no binaries needed)",
    )
    sub.add_argument(
        "--llama-server",
        dest="llama_server_bin",
        default="",
        help="llama-server binary to spawn behind the proxy "
        "(simple form: just the path; see --backend-cmd for "
        "passing the full command with options)",
    )
    sub.add_argument(
        "--model-file",
        dest="model_path",
        default="",
        help="GGUF/weights path passed to llama-server --model",
    )
    sub.add_argument(
        "--ctx-size",
        type=int,
        default=0,
        help="context size passed to llama-server (optional)",
    )
    sub.add_argument(
        "--backend-cmd",
        default="",
        help="complete backend command line including all options, "
        'e.g. "llama-server -m qwen.gguf --n-gpu-layers 99 '
        '--ctx-size 16384". Use {host} / {port} placeholders '
        "if the server needs them spelled your way; otherwise "
        "--host/--port are appended. Replaces "
        "--llama-server/--model-file/--ctx-size when given.",
    )
    sub.add_argument(
        "--backend-url",
        default="",
        help="root (or /v1) URL of an already-running server "
        "to proxy instead of spawning one",
    )
    sub.add_argument(
        "--backend-api-key",
        default="",
        help="bearer token sent to the chat backend on every LLM "
        "call it fronts (needed for hosted OpenAI-compatible "
        "backends, e.g. --backend-url https://api.featherless.ai "
        "--backend-api-key KEY). Local servers ignore it; "
        "when unset the proxy relays any Authorization "
        "header clients send",
    )
    sub.add_argument(
        "--systemone-binary",
        dest="systemone_bin",
        default="",
        help="SystemOne fork/server binary to spawn behind the proxy",
    )
    sub.add_argument(
        "--systemone-cmd",
        default="",
        help="complete SystemOne server command line including all "
        "options (placeholders {host}/{port} supported); "
        "replaces --systemone-binary when given.",
    )
    sub.add_argument(
        "--systemone-url",
        default="",
        help="bare root URL of an already-running System One decision "
        "server, scheme://host[:port] only (e.g. http://localhost:8010 or "
        "https://api.featherless.ai). Do not append endpoint paths such as "
        "/v1/decisions, /v1/systemone or /v1/classifier -- the client sets "
        "the full path to the decision endpoint itself",
    )
    sub.add_argument(
        "--systemone-model",
        default="",
        help="model id sent to the decision API; required by both "
        "the llama.cpp fork and Featherless (default: a bare "
        "name the fork accepts, set the real id for hosted)",
    )
    sub.add_argument(
        "--systemone-api-key",
        default="",
        help="bearer token for the decision API (defaults to " "$FEATHERLESS_API_KEY)",
    )
    sub.add_argument(
        "--systemone-passthrough",
        action="store_true",
        help="the backend itself answers the decision route " "(SystemOne fork build)",
    )
    sub.add_argument(
        "--log-dir", default="", help="where backend process logs go (default: tmpdir)"
    )
    sub.add_argument(
        "--json", action="store_true", help="print only the machine-readable READY line"
    )
    sub.add_argument(
        "--no-ui",
        dest="ui",
        action="store_false",
        help="do not host the bundled web studio at the root URL",
    )
    _add_harness_flags(sub)


def cmd_serve(a: argparse.Namespace) -> None:
    """One llama.cpp-shaped endpoint in front, 2-3 processes behind."""
    import dataclasses as _dc
    import json as _json
    import time as _time
    from .server import start_supervisor

    harness_kwargs = _dc.asdict(_cfg(a, ""))
    if a.max_context_tokens is None:
        # not pinned on the CLI -> supervisor sets it from the backend's ctx
        harness_kwargs["max_context_tokens"] = None

    if a.use_mock:
        backend_mode = "mock"
    elif a.backend_url:
        backend_mode = "external"
    elif a.llama_server_bin or getattr(a, "backend_cmd", ""):
        backend_mode = "spawn"
    else:
        sys.exit(
            "error: pick a backend: --use-mock, --backend-url URL, "
            "--llama-server PATH or --backend-cmd 'FULL COMMAND'"
        )

    backend_cmd = None
    if getattr(a, "backend_cmd", ""):
        # full command line including options, passed through verbatim
        import shlex

        backend_mode = "spawn"
        backend_cmd = shlex.split(a.backend_cmd)
    elif backend_mode == "spawn":
        backend_cmd = [a.llama_server_bin]
        if a.model_path:
            backend_cmd += ["--model", a.model_path]
        if a.ctx_size:
            backend_cmd += ["--ctx-size", str(a.ctx_size)]

    sys_cmd = None
    if a.systemone_passthrough:
        sys_mode = "passthrough"
    elif getattr(a, "systemone_cmd", ""):
        import shlex

        sys_mode, sys_cmd = "spawn", shlex.split(a.systemone_cmd)
    elif a.use_mock:
        sys_mode = "mock"
    elif a.systemone_bin:
        sys_mode, sys_cmd = "spawn", [a.systemone_bin]
    elif a.systemone_url:
        sys_mode = "external"
    else:
        sys_mode = "none"

    sup = start_supervisor(
        public_host=a.public_host,
        public_port=a.public_port,
        served_model=a.served_model,
        backend_mode=backend_mode,
        backend_cmd=backend_cmd,
        backend_url=a.backend_url or None,
        backend_api_key=a.backend_api_key,
        systemone_mode=sys_mode,
        systemone_cmd=sys_cmd if sys_mode == "spawn" else None,
        systemone_url=a.systemone_url or None,
        systemone_model=getattr(a, "systemone_model", ""),
        systemone_api_key=getattr(a, "systemone_api_key", ""),
        harness_kwargs=harness_kwargs,
        chat_temperature=a.temperature,
        chat_max_tokens=a.max_tokens,
        log_dir=a.log_dir or None,
        ui=a.ui,
    )

    procs = {h.name: f"{h.host}:{h.port}" for h in sup.manager.handles if h.alive()}
    hk = dict(sup.config.harness_kwargs or {})
    ready = {
        "ready": True,
        "base_url": sup.public_base,
        "root_url": sup.base_url,
        "model": a.served_model,
        "backend": sup.config.backend_root,
        "systemone": sup.config.systemone_root,
        "backend_ctx": sup.config.backend_ctx,
        "budget": hk.get("max_context_tokens"),
        "tools_enabled": bool(hk.get("tools_enabled")),
        "sandbox_mode": hk.get("sandbox_mode"),
        "approval_mode": hk.get("approval_mode"),
        "search_provider": hk.get("search_provider"),
        "ui_url": sup.ui_url,
        "mcp_url": f"{sup.base_url}/mcp" if sup.ui_url else None,
        "processes": procs,
    }
    if not a.json:
        print(f"clm-mem serve | llama.cpp-compatible drop-in at {sup.public_base}")
        print(f"  advertised model : {a.served_model}")
        print(f"  chat backend     : {sup.config.backend_root or '(none)'}")
        print(
            f"  context budget   : "
            f"{sup.config.harness_kwargs.get('max_context_tokens')} tokens "
            f"(llama.cpp window: {sup.config.backend_ctx or 'unknown'})"
        )
        print(f"  systemone api    : {sup.config.systemone_root or '(none)'}")
        if sup.ui_url:
            print(f"  web studio       : {sup.ui_url}  (MCP tools: {sup.base_url}/mcp)")
        if hk.get("tools_enabled"):
            print(
                f"  agent tools      : web/exec/fs ON "
                f"(approval={hk.get('approval_mode')}, "
                f"sandbox={hk.get('sandbox_mode')}, "
                f"search={hk.get('search_provider')})"
            )
        else:
            print(
                "  agent tools      : off — pass --tools to enable web "
                "search, sandboxed commands and file work"
            )
        print(
            "  memory sessions  : send header "
            f"X-Cllm-Mem-Session: <id> on chat requests"
        )
    print(_json.dumps(ready), flush=True)
    try:
        while True:
            _time.sleep(3600)
    except KeyboardInterrupt:  # pragma: no cover
        print("\nshutting down...", flush=True)
    finally:
        sup.shutdown()


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        prog="clm-mem",
        description="Zero-training long/short-term memory harness for open "
        "chat models (inspired by Context Language Models, "
        "arXiv:2609.37725)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("chat", help="interactive chat with memory management")
    sd = sub.add_parser("demo", help="offline eviction simulation (no server needed)")
    ss = sub.add_parser(
        "serve",
        help="llama.cpp-compatible drop-in server "
        "(proxy + backend + optional SystemOne)",
    )
    for s in (sc, sd):
        _add_common(s)
    _add_serve_flags(ss)
    a = p.parse_args(argv)
    try:
        if a.cmd == "chat":
            cmd_chat(a)
        elif a.cmd == "serve":
            cmd_serve(a)
        else:
            cmd_demo(a)
    except ValueError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
