"""Tests for the agent layer: sandbox jail, web providers, approvals, the
harness tool loop and the serve REST surface for tools/sandbox/activity.

Everything runs offline: search uses the deterministic ``stub`` provider and
the bundled mock backends answer tool_calls from StubChat's planner.
"""
import json
import os
import urllib.error
import urllib.request
import pytest
from clm_mem import HarnessConfig, MemoryHarness, StubChat, start_supervisor
from clm_mem.tools import ToolCall, Sandbox


# ------------------------------------------------------------------ sandbox

@pytest.fixture()
def sb(tmp_path):
    return Sandbox(root=str(tmp_path / "sandbox"), mode="jail", timeout=10.0,
                   output_limit=256, disk_mb=8, mem_mb=128)


def test_sandbox_exec_and_files(sb):
    res = sb.exec("printf 'hello\n' > hi.txt && cat hi.txt")
    assert res.ok and "hello" in res.stdout
    assert res.isolation == "process-jail"
    files = sb.list_files()
    assert {"hi.txt"} <= {f["path"] for f in files}


def test_sandbox_env_is_scrubbed(sb, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret")
    res = sb.exec('echo "path=$PATH"; echo "secret=${OPENAI_API_KEY:-none}"')
    assert "secret=none" in res.stdout        # host secrets never leak
    assert "path=" in res.stdout and len(res.stdout) > len("path=")


def test_sandbox_timeout_kills_long_command(sb):
    sb.timeout = 1.0
    res = sb.exec("sleep 5", timeout=1)
    assert not res.ok
    assert res.timed_out and res.note


def test_sandbox_output_capped(sb):
    res = sb.exec("seq 1 2000")
    assert "output capped" in res.stdout
    assert len(res.stdout) <= sb.output_limit + 64


def test_sandbox_path_containment(sb):
    with pytest.raises(ValueError):
        sb.write_file("../escape.txt", "x")
    with pytest.raises(ValueError):
        sb.read_file("/etc/hostname")
    sb.write_file("deep/dir/a.py", "print(1)")
    assert sb.read_file("deep/dir/a.py")["text"] == "print(1)"


def test_sandbox_disabled_mode(tmp_path):
    s = Sandbox(root=str(tmp_path / "s"), mode="off")
    assert not s.enabled
    with pytest.raises(RuntimeError):
        s.exec("echo hi")


def test_sandbox_reset_and_usage(sb):
    sb.write_file("x.py", "x = 1")
    n, b = sb.usage()
    assert n == 1 and b > 0
    sb.reset()
    assert sb.usage()[0] == 0


# ------------------------------------------------------------ tool manager

def cfg_of(tmp_path, **kw):
    base = dict(workdir=str(tmp_path), max_context_tokens=600,
                tools_enabled=True, search_provider="stub",
                sandbox_mode="jail", tool_timeout=10.0)
    base.update(kw)
    return HarnessConfig(**base)


@pytest.fixture()
def h(tmp_path):
    return MemoryHarness(StubChat(), cfg_of(tmp_path))


def test_manager_state_and_toggle(h):
    st = h.tools.state()
    assert st["active"] and "web_search" in st["enabled_tools"]
    h.tools.toggle("exec", False)
    assert h.tools.state()["enabled_categories"]["exec"] == "off"
    assert not any(a.startswith("run_") for a in
                   h.tools.state()["enabled_tools"])
    h.tools.toggle("exec", True)
    assert "run_command" in h.tools.state()["enabled_tools"]
    h.tools.toggle("all", False)
    assert h.tools.state()["enabled_tools"] == []
    h.tools.toggle("web", True)         # enabling a tool turns the master on
    assert h.tools.state()["tools_enabled"]


def test_web_search_stub_and_fetch_errors(h):
    res = h.tools.execute(ToolCall(id="1", name="web_search",
                                   args={"query": "qwen", "max_results": 2}))
    assert res.ok and len(res.data["hits"]) == 2
    assert res.summary.endswith("2 hits")
    bad = h.tools.execute(ToolCall(id="2", name="web_fetch",
                                   args={"url": "not-a-scheme-at-all"}))
    assert not bad.ok or bad.data            # fetch failure is reported, no crash


def test_provider_off_blocks_web(tmp_path):
    h = MemoryHarness(StubChat(), cfg_of(tmp_path, search_provider="off"))
    res = h.tools.execute(ToolCall(id="1", name="web_search",
                                   args={"query": "x"}))
    assert not res.ok and "disabled" in res.error


def test_auto_policy_runs_exec(h):
    res = h.tools.execute(ToolCall(id="1", name="run_command",
                                   args={"command": "echo ok"}))
    assert res.ok and "ok" in res.detail


def test_ask_policy_denies_without_responder(tmp_path):
    h = MemoryHarness(StubChat(), cfg_of(tmp_path, approval_mode="ask",
                                         approval_timeout=1.0))
    res = h.tools.execute(ToolCall(id="1", name="run_command",
                                   args={"command": "echo nope"}))
    assert not res.ok and "denied" in (res.error + res.summary).lower()
    assert h.tools.activity()[0]["status"] == "denied"


def test_ask_policy_with_responder_approves(tmp_path):
    seen = []

    def responder(rec):
        seen.append(rec.tool)
        return True
    h = MemoryHarness(StubChat(), cfg_of(tmp_path, approval_mode="ask"),
                      responder=responder)
    res = h.tools.execute(ToolCall(id="1", name="run_command",
                                   args={"command": "echo yes"}))
    assert res.ok and seen == ["run_command"]


def test_deny_policy_blocks_exec_allows_web(tmp_path):
    h = MemoryHarness(StubChat(), cfg_of(tmp_path, approval_mode="deny"))
    assert not h.tools.execute(ToolCall(id="1", name="run_command",
                                        args={"command": "echo x"})).ok
    assert h.tools.execute(ToolCall(id="2", name="web_search",
                                    args={"query": "x"})).ok


# ------------------------------------------------------------ harness loop

def test_step_runs_tool_loop(h):
    reply = h.step('search "quantum bits" please')
    assert "quantum" in reply.lower()
    acts = h.tools.activity(10)
    assert acts[0]["tool"] == "web_search" and acts[0]["status"] == "ok"
    # the transcript is context, not chat: role tool blocks exist, replies clean
    assert any(b.role == "tool" for b in h.blocks)
    assert "[tool" not in reply


def test_tool_transcript_tagged_short_and_bounded(h):
    h.step('run command "echo abc > a.txt && cat a.txt"')
    tool_blocks = [b for b in h.blocks if b.role == "tool"]
    assert tool_blocks and all(b.tag == "shortterm" for b in tool_blocks)
    assert h.tokens() <= h.cfg.max_context_tokens


def test_disabled_tools_plain_chat(tmp_path):
    h = MemoryHarness(StubChat(), cfg_of(tmp_path, tools_enabled=False))
    reply = h.step('search "whatever"')
    assert not h.tools.active
    assert h.tools.activity() == []
    assert reply                                    # stub still answers normally


# ------------------------------------------------------------- serve REST

def http_call(sup, path, method="GET", body=None, headers=None):
    url = f"{sup.base_url}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw
    raw = resp.read().decode()
    try:
        return resp.status, json.loads(raw)
    except Exception:
        return resp.status, raw


@pytest.fixture()
def sup(tmp_path):
    s = start_supervisor(
        public_port=0, served_model="mock-model", backend_mode="mock",
        systemone_mode="mock", timeout=30,
        harness_kwargs={"workdir": str(tmp_path), "max_context_tokens": 800,
                        "tools_enabled": True, "search_provider": "stub",
                        "sandbox_mode": "jail"})
    yield s
    s.shutdown()


HDR = {"X-Cllm-Mem-Session": "ws:agent"}


def test_serve_agent_turn_and_activity(sup):
    st, resp = http_call(sup, "/v1/chat/completions", "POST",
                         {"messages": [{"role": "user",
                                        "content": 'search "unsloth studio"'}]},
                         HDR)
    assert st == 200
    assert "[tool" not in resp["choices"][0]["message"]["content"]
    kinds = [e["kind"] for e in resp["clm_mem"]["events"]]
    assert any(k.startswith("tool") for k in kinds)

    st, d = http_call(sup, "/v1/clm/sessions/ws%3Aagent/activity?limit=5",
                      headers=HDR)
    assert st == 200 and d["activity"][0]["tool"] == "web_search"


def test_serve_tools_state_and_toggle(sup):
    st, d = http_call(sup, "/v1/clm/sessions/ws%3Aagent/tools", headers=HDR)
    assert st == 200 and "web_search" in d["enabled_tools"]
    assert d["sandbox"]["mode"] == "jail"

    st, d = http_call(sup, "/v1/clm/sessions/ws%3Aagent/tools", "POST",
                      {"target": "exec", "on": False}, HDR)
    assert st == 200 and not any(t.startswith("run_")
                                 for t in d["enabled_tools"])
    # exec now off -> a model call to run_command is refused, web still works
    st, resp = http_call(sup, "/v1/chat/completions", "POST",
                         {"messages": [{"role": "user",
                                        "content": 'run command "echo x"'}]},
                         HDR)
    assert st == 200

    st, d = http_call(sup, "/v1/clm/sessions/ws%3Aagent/tools", "POST",
                      {"target": "exec", "on": True}, HDR)
    assert "run_command" in d["enabled_tools"]


def test_serve_sandbox_exec_files_write_read(sup):
    st, d = http_call(sup, "/v1/clm/sessions/ws%3Aagent/sandbox/exec", "POST",
                      {"command": "printf 'x=1\n' > prog.py; cat prog.py"}, HDR)
    assert st == 200 and d["ok"] and "x=1" in d["detail"]

    st, d = http_call(sup, "/v1/clm/sessions/ws%3Aagent/sandbox", headers=HDR)
    assert st == 200 and "prog.py" in [f["path"] for f in d["files"]]

    st, d = http_call(sup, "/v1/clm/sessions/ws%3Aagent/sandbox/file?path=prog.py",
                      headers=HDR)
    assert st == 200 and "x=1" in d["text"]

    st, d = http_call(sup, "/v1/clm/sessions/ws%3Aagent/sandbox/write", "POST",
                      {"path": "notes.md", "content": "hi"}, HDR)
    assert st == 200 and d["bytes"] == 2

    # traversal is rejected through the API too
    st, d = http_call(sup, "/v1/clm/sessions/ws%3Aagent/sandbox/file"
                      "?path=../../etc/passwd", headers=HDR)
    assert st in (400, 404)

    st, d = http_call(sup, "/v1/clm/sessions/ws%3Aagent/sandbox", "DELETE",
                      headers=HDR)
    assert st == 200


def test_serve_approvals_round_trip(sup):
    # switch the session to ask mode with a short timeout, then trigger a call
    http_call(sup, "/v1/clm/sessions/ws%3Aagent/tools", "POST",
              {"approval_mode": "ask"}, HDR)
    st, d = http_call(sup, "/v1/clm/sessions/ws%3Aagent/approvals", headers=HDR)
    assert st == 200 and d["mode"] == "ask"

    # deny path: no responder -> times out denied (approval_timeout default 300
    # is long; instead approve through the API from a background thread)
    import threading, time as _t
    def approver():
        for _ in range(60):
            st2, d2 = http_call(sup, "/v1/clm/sessions/ws%3Aagent/approvals",
                                headers=HDR)
            pend = d2.get("pending") or []
            if pend:
                http_call(sup, "/v1/clm/sessions/ws%3Aagent/approvals", "POST",
                          {"id": pend[0]["id"], "approve": True,
                           "by": "test"}, HDR)
                return
            _t.sleep(0.25)
    t = threading.Thread(target=approver, daemon=True)
    t.start()
    st, resp = http_call(sup, "/v1/chat/completions", "POST",
                         {"messages": [{"role": "user",
                                        "content": 'run command "echo approved"'}]},
                         HDR)
    t.join(10)
    assert st == 200
    acts = http_call(sup, "/v1/clm/sessions/ws%3Aagent/activity?limit=4",
                     headers=HDR)[1]["activity"]
    assert any(a["tool"] == "run_command" and a["status"] == "ok"
               for a in acts), acts


def test_serve_direct_web_search(sup):
    st, d = http_call(sup, "/v1/clm/sessions/ws%3Aagent/search", "POST",
                      {"query": "hello world", "max_results": 2}, HDR)
    assert st == 200 and len(d["hits"]) == 2


# --------------------------------------------------- CLI slash-command layer

def test_cli_agent_slash_commands_local(h):
    """handle_agent_command drives the same agent surface as /tools etc."""
    from clm_mem.cli import handle_agent_command
    seen = handle_agent_command(h, "/tools", 0)
    out = handle_agent_command(h, "/web unsloth", seen)
    acts = h.tools.activity(5)
    assert acts[0]["tool"] == "web_search"
    seen = handle_agent_command(h, '/run echo cli-yes', out)
    assert any(a["tool"] == "run_command" and a["status"] == "ok"
               for a in h.tools.activity(5))
    handle_agent_command(h, "/files .", seen)
    handle_agent_command(h, "/sandbox", seen)
    handle_agent_command(h, "/activity 3", seen)
    handle_agent_command(h, "/tools off exec", seen)
    assert h.tools.state()["enabled_categories"]["exec"] == "off"
    # exec off -> a direct /run is refused cleanly (no exception escapes)
    handle_agent_command(h, "/run echo nope", seen)
    handle_agent_command(h, "/tools on exec", seen)
    handle_agent_command(h, "/tools ask", seen)
    assert h.tools.broker.mode == "ask"
    handle_agent_command(h, "/approvals", seen)
    handle_agent_command(h, "/tools auto", seen)


def test_cli_responder_approves_local_ask(tmp_path):
    import builtins
    h = MemoryHarness(StubChat(), cfg_of(tmp_path, approval_mode="ask"))
    from clm_mem.cli import _install_prompt_responder
    answers = iter(["a"])                       # "always" -> auto afterwards
    orig = builtins.input
    builtins.input = lambda *a, **k: next(answers)
    try:
        _install_prompt_responder(h)
        res = h.tools.execute(ToolCall(id="1", name="run_command",
                                       args={"command": "echo prompted"}))
    finally:
        builtins.input = orig
    assert res.ok and h.tools.broker.mode == "auto"


def test_snapshot_carries_tool_state(sup):
    http_call(sup, "/v1/chat/completions", "POST",
              {"messages": [{"role": "user", "content": "search hi"}]}, HDR)
    st, snap = http_call(sup, "/v1/clm/sessions/ws%3Aagent/snapshot",
                         headers=HDR)
    assert st == 200 and snap["tools"]["active"]
    assert snap["activity"] and snap["activity"][0]["tool"] == "web_search"
