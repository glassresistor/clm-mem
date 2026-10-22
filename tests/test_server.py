"""End-to-end tests for `clm-mem serve`: proxy + mock backends as real
subprocesses over localhost HTTP. No GPU, no binaries, no downloads.
"""

import json
import urllib.error
import urllib.request
import pytest
from clm_mem import start_supervisor


def http_call(sup, path, method="GET", body=None, headers=None):
    url = f"{sup.base_url}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:  # we want the error body too
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


def sse_call(sup, path, body, headers=None):
    req = urllib.request.Request(
        f"{sup.base_url}{path}", data=json.dumps(body).encode(), method="POST"
    )
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    resp = urllib.request.urlopen(req, timeout=30)
    assert resp.headers.get("Content-Type", "").startswith("text/event-stream")
    text, done = "", False
    for raw_line in resp:
        line = raw_line.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "":
            continue
        if payload == "[DONE]":
            done = True
            break
        obj = json.loads(payload)
        delta = obj["choices"][0]["delta"]
        text += delta.get("content", "") or ""
    resp.close()
    return text, done


@pytest.fixture()
def sup(tmp_path):
    s = start_supervisor(
        public_port=0,
        served_model="mock-model",
        backend_mode="mock",
        systemone_mode="mock",
        harness_kwargs={
            "workdir": str(tmp_path),
            "max_context_tokens": 600,
            "memory_preamble": True,
        },
        timeout=30,
    )
    yield s
    s.shutdown()


# -- llama.cpp-shaped surface ---------------------------------------------------
def test_health_version_models(sup):
    st, health = http_call(sup, "/health")
    assert st == 200 and health["status"] == "ok"
    assert health["model"] == "mock-model"
    assert set(health["processes"]) >= {"backend-mock", "systemone-mock"}

    st, ver = http_call(sup, "/version")
    assert st == 200 and "llama.cpp" in ver

    st, models = http_call(sup, "/v1/models")
    ids = [m["id"] for m in models["data"]]
    assert "mock-model" in ids


def test_native_props_passthrough(sup):
    st, props = http_call(sup, "/props")
    assert st == 200 and props.get("model_type") == "mock"


def test_options_cors(sup):
    req = urllib.request.Request(
        f"{sup.base_url}/v1/chat/completions", method="OPTIONS"
    )
    resp = urllib.request.urlopen(req, timeout=30)
    assert resp.status == 204
    assert "X-Cllm-Mem-Session" in (
        resp.headers.get("Access-Control-Allow-Headers") or ""
    )


# -- chat: passthrough and memory-managed ---------------------------------------
def test_chat_passthrough_without_session(sup):
    st, resp = http_call(
        sup,
        "/v1/chat/completions",
        "POST",
        {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hello there"}],
        },
    )
    assert st == 200
    assert resp["choices"][0]["message"]["content"]
    assert "clm_mem" not in resp  # plain llama.cpp clients unchanged


def test_chat_streaming_passthrough(sup):
    text, done = sse_call(
        sup,
        "/v1/chat/completions",
        {
            "model": "mock-model",
            "stream": True,
            "messages": [{"role": "user", "content": "stream me please"}],
        },
    )
    assert done and text


def test_chat_session_memory_and_status(sup):
    hdr = {"X-Cllm-Mem-Session": "alice"}
    st, resp = http_call(
        sup,
        "/v1/chat/completions",
        "POST",
        {
            "model": "mock-model",
            "messages": [
                {"role": "user", "content": "Remember: API key KEY-9 port 8090"}
            ],
        },
        hdr,
    )
    assert st == 200
    assert resp["choices"][0]["message"]["content"]
    assert resp["clm_mem"]["session_id"] == "alice"

    st, status = http_call(sup, "/v1/clm/sessions/alice/status")
    assert st == 200
    assert status["turns"] == 1
    assert status["counts"]["longterm"] >= 1  # fact survived as longterm
    assert status["tokens"] <= 600  # budget respected

    # second turn still works and reuses the same harness
    st, resp2 = http_call(
        sup,
        "/v1/chat/completions",
        "POST",
        {
            "model": "mock-model",
            "messages": [
                {"role": "user", "content": "what was my API key? remember it"}
            ],
        },
        hdr,
    )
    assert st == 200 and resp2["choices"][0]["message"]["content"]

    st, status = http_call(sup, "/v1/clm/sessions/alice/status")
    assert status["turns"] == 2
    assert status["tokens"] <= 600


def test_chat_session_streaming(sup):
    hdr = {"X-Cllm-Mem-Session": "streamer"}
    text, done = sse_call(
        sup,
        "/v1/chat/completions",
        {
            "model": "mock-model",
            "stream": True,
            "messages": [
                {"role": "user", "content": "Remember: codename Aurora port 8042"}
            ],
        },
        hdr,
    )
    assert done and text
    st, status = http_call(sup, "/v1/clm/sessions/streamer/status")
    assert st == 200 and status["turns"] == 1


def test_session_list_reset_delete(sup):
    hdr = {"X-Cllm-Mem-Session": "bob"}
    http_call(
        sup,
        "/v1/chat/completions",
        "POST",
        {"model": "mock-model", "messages": [{"role": "user", "content": "hi there"}]},
        hdr,
    )

    st, lst = http_call(sup, "/v1/clm/sessions")
    assert st == 200 and "bob" in lst["sessions"]

    st, msg = http_call(sup, "/v1/clm/sessions/bob/reset", "POST")
    assert st == 200 and msg["reset"] == "bob"

    st, _ = http_call(sup, "/v1/clm/sessions/bob/status")
    assert st == 404  # reset dropped it

    http_call(
        sup,
        "/v1/chat/completions",
        "POST",
        {"model": "mock-model", "messages": [{"role": "user", "content": "hi again"}]},
        hdr,
    )
    st, msg = http_call(sup, "/v1/clm/sessions/bob", "DELETE")
    assert st == 200 and msg["deleted"] == "bob"


def test_unknown_session_status_404(sup):
    st, _ = http_call(sup, "/v1/clm/sessions/nobody/status")
    assert st == 404


# -- systemone proxy -------------------------------------------------------------
SYS_Q = {
    "duration": {
        "type": "choice",
        "instructions": "tag this block",
        "criteria": {
            "longterm": "durable",
            "shortterm": "working",
            "transient": "chatter",
        },
    }
}


def test_systemone_proxied_through_public_port(sup):
    st, resp = http_call(
        sup,
        "/v1/decisions",
        "POST",
        {
            "model": "sys-one",
            "state": {"text": "Remember: password hunter2"},
            "questions": SYS_Q,
        },
    )
    assert st == 200
    assert resp["answers"]["duration"]["choice"] == "longterm"


def test_decision_aliases_proxied(sup):
    """Legacy provider spellings (/v1/systemone, /v1/classifier) are still
    proxied to the decision server; same request/response contract."""
    for alias in ("/v1/systemone", "/v1/classifier"):
        st, resp = http_call(
            sup,
            alias,
            "POST",
            {
                "model": "sys-one",
                "state": "Remember: password hunter2",
                "questions": SYS_Q,
            },
        )
        assert st == 200, alias
        assert resp["answers"]["duration"]["choice"] == "longterm", alias


def test_systemone_client_sends_model_to_canonical_path():
    """Regression: the client sends the required 'model' field and always
    hits the full decision path (base + /v1/decisions) -- it never appends
    provider aliases onto a configured URL."""
    from clm_mem.llm import SystemOneClient
    from clm_mem.processes import ProcessManager, free_port

    mgr = ProcessManager()
    try:
        port = free_port()
        # the mock serves the canonical decision route (plus legacy aliases)
        h = mgr.start("s1", mgr.mock_argv("systemone"), port=port, timeout=30)
        c = SystemOneClient(h.root_url, model="featherless-ai/Test-classifier")
        out = c.system_one(
            "Remember: codename Nova",
            {
                "duration": {
                    "type": "choice",
                    "instructions": "tag",
                    "criteria": {"longterm": "l", "shortterm": "s", "transient": "t"},
                }
            },
        )
        assert out["answers"]["duration"]["choice"] == "longterm"
        assert c.model == "featherless-ai/Test-classifier"

        # a stub that rejects anything but /v1/decisions: the client must go
        # straight to the canonical path (and reject misconfigured URLs)
        import threading
        from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
        import json as _json

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status, obj):
                raw = _json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = _json.loads(self.rfile.read(n).decode())
                if not body.get("model"):  # mirrors both real APIs
                    self._send(422, {"error": "model required"})
                    return
                if self.path != "/v1/decisions":
                    # only the canonical decision route exists here
                    self._send(404, {"error": "no such route"})
                    return
                self._send(
                    200,
                    {
                        "answers": {
                            "q": {
                                "type": "choice",
                                "choice": "longterm",
                                "confidence": 1.0,
                            }
                        }
                    },
                )

            def log_message(self, *a):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as only:
            port2 = only.server_port
            t = threading.Thread(target=only.serve_forever, daemon=True)
            t.start()
            c2 = SystemOneClient(f"http://127.0.0.1:{port2}", model="m")
            out = c2.system_one(
                "x",
                {
                    "q": {
                        "type": "choice",
                        "instructions": "t",
                        "criteria": {"a": None, "b": None},
                    }
                },
            )
            assert out["answers"]["q"]["choice"] == "longterm"
            # a URL with an endpoint path appended is rejected outright --
            # the client sets the full path itself, never concatenates.
            for bad in (
                f"http://127.0.0.1:{port2}/v1/decisions",
                f"http://127.0.0.1:{port2}/v1/systemone",
                f"http://127.0.0.1:{port2}/v1/classifier",
            ):
                with pytest.raises(ValueError, match="bare server root"):
                    SystemOneClient(bad, model="m")
            only.shutdown()  # inside the with: stop serve_forever
    finally:
        mgr.shutdown()


def test_supervisor_rejects_systemone_url_with_path(tmp_path):
    """--systemone-url must stay a bare root; appending the decision endpoint
    (or any provider alias) to it is rejected at supervisor startup."""
    from clm_mem.server import normalize_decision_root

    assert (
        normalize_decision_root("http://127.0.0.1:8010/")
        == "http://127.0.0.1:8010"
    )
    assert normalize_decision_root("https://api.featherless.ai") == (
        "https://api.featherless.ai"
    )
    for bad in (
        "https://api.featherless.ai/v1/decisions",
        "http://127.0.0.1:8010/v1/systemone",
        "http://127.0.0.1:8010/v1/classifier",
    ):
        with pytest.raises(ValueError, match="bare server root"):
            start_supervisor(
                public_port=0,
                backend_mode="none",
                systemone_mode="external",
                systemone_url=bad,
                harness_kwargs={"workdir": str(tmp_path)},
                timeout=30,
            )


def test_decision_canonical_path_first(sup):
    """The proxy tries /v1/decisions before provider aliases, and relays the
    server's own error body when no route answers."""
    # mock serves the canonical path; a successful call caches it sticky
    st, resp = http_call(
        sup,
        "/v1/decisions",
        "POST",
        {
            "model": "sys-one",
            "state": "Remember: password hunter2",
            "questions": SYS_Q,
        },
    )
    assert st == 200
    assert sup.config.systemone_alias == "/v1/decisions"


def test_supervisor_spawns_and_kills_processes(tmp_path):
    s = start_supervisor(
        public_port=0,
        backend_mode="mock",
        systemone_mode="mock",
        timeout=30,
        harness_kwargs={"workdir": str(tmp_path)},
    )
    try:
        pids = [h.proc.pid for h in s.manager.handles]
        assert len(pids) == 2 and all(h.alive() for h in s.manager.handles)
    finally:
        s.shutdown()
    # everything must be reaped, including the proxy's own threads
    import os
    import time

    for pid in pids:
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except OSError:  # gone (or never existed) -> good
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"pid {pid} survived shutdown")
    assert s.manager.handles == []


def test_external_backend_mode(tmp_path):
    """backend_mode='external' proxies a server we did not spawn."""
    from clm_mem.processes import ProcessManager

    mgr = ProcessManager()
    try:
        backend = mgr.start("ext-backend", mgr.mock_argv("llm"), timeout=30)
        s = start_supervisor(
            public_port=0,
            served_model="mock-model",
            backend_mode="external",
            backend_url=f"{backend.base_url}",  # /v1 gets stripped
            systemone_mode="mock",
            timeout=30,
            harness_kwargs={"workdir": str(tmp_path)},
        )
        try:
            st, resp = http_call(
                s,
                "/v1/chat/completions",
                "POST",
                {
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "external hello"}],
                },
            )
            assert st == 200 and resp["choices"][0]["message"]["content"]
        finally:
            s.shutdown()
    finally:
        mgr.shutdown()


def test_spawn_requires_command():
    with pytest.raises(ValueError):
        start_supervisor(public_port=0, backend_mode="spawn", timeout=5)


def test_spawn_full_command_with_placeholders(tmp_path):
    """A spawn command may carry {host}/{port} itself (full-command style);
    then the default --host/--port flags are NOT appended a second time."""
    import sys

    s = start_supervisor(
        public_port=0,
        served_model="mock-model",
        backend_mode="spawn",
        systemone_mode="none",
        backend_cmd=[
            sys.executable,
            "-m",
            "clm_mem.mock_backends",
            "--kind",
            "llm",
            "--host",
            "{host}",
            "--port",
            "{port}",
        ],
        backend_flags=None,
        harness_kwargs={"workdir": str(tmp_path)},
        timeout=30,
    )
    try:
        st, resp = http_call(
            s,
            "/v1/chat/completions",
            "POST",
            {
                "model": "mock-model",
                "messages": [{"role": "user", "content": "full cmd hello"}],
            },
        )
        assert st == 200 and resp["choices"][0]["message"]["content"]
    finally:
        s.shutdown()


# -- context-window lockstep -----------------------------------------------------
def test_budget_follows_backend_ctx(tmp_path):
    """With no pinned budget, the harness adopts the backend's ctx size."""
    s = start_supervisor(
        public_port=0,
        served_model="mock-model",
        backend_mode="mock",
        systemone_mode="none",
        harness_kwargs={"workdir": str(tmp_path)},
        timeout=30,
    )
    try:
        assert s.config.backend_ctx == 4096  # mock /props n_ctx
        st, health = http_call(s, "/health")
        assert health["backend_ctx"] == 4096
        http_call(
            s,
            "/v1/chat/completions",
            "POST",
            {
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi there"}],
            },
            {"X-Cllm-Mem-Session": "ctx1"},
        )
        st, snap = http_call(s, "/v1/clm/sessions/ctx1/snapshot")
        assert snap["budget"] == 4096
        assert snap["backend_ctx"] == 4096
    finally:
        s.shutdown()


def test_settings_budget_clamped_to_backend_ctx(tmp_path):
    """The UI may shrink the budget, but never past the fixed llama.cpp window."""
    s = start_supervisor(
        public_port=0,
        served_model="mock-model",
        backend_mode="mock",
        systemone_mode="none",
        harness_kwargs={"workdir": str(tmp_path)},
        timeout=30,
    )
    try:
        http_call(
            s,
            "/v1/chat/completions",
            "POST",
            {
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi there"}],
            },
            {"X-Cllm-Mem-Session": "ctx2"},
        )
        st, d = http_call(
            s, "/v1/clm/sessions/ctx2/settings", "POST", {"max_context_tokens": 8192}
        )
        assert st == 200
        assert d["changed"]["max_context_tokens"] == 4096
        assert "clamped" in d["note"]
        # shrinking is fine and sticks
        st, d = http_call(
            s, "/v1/clm/sessions/ctx2/settings", "POST", {"max_context_tokens": 2048}
        )
        assert d["changed"]["max_context_tokens"] == 2048
        assert "note" not in d
    finally:
        s.shutdown()


def test_backend_none_mode_errors_cleanly(tmp_path):
    s = start_supervisor(
        public_port=0,
        backend_mode="none",
        systemone_mode="none",
        timeout=30,
        harness_kwargs={"workdir": str(tmp_path)},
    )
    try:
        st, body = http_call(
            s,
            "/v1/chat/completions",
            "POST",
            {"model": "x", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert st == 503
    finally:
        s.shutdown()
