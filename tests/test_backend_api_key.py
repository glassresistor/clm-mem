"""Tests for --backend-api-key: bearer credentials on backend-bound calls.

A tiny auth-requiring stub stands in for a hosted OpenAI-compatible
service: it answers /props, /v1/models, /v1/chat/completions and the
SystemOne aliases only when the expected Authorization header arrives
(or accepts anything when its key is None), letting the tests observe
exactly which credential each hop carried.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from clm_mem import start_supervisor


def make_stub(key="", sys_key=""):
    """Start an auth-gated stub backend; returns (root_url, calls)."""
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # pragma: no cover
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")

        def _send(self, status, obj):
            payload = json.dumps(obj).encode()
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _auth(self):
            return self.headers.get("Authorization") or ""

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            auth = self._auth()
            calls.append(("GET", path, auth))
            want = f"Bearer {key}" if key else None
            if want and auth != want:
                self._send(401, {"error": {"message": "bad or missing key"}})
                return
            if path in ("/props", "/v1/props"):
                self._send(200, {"model_type": "authstub", "n_ctx": 4096})
            elif path == "/v1/models":
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": "auth-model", "object": "model"}],
                    },
                )
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?")[0]
            length = int(self.headers.get("Content-Length") or 0)
            body = (
                json.loads(self.rfile.read(length).decode() or "{}") if length else {}
            )
            auth = self._auth()
            calls.append(("POST", path, auth))
            if path == "/v1/chat/completions":
                want = f"Bearer {key}" if key else None
                if want and auth != want:
                    self._send(401, {"error": {"message": "bad or missing key"}})
                    return
                self._send(
                    200,
                    {
                        "id": "stub-1",
                        "object": "chat.completion",
                        "model": body.get("model", "auth-model"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"auth={auth or '(none)'}",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            elif path in ("/v1/decisions", "/v1/systemone", "/v1/classifier"):
                want = f"Bearer {sys_key}" if sys_key else None
                if want and auth != want:
                    self._send(401, {"error": {"message": "bad or missing key"}})
                    return
                answers = {}
                for qid, spec in (body.get("questions") or {}).items():
                    t = (spec or {}).get("type", "choice")
                    answers[qid] = {"type": t}
                    if t == "choice":
                        answers[qid]["choice"] = "shortterm"
                    elif t == "noul":
                        answers[qid]["noul"] = 0.25
                    else:
                        answers[qid]["score"] = 0.5
                self._send(200, {"answers": answers})
            else:
                self._send(404, {"error": "not found"})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    root = f"http://127.0.0.1:{httpd.server_port}"

    def shutdown():
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)

    return root, calls, shutdown


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
def keyed(tmp_path):
    """Stub backend that demands `Bearer secret-llm` on every route."""
    root, calls, stop = make_stub(key="secret-llm", sys_key=None)
    sup = start_supervisor(
        public_port=0,
        served_model="auth-model",
        backend_mode="external",
        backend_url=root,
        backend_api_key="secret-llm",
        systemone_mode="none",
        harness_kwargs={"workdir": str(tmp_path), "max_context_tokens": 600},
        timeout=30,
    )
    yield sup, calls
    sup.shutdown()
    stop()


def test_discovery_and_llm_calls_carry_the_key(keyed):
    sup, calls = keyed
    # context discovery (/props) needed auth too -- proof the key flows there
    assert sup.config.backend_ctx == 4096

    # session (harness/OpenAIChat) path
    st, resp = http_call(
        sup,
        "/v1/chat/completions",
        "POST",
        {
            "model": "auth-model",
            "messages": [{"role": "user", "content": "hello there"}],
        },
        headers={"X-Cllm-Mem-Session": "s-keyed"},
    )
    assert st == 200, resp
    content = resp["choices"][0]["message"]["content"]
    assert "auth=Bearer secret-llm" in content

    # plain passthrough path
    st, resp = http_call(
        sup,
        "/v1/chat/completions",
        "POST",
        {"model": "auth-model", "messages": [{"role": "user", "content": "plain"}]},
    )
    assert st == 200, resp
    assert "auth=Bearer secret-llm" in resp["choices"][0]["message"]["content"]

    # nothing reached the stub with any other credential
    assert calls
    bad = [c for c in calls if c[2] != "Bearer secret-llm"]
    assert not bad, bad


def test_client_credential_is_relayed_when_unset(tmp_path):
    root, calls, stop = make_stub(key=None)  # accepts anything, echoes
    sup = start_supervisor(
        public_port=0,
        served_model="auth-model",
        backend_mode="external",
        backend_url=root,
        systemone_mode="none",
        harness_kwargs={"workdir": str(tmp_path), "max_context_tokens": 600},
        timeout=30,
    )
    try:
        st, resp = http_call(
            sup,
            "/v1/chat/completions",
            "POST",
            {"model": "auth-model", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer client-abc"},
        )
        assert st == 200, resp
        assert "auth=Bearer client-abc" in resp["choices"][0]["message"]["content"]

        # with no credential anywhere, the stub sees nothing at all
        st, resp = http_call(
            sup,
            "/v1/chat/completions",
            "POST",
            {"model": "auth-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert st == 200, resp
        assert "auth=(none)" in resp["choices"][0]["message"]["content"]
    finally:
        sup.shutdown()
        stop()


def test_systemone_route_uses_configured_decision_key(tmp_path):
    root, calls, stop = make_stub(key=None, sys_key="secret-sys")
    sup = start_supervisor(
        public_port=0,
        served_model="auth-model",
        backend_mode="external",
        backend_url=root,
        systemone_mode="external",
        # complete endpoint URL, exactly as providers expose it
        systemone_url=f"{root}/v1/decisions",
        systemone_api_key="secret-sys",
        harness_kwargs={"workdir": str(tmp_path), "max_context_tokens": 600},
        timeout=30,
    )
    try:
        # proxied decision route: proxy must attach the systemone key itself
        st, resp = http_call(
            sup,
            "/v1/decisions",
            "POST",
            {
                "model": "systemone",
                "state": "hello",
                "questions": {"q": {"type": "noul", "instructions": "?"}},
            },
        )
        assert st == 200, resp
        assert "answers" in resp
        sys_calls = [c for c in calls if c[1].startswith("/v1/decisions")]
        assert sys_calls and all(
            c[2] == "Bearer secret-sys" for c in sys_calls
        ), sys_calls
    finally:
        sup.shutdown()
        stop()


def test_fork_passthrough_decision_route_inherits_backend_key(tmp_path):
    # stub requires `Bearer dual` on BOTH chat and decision routes, but the
    # supervisor is only given a backend key (fork serves the decision route
    # itself)
    root, calls, stop = make_stub(key="dual", sys_key="dual")
    sup = start_supervisor(
        public_port=0,
        served_model="auth-model",
        backend_mode="external",
        backend_url=root,
        backend_api_key="dual",
        systemone_mode="passthrough",
        harness_kwargs={"workdir": str(tmp_path), "max_context_tokens": 600},
        timeout=30,
    )
    try:
        st, resp = http_call(
            sup,
            "/v1/decisions",
            "POST",
            {
                "model": "systemone",
                "state": "hello",
                "questions": {"q": {"type": "noul", "instructions": "?"}},
            },
        )
        assert st == 200, resp
        assert "answers" in resp
    finally:
        sup.shutdown()
        stop()
