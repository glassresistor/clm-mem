"""Mock HTTP backends so the whole serve/proxy stack can be exercised without a
GPU, weights, or any compiled binaries.

Two kinds of mock server live here:

* ``llm`` -- an OpenAI-compatible chat server (plus the llama.cpp native
  routes ``/completion``, ``/props``, ``/metrics``, ``/version``) driven by
  :class:`clm_mem.llm.StubChat`.
* ``systemone`` -- the System One Lite decision API
  (``POST /v1/systemone``) driven by :class:`clm_mem.llm.MockSystemOne`.

Run them as modules:
    python -m clm_mem.mock_backends --kind llm       --host 127.0.0.1 --port 8081
    python -m clm_mem.mock_backends --kind systemone --host 127.0.0.1 --port 8010

Once listening they print a single READY line on stdout:
    {"ready": true, "pid": ..., "host": "...", "port": ...}
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _word_chunks(text: str, size: int = 8):
    """Split into smallish SSE-ish chunks for streaming replies."""
    parts = re.findall(r"\S+\s?|\s+", text) or ([text] if text else [])
    out, buf = [], ""
    for p in parts:
        buf += p
        if len(buf) >= size:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out or ([text] if text else [])


class _Silent(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # pragma: no cover - noise control
        if os.environ.get("CLM_MEM_VERBOSE"):
            super().log_message(fmt, *args)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Cllm-Mem-Session",
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")

    def _send(
        self,
        status: int,
        payload: bytes | None = None,
        content_type: str = "application/json",
        extra=None,
    ):
        self.send_response(status)
        self._cors()
        if payload is None:
            payload = b""
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def send_json(self, obj, status: int = 200):
        self._send(status, json.dumps(obj).encode())

    def read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode() or "{}")
        except Exception:
            return {}


class LLMHandler(_Silent):
    MODEL = os.environ.get("CLM_MEM_MODEL", "mock-model")

    def setup_chat(self):
        from .llm import StubChat  # lazy, keeps stdlib-only startup

        if getattr(type(self), "_chat", None) is None:
            type(self)._chat = StubChat()  # shared across requests (stateless anyway)
        return type(self)._chat

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _reply_for(self, body: dict) -> str:
        chat = self.setup_chat()
        msg = self._message_for(body, chat)
        return msg.get("content") or ""

    def _message_for(self, body: dict, chat=None) -> dict:
        """Reply as a message dict, honouring ``tools`` with real tool_calls.

        The bundled stub plans one deterministic call per turn (see
        :class:`clm_mem.llm.StubChat.chat_tools`), so the whole agent loop --
        web search, sandbox exec, file work -- is exercisable without a GPU.
        """
        if chat is None:
            chat = self.setup_chat()
        msgs = body.get("messages")
        tools = body.get("tools")
        if isinstance(msgs, list) and msgs:
            if tools and hasattr(chat, "chat_tools"):
                return chat.chat_tools(msgs, tools)
            return {"role": "assistant", "content": self._plain_reply(chat, msgs)}
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)
        return {
            "role": "assistant",
            "content": chat.chat([{"role": "user", "content": str(prompt)}]),
        }

    @staticmethod
    def _plain_reply(chat, msgs) -> str:
        return chat.chat(msgs)

    @staticmethod
    def _tool_delta(message: dict):
        calls = message.get("tool_calls")
        if not calls:
            return None
        return [
            {
                "id": c.get("id"),
                "type": "function",
                "function": {
                    "name": (c.get("function") or {}).get("name"),
                    "arguments": (c.get("function") or {}).get("arguments"),
                },
            }
            for c in calls
        ]

    def _sse_stream(self, reply: str, model: str, message: dict | None = None):
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        created = int(time.time())

        def chunk(delta, finish=None):
            obj = {
                "id": "mock-proxy-1",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        **({"finish_reason": finish} if finish else {}),
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        chunk({"role": "assistant"})
        for piece in _word_chunks(reply):
            chunk({"content": piece})
        calls = self._tool_delta(message or {})
        if calls:
            chunk({"tool_calls": calls})
        chunk({}, finish="stop")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/health", "/v1/health"):
            self.send_json({"status": "ok", "model": self.MODEL})
        elif path == "/version":
            self.send_json(
                {"version": "mock", "built_at": 0, "llama.cpp": {"version": "mock"}}
            )
        elif path == "/v1/models":
            self.send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.MODEL,
                            "object": "model",
                            "owned_by": "clm-mem-mock",
                        }
                    ],
                }
            )
        elif path == "/props":
            self.send_json({"model_type": "mock", "n_ctx": 4096})
        elif path == "/metrics":
            self.send_json({"objective": {}})
        else:
            self.send_json({"error": "not found", "path": path}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self.read_body()
        if path == "/v1/chat/completions":
            message = self._message_for(body)
            reply = message.get("content") or ""
            model = body.get("model") or self.MODEL
            if body.get("stream"):
                self._sse_stream(reply, model, message)
                return
            msg: dict = {"role": "assistant", "content": reply}
            if message.get("tool_calls"):
                msg["tool_calls"] = message["tool_calls"]
            resp = {
                "id": "mock-proxy-1",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            self.send_json(resp)
        elif path == "/completion":
            reply = self._reply_for(body)
            self.send_json(
                {
                    "content": reply,
                    "truncated": 0,
                    "stop_type": "eos",
                    "token_count": len(reply.split()),
                }
            )
        else:
            self.send_json({"error": "not found", "path": path}, 404)


class SystemOneHandler(_Silent):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/health", "/v1/health"):
            self.send_json({"status": "ok", "service": "mock-systemone"})
        else:
            self.send_json({"error": "not found", "path": path}, 404)

    # canonical decision route plus the provider aliases some builds expose;
    # all three answer with the same request/response contract.
    SYSTEMONE_PATHS = ("/v1/decisions", "/v1/systemone", "/v1/classifier")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in self.SYSTEMONE_PATHS:
            self.send_json({"error": "not found", "path": path}, 404)
            return
        from .llm import MockSystemOne  # lazy

        body = self.read_body()
        mock = MockSystemOne()
        try:
            resp = mock.system_one(body.get("state", ""), body.get("questions", {}))
        except Exception as e:  # never crash the mock server
            resp = {"error": str(e)}
        self.send_json(resp)


HANDLERS = {"llm": LLMHandler, "systemone": SystemOneHandler}


def serve_forever_with_mock(kind: str, host: str = "127.0.0.1", port: int = 0):
    """Start a mock server and block forever serving requests."""
    if kind not in HANDLERS:
        raise ValueError(f"unknown mock kind {kind!r}")
    httpd = ThreadingHTTPServer((host, port), HANDLERS[kind])
    httpd.daemon_threads = True
    print(
        json.dumps(
            {"ready": True, "pid": os.getpid(), "host": host, "port": httpd.server_port}
        ),
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        pass
    finally:
        httpd.server_close()


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="clm_mem.mock_backends",
        description="Mock llama.cpp / SystemOne HTTP backends for testing.",
    )
    p.add_argument("--kind", required=True, choices=sorted(HANDLERS))
    # ProcessManager exports CLM_MEM_HOST / CLM_MEM_PORT; fall back to them so
    # a bare `python -m clm_mem.mock_backends --kind llm` binds the right port.
    p.add_argument("--host", default=os.environ.get("CLM_MEM_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("CLM_MEM_PORT", "0")))
    a = p.parse_args(argv)
    serve_forever_with_mock(a.kind, host=a.host, port=a.port)


if __name__ == "__main__":
    sys.exit(main())
