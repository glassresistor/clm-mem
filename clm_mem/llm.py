"""LLM clients. Anything with a ``chat(messages) -> str`` method works;
OpenAIChat targets any OpenAI-compatible server, StubChat powers the offline
demo and the test-suite without a GPU.

SystemOneClient wraps the System One Lite decision API (http://127.0.0.1:8010)
for typed questions -- used for memory classification and eviction backstops."""

from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional
from .blocks import LONG, SHORT, TRANSIENT


class OpenAIChat:
    """Thin wrapper over any OpenAI-compatible endpoint (vLLM, Ollama,
    llama.cpp server, LM Studio, ...)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-8B-Instruct",
        api_key: str = "EMPTY",
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ):
        from openai import OpenAI  # lazy: keeps `demo` runnable without the dep

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_tools(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        """One tool-capable completion; returns the raw assistant message.

        Servers that reject ``tools`` raise, and the caller falls back to a
        plain chat -- old llama.cpp builds stay supported.
        """
        kwargs: Dict[str, object] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        try:
            out = msg.model_dump()
        except AttributeError:  # pragma: no cover - older openai SDK
            out = {"role": "assistant", "content": getattr(msg, "content", None) or ""}
            if getattr(msg, "tool_calls", None):
                out["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.function.name,
                            "arguments": c.function.arguments,
                        },
                    }
                    for c in msg.tool_calls
                ]
        out.setdefault("role", "assistant")
        return out


class StubChat:
    """Deterministic fake LLM: emits inline memory tags and answers the
    memory-router classification prompt with keyword heuristics."""

    _FACTS = [
        "<longterm>User's project codename is {i}-Aurora; deploy port is 80{i}.</longterm> Noted.",
        "Sure -- working through it now. Step A done, step B pending (call {i}).",
        "Remember: final decision v{i} ships Friday with schema rev {i}."
        "<shortterm>open item: review diff {i}</shortterm>",
        "Ha, good one! Anyway, continuing.",
        "<longterm>API key stored as KEY_{i}; table name is users_v{i}.</longterm> Saved it.",
    ]
    _LONG_KW = (
        "remember",
        "decision",
        "password",
        "api key",
        "codename",
        "deadline",
        "spec",
        "final",
        "port",
        "schema",
    )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        first = messages[0]["content"] if messages else ""
        last = messages[-1]["content"] if messages else ""
        if "memory router" in first:
            return self._classify(last)
        i = last.count("\n") + len(last) % 97
        return self._FACTS[i % len(self._FACTS)].format(i=i)

    # -- deterministic tool-calling (offline agent demos / tests) ------------
    #: marker the harness uses for executed-tool transcript blocks
    TOOL_MARK = "tool "

    _QUOTED = re.compile(r"[\"'`]([^\"'`]{1,200})[\"'`]")

    def chat_tools(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        """Plan one tool call from the user's wording, or converge to an answer.

        Deterministic keyword planner -- no network, no GPU. If a tool
        transcript is already in context it answers with a summary instead of
        calling again, so offline agent loops always terminate.
        """
        specs = {} if not tools else {}
        for t in tools or []:
            fn = t.get("function") or {}
            name = fn.get("name") or t.get("name")
            if name:
                specs[name] = t
        if not specs:
            return {"role": "assistant", "content": self.chat(messages)}
        # only transcripts emitted *after* the current user message count, so
        # one turn's tool work never blocks the next turn from planning
        idx = -1
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                idx = i
        transcript = [
            m
            for m in messages[idx + 1 :]
            if str(m.get("content", "") or "").startswith("[" + self.TOOL_MARK)
        ]
        if transcript:
            return {"role": "assistant", "content": self._tool_answer(transcript)}
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break
        plan = self._plan_tool(last_user, specs)
        if plan is None:
            return {"role": "assistant", "content": self.chat(messages)}
        name, args = plan
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "stub-call-1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }

    def _plan_tool(self, text: str, specs: Dict[str, Dict]):
        low = text.lower()
        quoted = self._QUOTED.findall(text)
        if (
            any(
                k in low
                for k in ("search", "look up", "find out", "google", "latest", "who is")
            )
            and "web_search" in specs
        ):
            q = quoted[0] if quoted else text.strip()
            return "web_search", {"query": q, "max_results": 3}
        if (
            any(
                k in low
                for k in ("run command", "shell", "terminal", "execute", "run ")
            )
            and "run_command" in specs
        ):
            cmd = quoted[0] if quoted else "echo hello from the sandbox"
            return "run_command", {"command": cmd}
        if (
            any(
                k in low
                for k in ("python", "code", "compute", "calculate", "write a function")
            )
            and "run_python" in specs
        ):
            code = quoted[0] if quoted else "print(sum(i*i for i in range(10)))"
            return "run_python", {"code": code}
        if (
            any(
                k in low
                for k in ("write file", "save to file", "create a file", "asset")
            )
            and "write_file" in specs
        ):
            path = "notes.md"
            body = quoted[0] if quoted else "created by clm-mem agent"
            return "write_file", {"path": path, "content": body + "\n"}
        if any(k in low for k in ("list files", "what files", "read file")):
            if "list_files" in specs:
                return "list_files", {"path": "."}
            if "read_file" in specs and quoted:
                return "read_file", {"path": quoted[0]}
        return None

    def _tool_answer(self, transcript: List[Dict]) -> str:
        lines = []
        for m in transcript:
            content = str(m.get("content", ""))
            head = content.split("\n", 1)[0]
            body = content.split("\n", 1)[1] if "\n" in content else ""
            # show the call itself, not the harness markup
            lines.append(head.replace("[tool ", "").replace("]", "")[:160])
            if body:
                lines.append(body[:200].replace("\n", " "))
        return "Done — I used the tools you asked for:\n" + "\n".join(lines)

    @classmethod
    def _classify(cls, prompt: str) -> str:
        labels = []
        for m in re.finditer(r"id=(\d+) role=(\w+) text=(.*)", prompt):
            bid, _role, txt = m.groups()
            low = txt.lower()
            tag = (
                LONG
                if any(k in low for k in cls._LONG_KW)
                else (TRANSIENT if len(txt) < 60 else SHORT)
            )
            labels.append({"id": int(bid), "tag": tag})
        return json.dumps({"labels": labels})


#: The canonical decision endpoint path. Every SystemOne-style server -- the
#: llama.cpp fork, the mock, and the hosted Featherless classifier service --
#: answers the *same* request/response contract at this route; providers may
#: additionally expose aliases (/v1/systemone, /v1/classifier) but clients
#: talk to ``/v1/decisions`` directly. ``--systemone-url`` must therefore be
#: a bare root (scheme://host[:port]); the client appends this path and never
#: appends anything else to the configured URL.
SYSTEMONE_PATH = "/v1/decisions"

#: Provider-specific aliases for the decision endpoint (llama.cpp forks answer
#: /v1/systemone; Featherless documents /v1/classifier). Clients never use
#: them -- they exist only so a misconfigured ``--systemone-url`` that already
#: ends in one of these paths can be rejected with a helpful message.
SYSTEMONE_ALIASES = ("/v1/systemone", "/v1/classifier")

#: Both APIs reject bodies without a non-empty ``model`` field, so the client
#: always sends one; override via --systemone-model / the studio settings.
DEFAULT_SYSTEMONE_MODEL = "systemone"


class SystemOneClient:
    """HTTP client for the System One Lite decision API.

    Sends typed questions (choice, score, noul) and returns calibrated
    answers. Used for memory classification and eviction backstops.

    Works against both the local llama.cpp SystemOne fork and the hosted
    Featherless classifier endpoint: every provider exposes the same request
    and response contract, so the client always POSTs to the full decision
    path (``base_url + /v1/decisions``). ``base_url`` must be a bare root --
    never append paths (or aliases like /v1/systemone or /v1/classifier) to
    it. The request body always carries a ``model`` field (required by both
    servers), plus optional bearer auth.

    Default base URL is http://127.0.0.1:8010 (local System One Lite server).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8010",
        model: str = "",
        api_key: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.lower().startswith(("http://", "https://")):
            raise ValueError(
                f"systemone-url must be a full URL with a scheme, e.g. "
                f"http://127.0.0.1:8010 or https://api.featherless.ai; got "
                f"{base_url!r}"
            )
        if self.base_url.endswith(SYSTEMONE_PATH) or self.base_url.endswith(
            SYSTEMONE_ALIASES
        ):
            raise ValueError(
                f"systemone-url must be the bare server root (e.g. "
                f"http://127.0.0.1:8010 or https://api.featherless.ai); got "
                f"{base_url!r} -- do not append endpoint paths to it; the "
                f"client sets the full path to the decision endpoint itself"
            )
        self.model = model or DEFAULT_SYSTEMONE_MODEL
        self.api_key = api_key or os.environ.get("FEATHERLESS_API_KEY", "")

    def system_one(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a state against typed questions.

        Always POSTs to the full decision path (``base_url + /v1/decisions``);
        no URL is appended to the configured root beyond that fixed path.

        Args:
            state: The input text or JSON object to evaluate.
            questions: Dict of question_id -> question spec. Each spec has
                'type' (choice/score/noul), 'instructions', and type-specific
                fields ('criteria' for choice, etc.).

        Returns:
            Response dict with 'answers' keyed by question_id.
        """
        import urllib.request
        import urllib.error

        url = f"{self.base_url}{SYSTEMONE_PATH}"
        body = json.dumps(
            {"model": self.model, "state": state, "questions": questions}
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            # Featherless returns 404 for "model not enabled for the
            # classifier"; carry the error body through so a wrong
            # --systemone-model surfaces as the server's message.
            try:
                detail = e.read().decode(errors="replace")[:400]
            except Exception:
                detail = ""
            raise RuntimeError(f"SystemOne API error {e.code} at {url}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"SystemOne API error: {e}") from e

    def classify_memory(self, text: str) -> str:
        """Classify a conversation block's memory duration.

        Returns one of 'longterm', 'shortterm', or 'transient'.
        """
        questions = {
            "duration": {
                "type": "choice",
                "instructions": (
                    "What memory duration should this conversation block have? "
                    "Choose based on whether it contains durable facts/decisions "
                    "that must survive long-term, working state needed for a while, "
                    "or transient chatter that can be forgotten."
                ),
                "criteria": {
                    LONG: "Durable facts, decisions, IDs, specs, credentials, requirements likely needed much later",
                    SHORT: "Working state, intermediate reasoning, recent task output useful for a while but not forever",
                    TRANSIENT: "Pleasantries, filler, redundant restatement, small talk, ephemeral chatter",
                },
            }
        }
        resp = self.system_one({"text": text}, questions)
        return resp["answers"]["duration"]["choice"]

    def should_evict(self, block_content: str) -> bool:
        """Backstop check: is this block safe to evict from live memory?

        Returns True if safe to drop, False if it should be kept.
        """
        questions = {
            "safe_to_drop": {
                "type": "noul",
                "instructions": (
                    "Is this conversation block safe to remove from the live "
                    "context window? Answer yes only if it's transient chatter, "
                    "redundant information, or content that won't be needed again. "
                    "Answer no if it contains facts, decisions, working state, or "
                    "anything that might be referenced later."
                ),
            }
        }
        resp = self.system_one({"block": block_content}, questions)
        # noul >= 0.7 means reasonably confident it's safe to drop
        return resp["answers"]["safe_to_drop"]["noul"] >= 0.7


class MockSystemOne:
    """Offline mock of SystemOneClient for testing and demo mode.

    Uses keyword heuristics instead of an actual model to simulate typed
    decisions. No HTTP calls required.
    """

    _LONG_KW = (
        "remember",
        "decision",
        "password",
        "api key",
        "codename",
        "deadline",
        "spec",
        "final",
        "port",
        "schema",
    )

    def __init__(self):
        pass

    def system_one(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate typed decisions using keyword heuristics."""
        answers = {}
        for qid, spec in questions.items():
            if spec["type"] == "choice":
                # Extract text from state
                if isinstance(state, dict):
                    text = " ".join(str(v) for v in state.values())
                else:
                    text = str(state)
                low = text.lower()
                if any(k in low for k in self._LONG_KW):
                    answers[qid] = {
                        "type": "choice",
                        "choice": LONG,
                        "confidence": 0.85,
                    }
                elif len(text) < 60:
                    answers[qid] = {
                        "type": "choice",
                        "choice": TRANSIENT,
                        "confidence": 0.75,
                    }
                else:
                    answers[qid] = {
                        "type": "choice",
                        "choice": SHORT,
                        "confidence": 0.80,
                    }
            elif spec["type"] == "noul":
                # Simulate: short/filler content is safe to drop
                if isinstance(state, dict):
                    text = " ".join(str(v) for v in state.values())
                else:
                    text = str(state)
                low = text.lower()
                has_facts = any(k in low for k in self._LONG_KW)
                is_short = len(text) < 60
                # High noul (yes, safe to drop) for short filler without facts
                if is_short and not has_facts:
                    answers[qid] = {"type": "noul", "noul": 0.85}
                elif has_facts:
                    answers[qid] = {"type": "noul", "noul": 0.25}
                else:
                    answers[qid] = {"type": "noul", "noul": 0.60}
            else:
                answers[qid] = {"type": spec["type"]}
        return {"answers": answers, "model": "mock-systemone"}

    def classify_memory(self, text: str) -> str:
        """Mock classification using keyword heuristics."""
        low = text.lower()
        if any(k in low for k in self._LONG_KW):
            return LONG
        elif len(text) < 60:
            return TRANSIENT
        else:
            return SHORT

    def should_evict(self, block_content: str) -> bool:
        """Mock eviction decision using keyword heuristics."""
        low = block_content.lower()
        has_facts = any(k in low for k in self._LONG_KW)
        is_short = len(block_content) < 60
        return (is_short and not has_facts) or not has_facts
