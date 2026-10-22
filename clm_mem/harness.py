"""The memory harness: context-as-a-file mirror + three-phase eviction engine."""
from __future__ import annotations
import dataclasses
import json
import math
import os
import re
import time
from typing import Dict, List, Optional
from .blocks import LONG, SHORT, TRANSIENT, VALID_TAGS, Block

#: Base value assigned at classification time (forgetting-curve seed).
BASE_IMPORTANCE = {LONG: 1.0, SHORT: 0.6, TRANSIENT: 0.3}

#: Subject-slot extraction for supersession: ``codename Nova-7``, ``port=8012``,
#: ``port is 8012``. Deterministic lexical slots -- no model call, no embeddings.
#: ``id`` deliberately excluded as a named slot (too generic): free-standing
#: identifiers participate in reheating only, never in supersession.
SLOT_RE = re.compile(
    r"\b(codename|password|secret|token|api[_-]?key|version|port)\b"
    r"(?:[\s:=]{0,3}(?:(?:is|are|were|was|set|to|named|called)[\s:=]+)*?)"
    r"([A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*)", re.I)
#: Free-standing versioned identifiers (``Nova-7``) usable for reheating.
ID_RE = re.compile(r"\b([A-Za-z][A-Za-z+]+-\d+)\b")
from .config import HarnessConfig
from .tagging import CLASSIFIER_SYSTEM, MEMORY_PREAMBLE, TAG_RE, split_tagged, strip_tags
from .tokens import TokenCounter
from .tools import ToolCall, parse_llm_tool_calls, render_call


class MemoryHarness:
    """Wraps any chat backend and keeps its context inside a token budget.

    Lifecycle of one :meth:`step`:
      1. append the user message,
      2. evict to make room BEFORE generating,
      3. call the model on the current live context,
      4. split the reply on inline ``<longterm>/<shortterm>`` tags ("classify
         as it decodes"), optionally re-label with a post-turn router pass,
      5. evict again, then mirror the window to ``context.md`` / ``context.jsonl``.
    """

    def __init__(self, llm, cfg: Optional[HarnessConfig] = None,
                 systemone=None, tools_root: str = "", responder=None):
        """
        Args:
            llm: Chat backend with chat(messages) -> str method.
            cfg: Harness configuration.
            systemone: Optional SystemOneClient for typed decision queries
                (pre-classification + eviction backstop). If provided, user
                messages are classified before being sent to the LLM, and
                eviction checks run a backstop query before dropping blocks.
            tools_root: sandbox directory for agent tools; defaults to
                ``<workdir>/sandbox`` so each workspace keeps its own files.
            responder: optional blocking approval callback (CLI prompt) used
                when ``approval_mode='ask'``.
        """
        self.llm = llm
        self.cfg = cfg or HarnessConfig()
        self.systemone = systemone
        self.tools_root = tools_root or os.path.join(self.cfg.workdir,
                                                     "sandbox")
        from .tools import ToolManager      # here, not at import time, so the
        self.tools = ToolManager(self.cfg, root=self.tools_root,   # package stays
                                responder=responder)               # import-cheap
        self.counter = TokenCounter(self.cfg.tokenizer_name)
        self.blocks: List[Block] = []
        self.turn = 0
        # Events are structured so the web UI can render them as hints/timeline
        # entries instead of printing raw context at the user.
        self.events: List[Dict] = []
        self._next_id = 0
        self._seq = 0
        if self.cfg.memory_preamble:
            self.add("system", MEMORY_PREAMBLE, tag=LONG, pinned=True)

    # -- paths ------------------------------------------------------------
    @property
    def archive_path(self) -> str:
        return os.path.join(self.cfg.workdir, "archive.jsonl")

    @property
    def context_md_path(self) -> str:
        return os.path.join(self.cfg.workdir, "context.md")

    @property
    def context_jsonl_path(self) -> str:
        return os.path.join(self.cfg.workdir, "context.jsonl")

    def _mk(self, kind: str, text: str) -> Dict:
        """Structured event (eviction phase, classifier decision...).

        `kind` is a short machine-readable label the UI maps to a visual hint:
        evict-phase1/2/3, backstop-*, classify, warn.
        """
        self._seq += 1
        return {"seq": self._seq, "t": time.time(),
                "kind": kind, "text": text}

    def _event(self, kind: str, text: str) -> None:
        self.events.append(self._mk(kind, text))

    # -- basic ops ----------------------------------------------------------
    def add(self, role: str, content: str, tag: str = TRANSIENT,
            pinned: bool = False) -> Block:
        b = Block(id=self._next_id, role=role, content=content.strip(),
                  tag=tag, turn=self.turn, pinned=pinned,
                  importance=BASE_IMPORTANCE.get(tag, 0.3),
                  last_access=self.turn)
        self._next_id += 1
        self.blocks.append(b)
        return b

    def _retag(self, b: Block, new_tag: str) -> None:
        """Re-tag a block and re-seed its importance (never below the floor)."""
        b.tag = new_tag
        b.importance = max(BASE_IMPORTANCE.get(new_tag, 0.3), b.importance)

    def tokens(self) -> int:
        return sum(self.counter.count_block(b) for b in self.blocks)

    def over_budget(self) -> bool:
        return self.tokens() > self.cfg.max_context_tokens

    # -- classification (same model, no retraining) -------------------------
    def classify_new(self, force: bool = False) -> None:
        """Post-turn router pass: relabel this turn's untagged blocks.

        Uses SystemOne choice questions when available (faster, typed),
        falls back to LLM call otherwise.

        In ``inline``/``both`` mode the pass is skipped when the model already
        used inline markers for this turn -- otherwise a full extra LLM call
        would run every single turn (the user message block is never tagged
        inline, so an ungated check always finds candidates).
        """
        if self.cfg.classify_mode not in ("post", "both"):
            return
        if not force and any(b.tag in (LONG, SHORT) for b in self.blocks
                             if b.turn == self.turn and b.role == "assistant"):
            return  # inline tagging succeeded this turn; nothing to re-route
        cands = [b for b in self.blocks
                 if not b.pinned and b.turn == self.turn and b.tag == TRANSIENT]
        if not cands:
            return

        # Prefer SystemOne choice questions (typed, faster)
        if self.systemone:
            try:
                self._classify_with_systemone(cands)
                return
            except RuntimeError as e:
                self._event("classify", f"systemone classify failed: {e}")

        # Fall back to LLM router pass
        lines = "\n".join(
            f"id={b.id} role={b.role} text="
            f"{b.content[:self.cfg.classifier_max_block_chars]}"
            for b in cands)
        try:
            raw = self.llm.chat(
                [{"role": "system", "content": CLASSIFIER_SYSTEM},
                 {"role": "user", "content": f"Conversation blocks:\n{lines}"}],
                temperature=0.0, max_tokens=self.cfg.classifier_max_tokens)
            data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except (ValueError, KeyError):
            self._event("classify",
                        "unparseable router response; kept transient")
            return
        by_id = {b.id: b for b in self.blocks}
        for lab in data.get("labels", []):
            bid, tag = lab.get("id"), str(lab.get("tag", "")).lower()
            if bid in by_id and tag in VALID_TAGS and not by_id[bid].pinned:
                self._retag(by_id[bid], tag)

    def _classify_with_systemone(self, cands: List[Block]) -> None:
        """Batched post-turn classification using SystemOne choice questions."""
        questions = {}
        for b in cands:
            questions[f"tag_{b.id}"] = {
                "type": "choice",
                "instructions": (
                    f"What memory duration should this conversation block have? "
                    f"Role: {b.role}. Content: {b.content[:self.cfg.classifier_max_block_chars]}"
                ),
                "criteria": {
                    LONG: "Durable facts, decisions, IDs, specs, credentials, requirements likely needed much later",
                    SHORT: "Working state, intermediate reasoning, recent task output useful for a while but not forever",
                    TRANSIENT: "Pleasantries, filler, redundant restatement, small talk, ephemeral chatter"
                }
            }

        resp = self.systemone.system_one(
            {"blocks": [b.content for b in cands]}, questions)

        by_id = {b.id: b for b in self.blocks}
        for b in cands:
            qid = f"tag_{b.id}"
            choice = resp["answers"].get(qid, {}).get("choice", TRANSIENT)
            if choice in VALID_TAGS and not by_id[b.id].pinned:
                old_tag = by_id[b.id].tag
                if old_tag != choice:
                    self._retag(by_id[b.id], choice)
                    self._event("classify",
                                f"reclassified #{b.id}: {old_tag} -> {choice}")

    # -- forgetting strategies (phase-0 maintenance) --------------------------
    # Two opt-in passes that attack longterm saturation from complementary
    # directions: decay & reheating forgets *slowly* along a time-based
    # forgetting curve; supersede forgets *fast* when an update or duplicate
    # contradicts what is already stored. Both are pure local heuristics --
    # no model calls -- so they stay deterministic and cheap.
    def eff_importance(self, b: Block) -> float:
        """Importance discounted by turns since last access (half-life curve)."""
        gap = max(0, self.turn - b.last_access)
        return b.importance * math.pow(0.5, gap / self.cfg.decay_half_life)

    @staticmethod
    def _slots(text: str) -> dict:
        """Named subject slots ``slot -> value`` (supersession keys)."""
        return {k.lower(): v for k, v in SLOT_RE.findall(text)}

    @classmethod
    def _key_values(cls, text: str) -> set:
        """Lowercased value tokens used for access matching (reheating): slot
        values plus free-standing identifiers like ``Nova-7`` or ``sk-42``."""
        vals = {v.lower() for v in cls._slots(text).values()}
        vals |= {t.lower() for t in ID_RE.findall(text)}
        return vals

    def reheat(self, text: str) -> int:
        """Access event: blocks whose identifiers appear in ``text`` warm up."""
        vals = self._key_values(text)
        if not vals:
            return 0
        n = 0
        for b in self.blocks:
            if b.pinned or b.last_access == self.turn:
                continue
            if MemoryHarness._key_values(b.content) & vals:
                b.last_access = self.turn
                b.importance = min(1.0, b.importance + self.cfg.reheat_boost)
                n += 1
        return n

    def _decay_demote(self, log: List[Dict]) -> None:
        """Longterm whose effective importance fell below the demotion floor
        becomes shortterm -- working state again, so phase-2 FIFO can pick it
        up. Demotion never destroys; archive paths stay append-only."""
        for b in self.blocks:
            if b.tag != LONG or b.pinned:
                continue
            eff = self.eff_importance(b)
            if eff < self.cfg.demote_threshold:
                self._retag(b, SHORT)
                log.append(self._mk(
                    "maintain-decay",
                    f"demoted longterm #{b.id} -> shortterm "
                    f"(eff={eff:.2f} stale {self.turn - b.last_access} turns)"))

    def _supersede_pass(self, log: List[Dict]) -> None:
        """Updates overwrite: a new long fact sharing a subject slot with an
        older live long fact supersedes it (old demotes and is labelled);
        an exact duplicate is dropped at write time instead of admitted."""
        fresh = [b for b in self.blocks
                 if b.tag == LONG and not b.pinned and b.turn == self.turn]
        if not fresh:
            return
        # Candidates include demoted/superseded blocks: a restatement of a
        # fact that already lost its long tag is still a duplicate write.
        older = [(o, MemoryHarness._slots(o.content)) for o in self.blocks
                 if not o.pinned and o.turn < self.turn]
        for new in fresh:
            nslots = MemoryHarness._slots(new.content)
            if not nslots:
                continue
            dup_of = None
            hits: List[tuple] = []          # (old block, shared slot names)
            for old, oslots in older:
                shared = sorted(set(nslots) & set(oslots))
                if not shared:
                    continue
                # duplicate only when *every* shared slot agrees on its value;
                # a partial match (same codename, new port) is an update.
                if all(nslots[s].lower() == oslots[s].lower() for s in shared):
                    dup_of = old
                    break
                hits.append((old, shared))
            if dup_of is not None:
                # duplicate write: keep the original, drop this copy (archived,
                # still recallable), and reheat the survivor as touched.
                dup_of.last_access = self.turn
                self._drop([new])
                log.append(self._mk(
                    "maintain-dedup",
                    f"dropped duplicate longterm #{new.id} "
                    f"(same value as #{dup_of.id})"))
                continue
            for old, shared in hits:
                old.last_access = self.turn
                prefix = f"[superseded by #{new.id}] "
                if not old.content.startswith(prefix):
                    old.content = prefix + old.content
                self._retag(old, SHORT)
                if self.cfg.supersede_fast:
                    # forgotten fast: a contradicted fact loses its value at
                    # once instead of riding the decay curve down
                    old.importance = min(0.3, old.importance)
                log.append(self._mk(
                    "maintain-supersede",
                    f"longterm #{old.id} superseded by new #{new.id} "
                    f"(slots: {', '.join(shared)})"))

    # -- eviction engine ------------------------------------------------------
    def evict(self) -> List[str]:
        """Run the staged policy until under budget. Returns event-log lines.

        Phase 1: delete all transient blocks (outside the recent-turns
                 coherence window; pinned blocks are exempt).
        Phase 2: context is now only long+short -> FIFO rounds on shortterm,
                 each round keeping the newest ``shortterm_keep_pct`` fraction.
        Phase 3: longterm alone overflows -> offload oldest to archive.jsonl
                 and leave an evictable pointer stub.
        """
        log: List[Dict] = []

        # Phase 0 -- forgetting strategies (opt-in): run maintenance passes on
        # every enforcement point. Both are local heuristics, so they stay
        # deterministic and never touch the backend.
        self._maintain(log)

        if not self.over_budget():
            self.events.extend(log)      # maintenance events still surface
            return log

        # Phase 1 -- delete everything that is neither longterm nor shortterm.
        min_turn = max(0, self.turn - self.cfg.keep_recent_turns + 1)
        doomed = [b for b in self.blocks
                  if b.tag == TRANSIENT and not b.pinned and b.turn < min_turn]
        if doomed:
            # Backstop: ask SystemOne before clearing memory (batched)
            if self.systemone:
                doomed = self._backstop_eviction(doomed, "phase1", log)
            self._drop(doomed)
            log.append(self._mk("evict-phase1",
                                f"deleted {len(doomed)} transient block(s)"))

        # Phase 2 -- shortterm FIFO, keeping `shortterm_keep_pct` per round.
        phase2_rounds = 0
        while self.over_budget() and phase2_rounds < 10:  # prevent infinite loop
            phase2_rounds += 1
            shorts = sorted((b for b in self.blocks if b.tag == SHORT),
                            key=lambda b: b.id)
            if not shorts:
                break
            keep_n = math.floor(len(shorts) * self.cfg.shortterm_keep_pct)
            drop = shorts[: len(shorts) - keep_n] or shorts[:1]  # guarantee progress
            # Backstop: ask SystemOne before clearing memory (batched)
            if self.systemone:
                drop = self._backstop_eviction(drop, "phase2", log)
            if not drop:
                break  # backstop kept everything; can't make progress
            self._drop(drop)
            log.append(self._mk(
                "evict-phase2",
                f"FIFO-evicted {len(drop)} of {len(shorts)} shortterm "
                f"(kept newest {self.cfg.shortterm_keep_pct:.0%})"))

        # Phase 3 -- longterm overflow: offload, leave a pointer. Strict FIFO
        # (oldest first) unless decay is on, in which case the weakest live
        # long fact leaves first regardless of age.
        if self.over_budget() and self.cfg.offload_overflow:
            longs = [b for b in self.blocks if b.tag == LONG and not b.pinned]
            if self.cfg.decay_enabled:
                longs.sort(key=lambda b: (self.eff_importance(b), b.id))
            else:
                longs.sort(key=lambda b: b.id)
            while self.over_budget() and longs:
                b = longs.pop(0)
                # Backstop: ask SystemOne before offloading longterm memory
                if self.systemone:
                    try:
                        if not self.systemone.should_evict(b.content):
                            log.append(self._mk(
                                "backstop-phase3",
                                f"kept longterm #{b.id}"))
                            continue  # skip this one, try next
                    except RuntimeError as e:
                        log.append(self._mk(
                            "backstop-phase3",
                            f"backstop error #{b.id}: {e}"))
                self._archive([b])
                b.content = f"[offloaded longterm #{b.id} -> archive.jsonl]"
                b.tag = SHORT   # the pointer itself is disposable later
                log.append(self._mk("evict-phase3",
                                    f"offloaded longterm #{b.id} to archive, "
                                    f"left pointer"))

        if self.over_budget():
            log.append(self._mk(
                "warn", "still over budget after all phases "
                "(pinned + longterm alone exceed the window)"))
        self.events.extend(log)
        return log

    def _maintain(self, log: List[Dict]) -> None:
        """Phase-0 passes: supersession first (an update must be adjudicated
        before decay removes its candidate), then the forgetting curve."""
        if self.cfg.supersede_enabled:
            self._supersede_pass(log)
        if self.cfg.decay_enabled:
            self._decay_demote(log)

    def _backstop_eviction(self, doomed: List[Block], phase: str,
                           log: List[str]) -> List[Block]:
        """Batched SystemOne backstop: ask which blocks are safe to evict.

        Sends all candidates in a single API call with one noul question per
        block. Returns the subset that SystemOne approved for eviction.
        """
        if not doomed or not self.systemone:
            return doomed

        questions = {}
        for b in doomed:
            questions[f"drop_{b.id}"] = {
                "type": "noul",
                "instructions": (
                    f"Is this {phase} conversation block safe to remove from the "
                    "live context window? Answer yes only if it's transient chatter, "
                    "redundant information, or content that won't be needed again. "
                    "Answer no if it contains facts, decisions, working state, or "
                    "anything that might be referenced later."
                )
            }

        try:
            # Batch all eviction candidates into one SystemOne call
            resp = self.systemone.system_one(
                {"blocks": [b.content for b in doomed]}, questions)

            kept = []
            for i, b in enumerate(doomed):
                qid = f"drop_{b.id}"
                noul = resp["answers"].get(qid, {}).get("noul", 0.5)
                if noul >= self.cfg.systemone_evict_threshold:
                    log.append(self._mk(f"backstop-{phase}",
                                        f"cleared #{b.id} (noul={noul:.2f})"))
                else:
                    kept.append(b)
                    log.append(self._mk(f"backstop-{phase}",
                                        f"kept #{b.id} (noul={noul:.2f})"))
            return [b for b in doomed if b not in kept]
        except RuntimeError as e:
            log.append(self._mk(f"backstop-{phase}",
                                f"batched backstop error: {e}"))
            return doomed  # on error, evict as planned

    def _drop(self, doomed: List[Block]) -> None:
        self._archive(doomed)   # evict from live window, keep a copy on disk
        ids = {id(b) for b in doomed}
        self.blocks = [b for b in self.blocks if id(b) not in ids]

    def _archive(self, blocks: List[Block]) -> None:
        os.makedirs(self.cfg.workdir, exist_ok=True)
        with open(self.archive_path, "a", encoding="utf-8") as f:
            for b in blocks:
                f.write(json.dumps({"id": b.id, "turn": b.turn, "role": b.role,
                                    "tag": b.tag, "content": b.content}) + "\n")

    # -- recall: offload & retrieval (ContextBench KV-store style) --------------
    def recall(self, query: str, limit: int = 50) -> List[dict]:
        """Substring search over the eviction archive."""
        if not os.path.exists(self.archive_path):
            return []
        hits: List[dict] = []
        with open(self.archive_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if query.lower() in rec["content"].lower():
                    hits.append(rec)
                    if len(hits) >= limit:
                        break
        return hits

    def reinject(self, block_id: int) -> bool:
        """Pull an archived block back into the live window as shortterm."""
        if not os.path.exists(self.archive_path):
            return False
        with open(self.archive_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("id") == block_id:
                    self.add(rec["role"],
                             f"[recalled #{block_id}] {rec['content']}", tag=SHORT)
                    return True
        return False

    # -- context-as-a-file mirror ------------------------------------------------
    def serialize_messages(self) -> List[Dict[str, str]]:
        """The live window rendered as chat messages for the backend.

        Tool transcripts (role ``tool``) go up as ordinary assistant text: any
        OpenAI-compatible server understands that, and it keeps multi-step tool
        reasoning in context without needing ``role=tool`` message support.
        """
        return [{"role": "assistant" if b.role == "tool" else b.role,
                 "content": b.content} for b in self.blocks]

    def mirror_files(self) -> None:
        os.makedirs(self.cfg.workdir, exist_ok=True)
        with open(self.context_md_path, "w", encoding="utf-8") as f:
            f.write(f"<!-- live context mirror | turn {self.turn} | "
                    f"{self.tokens()}/{self.cfg.max_context_tokens} tokens -->\n")
            for b in self.blocks:
                f.write(f"[[CTX_TURN {b.turn} id={b.id} role={b.role} "
                        f"tag={b.tag}{' PINNED' if b.pinned else ''}]]\n"
                        f"{b.content}\n\n")
        with open(self.context_jsonl_path, "w", encoding="utf-8") as f:
            for b in self.blocks:
                f.write(json.dumps(dataclasses.asdict(b)) + "\n")

    # -- agent tools ---------------------------------------------------------
    def chat_tools(self, messages: List[Dict], specs: List[Dict]) -> Dict:
        """Tool-capable completion when the backend supports it."""
        fn = getattr(self.llm, "chat_tools", None)
        if fn is None:
            raise RuntimeError(f"{type(self.llm).__name__} has no chat_tools()")
        return fn(messages, tools=specs)

    def _render_tool_transcript(self, results: List[tuple]) -> str:
        """What executed tools land in the model's context (bounded)."""
        out = []
        for call, res in results:
            head = f"[tool {render_call(call.name, call.args)}]"
            body = res.summary if res.ok else f"ERROR: {res.error or res.summary}"
            detail = (res.detail or "")[:1200]
            chunk = f"{head}\n{body}"
            if detail:
                chunk += f"\n{detail}"
            out.append(chunk)
        return "\n\n".join(out)

    def _tool_event(self, res, call: ToolCall) -> None:
        kind = "tool-call" if res.ok else "tool-error"
        self._event(kind, res.summary if res.ok
                    else f"{res.summary}: {res.error}")
        if res.data.get("approval") == "denied":
            self._event("tool-ask", f"{render_call(call.name, call.args)} denied")

    def _agent_turn(self) -> Optional[str]:
        """Run the tool loop for this turn; None when nothing was produced."""
        specs = self.tools.llm_tools()
        if not specs:
            return None
        raw: str = ""
        used = 0
        while used < self.cfg.max_tool_calls:
            try:
                msg = self.chat_tools(self.serialize_messages(), specs)
            except Exception as exc:
                self._event("warn",
                            f"tool loop unavailable ({exc}); plain chat")
                raw = self.llm.chat(self.serialize_messages())
                break
            calls = parse_llm_tool_calls(msg)
            if not calls:
                raw = str(msg.get("content") or "")
                break
            batch: List[tuple] = []
            for c in calls:
                res = self.tools.execute(c)
                self._tool_event(res, c)
                batch.append((c, res))
                used += 1
            # role "tool": visible to the model as assistant text (see
            # serialize_messages) but never echoed back as a chat reply, and
            # tagged short-term so the scratchpad stays forgettable.
            self.add("tool", self._render_tool_transcript(batch), tag=SHORT)
        else:
            # budget of tool calls exhausted: ask for a final answer directly
            self._event("tool-call",
                        f"tool loop stopped at {used} calls; answering without")
            raw = self.llm.chat(self.serialize_messages())
        if not raw:
            raw = self.llm.chat(self.serialize_messages())
        return raw

    # -- top-level: one conversational turn ----------------------------------------
    def step(self, user_text: str) -> str:
        """Send a user message; returns the (tag-stripped) assistant reply.

        Side effects: eviction may run, files are mirrored, ``self.events`` grows.
        When tools are active the model may call web/exec/fs tools first; their
        results enter the window as short-term blocks and obey the same budget.
        """
        self.turn += 1

        # Pre-classify user request using SystemOne before sending to LLM
        tag = TRANSIENT
        if self.systemone:
            try:
                tag = self.systemone.classify_memory(user_text)
                self._event("classify", f"pre-classified user msg -> {tag}")
            except RuntimeError as e:
                self._event("classify", f"pre-classify failed: {e}")
        self.add("user", user_text, tag=tag)

        if self.cfg.decay_enabled:
            n = self.reheat(user_text)
            if n:
                self._event("maintain-reheat",
                            f"reheated {n} block(s) on access")

        self.evict()                                  # room BEFORE generating
        agent_raw = self._agent_turn() if self.tools.active else None
        raw = agent_raw if agent_raw is not None else \
            self.llm.chat(self.serialize_messages())
        if TAG_RE.search(raw) and self.cfg.classify_mode in ("inline", "both"):
            parts = split_tagged(raw)
            if any(tag != TRANSIENT for tag, _ in parts):
                # The model honoured the inline contract: mark its untagged
                # user message short-term instead of waiting for a router call.
                for b in self.blocks:
                    if (b.turn == self.turn and b.role == "user"
                            and b.tag == TRANSIENT and not b.pinned):
                        b.tag = SHORT
            for tag, chunk in parts:                 # classify as it decodes
                self.add("assistant", chunk, tag=tag)
        else:
            self.add("assistant", raw)
        self.classify_new()                           # post-turn router pass
        log = self.evict()                            # enforce budget after growth
        self.mirror_files()
        for e in log:
            print(f"  [evict] {e['text']}")
        return "\n".join(strip_tags(b.content) for b in self.blocks
                         if b.role == "assistant" and b.turn == self.turn)

    # -- introspection ---------------------------------------------------------------
    def live_blocks(self, tag: str = "") -> List[Block]:
        return [b for b in self.blocks if not tag or b.tag == tag]

    def block_by_id(self, block_id: int) -> Optional[Block]:
        for b in self.blocks:
            if b.id == block_id:
                return b
        return None

    def archive_size(self) -> int:
        """Number of archived records (cheap line count)."""
        if not os.path.exists(self.archive_path):
            return 0
        with open(self.archive_path, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def snapshot(self, preview_chars: int = 320, events_tail: int = 24,
                 include_full: bool = False, activity_tail: int = 14) -> Dict:
        """JSON-friendly view of the live window -- what the web UI renders.

        Blocks carry a `preview` plus token counts so the UI can show hints
        (long/short/transient badges, truncation markers) without ever
        printing raw context at the user.
        """
        blocks = []
        for b in self.blocks:
            entry = {
                "id": b.id, "role": b.role, "tag": b.tag, "turn": b.turn,
                "pinned": b.pinned,
                "tokens": self.counter.count_block(b),
                "chars": len(b.content),
            }
            if self.cfg.decay_enabled:
                entry["importance"] = round(self.eff_importance(b), 3)
            if include_full or len(b.content) <= preview_chars:
                entry["content"] = b.content
                entry["truncated"] = False
            else:
                entry["content"] = b.content[:preview_chars]
                entry["truncated"] = True
            blocks.append(entry)
        return {
            "turn": self.turn,
            "tokens": self.tokens(),
            "budget": self.cfg.max_context_tokens,
            "counts": self.counts(),
            "shortterm_keep_pct": self.cfg.shortterm_keep_pct,
            "decay_enabled": self.cfg.decay_enabled,
            "supersede_enabled": self.cfg.supersede_enabled,
            "decay_half_life": self.cfg.decay_half_life,
            "demote_threshold": self.cfg.demote_threshold,
            "archive": self.archive_size(),
            "blocks": blocks,
            "events": self.events[-events_tail:],
            # agent surface: what tools are on, what they did, what is pending
            "tools": self.tools.state(),
            "activity": self.tools.activity(limit=activity_tail),
            "approvals": self.tools.approvals() if self.tools.broker.mode != "auto"
            else {"pending": [], "history": [], "mode": "auto"},
        }

    def counts(self) -> Dict[str, int]:
        out = {LONG: 0, SHORT: 0, TRANSIENT: 0}
        for b in self.blocks:
            out[b.tag] += 1
        return out

    def status(self) -> str:
        n = self.counts()
        return (f"turn={self.turn} tokens={self.tokens()}/"
                f"{self.cfg.max_context_tokens} blocks={len(self.blocks)} "
                f"(long={n[LONG]} short={n[SHORT]} transient={n[TRANSIENT]})")
