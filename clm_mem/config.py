"""Harness configuration. All knobs live here: ``shortterm_keep_pct`` controls
how much short-term memory survives each FIFO round, and the agent block below
decides whether a session may also *act* (web search, sandboxed commands,
file work) instead of only remembering."""
from __future__ import annotations

from dataclasses import dataclass

#: Fallback live-window budget when no backend context window can be found.
DEFAULT_CTX_BUDGET = 8192


@dataclass
class HarnessConfig:
    # --- budget ---------------------------------------------------------
    max_context_tokens: int = DEFAULT_CTX_BUDGET   # live-window token budget

    # --- eviction policy --------------------------------------------------
    shortterm_keep_pct: float = 0.5     # fraction of shortterm KEPT per FIFO round
    keep_recent_turns: int = 2          # coherence window exempt from phase-1 eviction
    offload_overflow: bool = True       # phase 3: longterm -> archive + pointer stub

    # --- classification (same model, no retraining) ------------------------
    classify_mode: str = "both"         # inline | post | both
    classifier_max_block_chars: int = 400
    classifier_max_tokens: int = 200    # generation cap for the router pass;
                                       # keeps a slow local backend from
                                       # rambling on the JSON label list

    # --- SystemOne decision API --------------------------------------------
    systemone_evict_threshold: float = 0.7  # noul confidence needed to evict

    # --- forgetting strategies (opt-in; defaults preserve prior behaviour) --
    # Strategy 1: decay & reheating -- a forgetting curve over block importance.
    # Stale longterm demotes to shortterm (so phase-2 FIFO can pick it up) and
    # phase-3 offload ranks by value instead of strict age. Reheat on access.
    decay_enabled: bool = False
    decay_half_life: int = 8        # turns for importance to halve without access
    demote_threshold: float = 0.45  # effective importance below which long -> short
    reheat_boost: float = 0.15      # importance bump on access (capped at 1.0)
    # Strategy 2: supersede -- updates and duplicates overwrite older longterm.
    # A new fact sharing a subject slot with an older one supersedes it; an
    # exact duplicate is dropped at write time instead of admitted twice.
    supersede_enabled: bool = False
    supersede_fast: bool = False    # when True, a superseded block loses its
    # importance immediately (forgotten fast); when False it keeps the value
    # it had and stays eligible for reheating until the decay curve demotes it

    # --- agent tools (opt-in; defaults preserve prior behaviour) --------------
    # The same session that remembers may also act: web search, sandboxed
    # commands and file work, fed back into context as short-term blocks.
    tools_enabled: bool = False       # master switch for the tool loop
    sandbox_mode: str = "jail"        # off | jail (limited process) | docker
    sandbox_image: str = "python:3.12-slim"   # docker mode only
    sandbox_disk_mb: int = 256        # soft disk quota per session sandbox
    sandbox_mem_mb: int = 512         # address-space / container memory cap
    approval_mode: str = "auto"       # auto | ask | deny (mutating tools)
    max_tool_calls: int = 8           # agent-loop iterations per turn
    tool_timeout: float = 30.0        # wall-clock cap per exec / fetch call
    approval_timeout: float = 300.0   # how long a pending approval waits
    search_provider: str = "ddg"      # ddg | brave | tavily | exa | stub | off
    max_search_results: int = 5
    fetch_max_chars: int = 6000       # web_fetch body cap
    exec_output_limit: int = 8000     # stdout/stderr cap per command

    workdir: str = "."                  # context.md / context.jsonl / archive.jsonl live here
    tokenizer_name: str = ""            # HF tokenizer id for exact counts (optional)
    memory_preamble: bool = True        # inject the CLM memory system prompt on first step

    def __post_init__(self) -> None:
        if not 0.0 <= self.shortterm_keep_pct < 1.0:
            raise ValueError("shortterm_keep_pct must be in [0, 1)")
        if self.classify_mode not in ("inline", "post", "both"):
            raise ValueError("classify_mode must be inline | post | both")
        if self.max_context_tokens < 64:
            raise ValueError("max_context_tokens too small to hold anything useful")
        if self.decay_half_life < 1:
            raise ValueError("decay_half_life must be >= 1 turn")
        if not 0.0 <= self.demote_threshold < 1.0:
            raise ValueError("demote_threshold must be in [0, 1)")
        if not 0.0 <= self.reheat_boost <= 1.0:
            raise ValueError("reheat_boost must be in [0, 1]")
        if self.sandbox_mode not in ("off", "jail", "docker"):
            raise ValueError("sandbox_mode must be off | jail | docker")
        if self.approval_mode not in ("auto", "ask", "deny"):
            raise ValueError("approval_mode must be auto | ask | deny")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")
        if self.tool_timeout < 1:
            raise ValueError("tool_timeout must be >= 1s")
        if not 1 <= self.max_search_results <= 25:
            raise ValueError("max_search_results must be in [1, 25]")
