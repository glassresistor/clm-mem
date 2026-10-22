# clm-mem

A **zero-training memory harness** for open chat models (Qwen and friends).
The same model that reads and writes text also classifies its own context as
**long-term** / **short-term** memory while decoding; a small deterministic
eviction engine then enforces the token budget.

Even though the eviction engine is deterministic the model itself is deciding
what to keep and how important it is as its creating context instead of after its grown.
So the model is in control but the hardware has more reliable expectations about its load
and no new LV cache strategies are required.

Design inspired by *Context Language Models* (Shao et al., arXiv:2609.37725):
context is treated as an editable artifact mirrored to disk, and context
management is delegated to the model via prompting  **no fine-tuning**,
works off-the-shelf with any OpenAI-compatible server (vLLM, Ollama,
llama.cpp, LM Studio).

One of CLMs core problems is that the method in the paper  invalidates the kvcache each request.
This means novel improvements had/have to be made to KV cache management for the paper to be effective
but  any improvements made would help this solution as well.
By compacting only when the context grows to a maximum size its reducing the amount KV churn.

This also means models can be improved by training within this harness and 
then having the trained model use the harness again until it performs worse than without it.


## How it works

Every message lives in the live window as a `Block` tagged
`longterm`, `shortterm`, or `transient`. Tagging happens two ways, using the
generated model itself:

1. **Inline ("classify as it decodes")**  a system preamble instructs the
   model to wrap durable content in `<longterm>…</longterm>` and working
   state in `<shortterm>…</shortterm>`. The harness splits each reply on
   those markers into separately-tagged blocks and strips the markup from
   what the user sees.
2. **Post-turn router pass**   a second cheap call to the *same* model
   labels this turn's untagged blocks as JSON (`longterm / shortterm /
   transient`). `classify_mode="both"` (default) uses both paths.

When the window exceeds `max_context_tokens`, eviction runs in phases:

| Phase | Condition | Action |
|---|---|---|
| **1** | over budget | delete every `transient` block (except the last `keep_recent_turns` turns and pinned system blocks) |
| **2** | window is now only long+short memory | evict **shortterm FIFO in rounds**: each round keeps the newest `shortterm_keep_pct` fraction, deletes the oldest; repeats while still over budget |
| **3** | long-term alone overflows | offload oldest long-term blocks to `archive.jsonl`, leave an evictable pointer stub (retrievable via `/recall`) |

Long-term memory is never silently destroyed   it is only ever offloaded to
the archive, fixing the "can offload but can't evict" failure mode the paper's
ContextBench exposes in tool-based baselines.

Each turn the live window is mirrored to `context.md` (human-readable,
paper-style `[[CTX_TURN n role=… tag=…]]` markers) and `context.jsonl`
(machine-readable), like the paper's context-as-a-file.

## Install

All dependencies are declared in **`pyproject.toml`** (the single source of
truth). Create a virtual environment and install the package into it:

```bash
cd clm-mem
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"     # package + pytest (recommended for hacking)
pip install -e .            # runtime only (needs: openai client)
pip install -e ".[tokens]"  # + exact token counting via the model's HF tokenizer
```

For a byte-for-byte reproducible environment, use the frozen lockfile that
pairs with this checkout (deps pinned to the exact versions tested):

```bash
pip install -r requirements.lock.txt   # pinned versions (also at workspace root)
pip install -e ./clm-mem               # then the package itself, editable
```

Deactivate / clean up any time with `deactivate` and by deleting `.venv/`.

Python ≥ 3.10.

## Serve a model

Any OpenAI-compatible endpoint works. Examples with Qwen:

```bash
# vLLM
vllm serve Qwen/Qwen3-8B-Instruct --max-model-len 16384

# Ollama (openai-compatible at :11434/v1)
ollama pull qwen3:8b
```

## Drop-in llama.cpp server: `clm-mem serve`

`clm-mem serve` turns the harness into a **single llama.cpp-shaped endpoint**:
to any OpenAI/llama.cpp client (Unsloth Studio, openai-python, text-gen-webui,
an LLM agent runtime...) it looks like one ordinary server at one host:port.
Behind it, `serve` brings up 2-3 background processes and wires the memory
harness into every request that carries a session header:

```
client ──http──> clm-mem proxy :8080          <- what clients talk to
                    ├── llama-server / Unsloth / vLLM / Ollama backend :auto
                    └── System One decision server (fork or mock)   :auto
```

Nothing is downloaded or compiled by `serve`  it only execs binaries or URLs
you point it at. With `--use-mock` it runs the bundled fake backends so the
whole stack works with no GPU and no weights.

```bash
# smoke-test the whole topology (mock backend + mock SystemOne):
clm-mem serve --use-mock --port 8080 --served-model mock-model \
    --max-context-tokens 600 --workdir /tmp/clm-serve

# real llama.cpp behind it:
clm-mem serve --port 8080 --llama-server ./build/bin/llama-server \
    --model-file models/Qwen3-8B-IQ2_K.gguf --ctx-size 16384 \
    --systemone-passthrough          # fork build answers the decision route itself

# already-running server + separate SystemOne API:
# NOTE: --systemone-url is the COMPLETE decision endpoint URL and is used
# exactly as given -- nothing is ever appended to it. Each provider exposes
# the decision endpoint under a different URL structure (/v1/decisions,
# /v1/systemone, /v1/classifier) but the request and response are identical
# everywhere, so point the flag at whichever your server answers.
clm-mem serve --backend-url http://localhost:8000/v1 \
    --systemone-url http://localhost:8010/v1/decisions

# hosted chat backend (Featherless, OpenRouter, ...): --backend-api-key is the
# bearer token the proxy sends on every LLM call it fronts -- including the
# harness's own calls and context-window discovery. Local servers ignore it;
# when unset, any Authorization header clients send is relayed instead:
clm-mem serve --port 8080 \
    --backend-url https://openrouter.ai/api/v1 \
    --backend-api-key "$OPENROUTER_API_KEY"

# pass the *entire* backend command line, options included (no flag guessing):
clm-mem serve --port 8080 \
    --backend-cmd "./build/bin/llama-server -m models/qwen.gguf --n-gpu-layers 99 --ctx-size 16384" \
    --systemone-cmd "./build/bin/server-decision -m systemone.gguf"

# hosted Featherless decision API instead of a local SystemOne server:
clm-mem serve --backend-url http://localhost:8000/v1 \
    --systemone-url https://api.featherless.ai/v1/classifier \
    --systemone-model featherless-ai/Qwen3.6-35B-A3B-classifier
    # auth via $FEATHERLESS_API_KEY or --systemone-api-key
```

When `--backend-cmd` (or `--systemone-cmd`) contains `{host}` / `{port}`
placeholders they are expanded in place; otherwise the server is launched with
`--host <ip> --port <port>` appended. The harness budget follows the backend:
if `--max-context-tokens` is not given, `serve` discovers the live llama.cpp
context window (`/props -> default_generation_settings.n_ctx`) and sets the
clm-mem budget to exactly that value, so the two windows never drift apart.

`serve` prints a machine-readable READY line (JSON) once everything is up, including the web-studio and MCP endpoints:

```json
{"ready": true, "base_url": "http://127.0.0.1:8080/v1", "root_url": "http://127.0.0.1:8080",
 "model": "mock-model", "backend": "http://127.0.0.1:...", "systemone": "http://127.0.0.1:...",
 "ui_url": "http://127.0.0.1:8080", "mcp_url": "http://127.0.0.1:8080/mcp",
 "processes": {"backend-mock": "...", "systemone-mock": "..."}}
```

**Client contract.** Point any OpenAI client at `base_url` with `api_key="EMPTY"`.
Requests *without* a session header are pure passthrough (behaves exactly like
plain llama.cpp). Requests *with* the header go through the memory harness:

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="EMPTY",
                       default_headers={"X-Cllm-Mem-Session": "unsloth-user"})
client.chat.completions.create(model="mock-model",
    messages=[{"role": "user",
               "content": "Remember: the deploy port is 8080."}])   # memory applies
```

Supported surface (llama.cpp + OpenAI union): `/v1/chat/completions` (incl.
SSE streaming), `/v1/models`, `/v1/decisions` (accepted incoming as the
provider alias spellings `/v1/systemone` and `/v1/classifier` too, same
contract; the proxy relays to `--systemone-url` verbatim -- that flag is the
complete endpoint URL and nothing is appended to it), native `/completion`,
`/props`, `/metrics`, `/version`, `/health`, CORS preflight, plus clm-mem
introspection routes `/v1/clm/sessions`, `/v1/clm/sessions/<id>/status`,
`.../reset`, and `DELETE /v1/clm/sessions/<id>` — and the full studio API
below.

## Web studio + workspaces + MCP

By default `serve` also hosts a single-page web studio at its root URL
(`http://127.0.0.1:8080/`; disable with `--no-ui`). It is built for people
juggling several projects:

* **Projects sidebar** — each workspace is an isolated sandbox (its own
  `context.md` / `archive.jsonl` under `<workdir>/workspaces/<name>`), with a
  live token meter and per-tier block dots. Create/delete from the UI or via
  `POST`/`DELETE /v1/clm/workspaces`.
* **Chat with context hints, not raw dumps** — long-term / short-term /
  transient appear as colored badges on your messages, eviction phases fire
  as compact chips under a reply (`🧹 phase2 · FIFO-evicted …`, `🛡 backstop ·
  kept #4`), and the context panel shows collapsed previews + token counts
  instead of full text ("show full block" fetches one block on demand).
* **Archive search** — type in the Archive tab to substring-search evicted
  memory; one click re-injects any hit back into the live window.
* **Tuning tab** — `shortterm_keep_pct`, budget and SystemOne threshold are
  changed live per project (`POST .../settings`).
* **Events timeline** — every classifier decision and eviction phase, with
  its kind badge and sequence number.

Session ids drive the studio: anything named `ws:<project>` gets that
project's sandbox; plain ids still behave like classic single sessions.

Studio API (same-origin, JSON):

```
GET    /v1/clm/workspaces                    list projects + live stats
POST   /v1/clm/workspaces        {name}     create a sandbox
DELETE /v1/clm/workspaces/<name>            delete it (files included)
GET    /v1/clm/sessions                    list live session ids
GET    /v1/clm/sessions/<id>/snapshot[?full=1]   full context view
GET    /v1/clm/sessions/<id>/archive?q=…   search the eviction archive
POST   /v1/clm/sessions/<id>/reinject {block_id}
POST   /v1/clm/sessions/<id>/settings {shortterm_keep_pct, max_context_tokens,
                                          systemone_evict_threshold}
GET    /v1/clm/sessions/<id>/tools          agent tool state (+ toggle)
POST   /v1/clm/sessions/<id>/tools {target,on} or {approval_mode}
GET    /v1/clm/sessions/<id>/activity?limit=&full=   tool activity log
GET    /v1/clm/sessions/<id>/approvals      pending + resolved approvals
POST   /v1/clm/sessions/<id>/approvals {id,approve}  decide one
POST   /v1/clm/sessions/<id>/search {query}         direct web search
POST   /v1/clm/sessions/<id>/toolcall {tool,args}   run any tool directly
GET    /v1/clm/sessions/<id>/sandbox[?path=]        sandbox info + file list
GET    /v1/clm/sessions/<id>/sandbox/file?path=     read one sandbox file
POST   /v1/clm/sessions/<id>/sandbox/exec {command} human terminal command
POST   /v1/clm/sessions/<id>/sandbox/write {path,content}
DELETE /v1/clm/sessions/<id>/sandbox              reset (wipe) the sandbox
```

**MCP toolbox.** `serve` is also an MCP server (streamable HTTP JSON-RPC at
`/mcp`): `initialize`, `tools/list`, `tools/call` with tools `clm_chat`,
`clm_status`, `clm_blocks`, `clm_recall`, `clm_reinject`,
`clm_set_keep_pct`, `clm_workspaces`, plus the agent surface — `clm_tools`,
`clm_tool_on`, `clm_tool_call`, `clm_sandbox_exec`, `clm_web_search`. Point
any MCP client at the URL and it gets the identical per-workspace memory and
the identical tools the web studio toggles control — same library, same
harnesses, no extra process.

## Agent mode: sessions that also act

Memory alone decides what a session keeps; `--tools` lets the same session
also *do* things — web search, sandboxed commands and file work. The point
of building it next to the memory harness is that **tool output is just more
context**: executed calls enter the live window as short-term blocks and obey
the same budget and eviction rules as chat, so an agent's long scratchpad is
forgettable working state instead of a context bomb.

Seven tools in three categories:

| Category | Tools | Notes |
|---|---|---|
| **web** | `web_search`, `web_fetch` | read-only; providers: `ddg` (no key), `brave`/`tavily`/`exa` (`$…_API_KEY`), `stub` (offline deterministic), `off` |
| **exec** | `run_command`, `run_python` | mutating; run inside the session sandbox |
| **fs** | `write_file`, `read_file`, `list_files` | read tools are read-only; paths are jail-contained |

**Sandbox levels** (`--sandbox-mode`): every workspace gets its own sandbox
under `<workdir>/workspaces/<name>/sandbox`. `jail` (default) is a *limited
process*: subprocess pinned to the sandbox dir with a scrubbed environment
(host secrets like API keys never leak in), wall-clock timeout, CPU/address-
space/file-size/process rlimits and output caps — strong enough for research
and demos, honest about not being a security boundary against a hostile
script. `docker` runs each command in a real container (`--network none`, memory/cpu/volume limits) when the docker CLI is available, falling back to
the jail with an explanatory note; the UI/CLI always show which level
actually ran. `off` removes exec/fs tools entirely.

**Approval policies** (`--ask` / `--approval-mode`): `auto` (default) runs
mutating tools inside the sandbox without asking; `ask` pauses them until a
human decides — inline prompt in a local CLI, live approval card in the web
studio, or REST/MCP from anywhere; `deny` never runs them. Read-only tools
never need approval, and human-initiated calls (composer buttons, slash
commands) are approved by definition.

```bash
clm-mem serve --use-mock --tools --search-provider stub   # offline demo
clm-mem serve --backend-url http://localhost:8000/v1 --tools \
    --sandbox-mode docker --ask                            # real model, containers
```

**Web studio UX (Unsloth-Studio-shaped).** The composer shows a tool bar —
toggles for 🌐 web / 💻 run / 📁 files plus an approval-policy select; when
the model uses tools mid-turn, **live activity cards** appear in the
transcript (spinner while running, ✓/✗/denied states, click to expand full
output), and under `ask` an **approval card** with approve/deny buttons
appears right where the request happened. A new **Sandbox tab** gives each
project a mini terminal (`$ …` log), file list with viewer, isolation level,
disk usage vs quota, and a reset button. Tool calls never print raw context
at you: summaries stay collapsed, details expand on demand.

**CLI UX (the same language at the prompt).** `clm-mem chat` gains an agent
cast of slash commands, identical in meaning to the studio's toggles/cards:

```
/tools                       state per category + policy + sandbox isolation
/tools on|off web|exec|fs|all|<tool>      toggles (same as the composer bar)
/tools auto|ask|deny         approval policy
/web <query>                 direct web search (results laid out, not dumped)
/fetch <url>                 fetch a page
/run <command>  /py <code>   sandbox exec / python snippet
/files [path]  /cat <path>  /rm <path>     file work
/sandbox [info|reset]        isolation level, usage vs quota, wipe
/activity [n]  /tool <seq>   activity log; expand one entry's output
/approvals   /approve <id>   /deny <id>    resolve pending requests
```

After a model turn that used tools the CLI prints a `⚡ tools used this
turn:` summary (one line per call, deduped across sessions). Under `ask`, a
local session prompts inline (`allow? [y/n/a]` — `a` switches the rest of the
session to auto); an *attached* session under `ask` resolves through the
studio cards or a second terminal running `/approvals` + `/approve <id>`.
The bundled mock backend answers `tools=[...]` requests with deterministic
tool-calls planned from wording ("search …", "run command …", "python …"),
so the whole agent loop — search → fetch → exec → files — is exercisable
offline end to end, in serve, web studio, CLI and MCP alike.

## Attach a CLI chat to a running server

`chat` and `serve` are two ends of the same stick — attach the prompt to a
running server (web studio stays usable meanwhile):

```bash
clm-mem serve --use-mock --port 8080                     # or with real backends
clm-mem chat --connect http://127.0.0.1:8080/v1 --workspace myproj
```

All in-session commands (`/status`, `/long`, `/recall`, `/reinject`, `/keep`)
then talk to the server's REST API — the same ones the studio's UI buttons
call — and `RemoteHarness` (in the library) mirrors live config knobs:
setting `h.cfg.shortterm_keep_pct = 0.8` pushes it to the running session.

## Run the chat harness

```bash
clm-mem chat \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-8B-Instruct \
  --max-context-tokens 8192 \
  --shortterm-keep-pct 0.5
```

Ollama variant:

```bash
clm-mem chat --base-url http://localhost:11434/v1 --model qwen3:8b \
  --max-context-tokens 6144 --shortterm-keep-pct 0.75
```

In-session commands:

```
/status                # tokens used, block counts per memory tier
/long    /short        # list live long-term / short-term blocks
/recall <text>         # substring search the eviction archive
/reinject <block-id>   # pull an archived block back into context (as shortterm)
/keep <pct>            # change shortterm_keep_pct mid-session, e.g. /keep 0.75
/quit
```

## Demo (no GPU, no server)

Exercises the full pipeline   inline tagging, router pass, all three eviction
phases   with a deterministic stub model:

```bash
clm-mem demo --max-context-tokens 900 --shortterm-keep-pct 0.5
```

You should see phases fire like:

```
  [evict] phase1: deleted 4 transient block(s)
  [evict] phase2: FIFO-evicted 3 of 5 shortterm (kept newest 50%)
  [evict] phase3: offloaded longterm #17 to archive, left pointer
```

## Key options

| Flag | Default | Meaning |
|---|---|---|
| `--max-context-tokens` | 8192 | live-window budget (keep below your server's `--max-model-len`) |
| `--shortterm-keep-pct` | 0.5 | **the memory knob**: fraction of short-term kept per FIFO round (`/keep` to change live) |
| `--keep-recent-turns` | 2 | trailing turns exempt from phase-1 transient deletion |
| `--classify-mode` | both | `inline`, `post`, or `both` tagging paths |
| `--tokenizer` | = model id | HF tokenizer for exact counts (heuristic ~4 chars/token if unavailable) |
| `--workdir` | `.` | where `context.md`, `context.jsonl`, `archive.jsonl` live |
| `--api-key` (`chat`/`demo`) | `EMPTY` | bearer token for the chat backend API; hosted services require one, local llama.cpp/vLLM/Ollama ignore it (`--backend-api-key` is an accepted alias) |
| `--backend-api-key` (`serve`) | unset | bearer token the proxy sends to the chat backend on every LLM call; when unset, clients' own `Authorization` headers are relayed |

## Use as a library

```python
from clm_mem import MemoryHarness, HarnessConfig, OpenAIChat

h = MemoryHarness(
    OpenAIChat("http://localhost:8000/v1", "Qwen/Qwen3-8B-Instruct"),
    HarnessConfig(max_context_tokens=8192, shortterm_keep_pct=0.5),
)

print(h.step("Remember: the staging deploy port is 8080."))
h.cfg.shortterm_keep_pct = 0.75          # tune the knob at runtime
print(h.status())                         # tokens/blocks per tier
print(h.recall("8080"))                   # search evicted/offloaded memory
```

## Tests

With the venv active (`source .venv/bin/activate`):

```bash
python -m pytest tests/ -q     # harness + serve/proxy tests, no GPU/server needed
```

## Limitations vs. the paper

- The eviction policy here is **harness-defined**; the paper's headline result
  comes from letting the model *edit* the context file itself (e.g. via a
  Bash tool). The `context.md` mirror is there if you want to bolt that on.
- Tag quality rides on the base model's zero-shot judgment; small models
  occasionally forget the markers   the post-turn router pass is the backstop.
- Phase-3 offload keeps pointer stubs in-window, so a pathological long-term-
  heavy session still needs `--max-context-tokens` headroom for the preamble.

## License

MIT
