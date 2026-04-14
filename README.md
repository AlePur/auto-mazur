# auto-mazur

An autonomous LLM agent that lives on a computer and does real work.

---

## Core Design

The agent runs as a **hot loop** — ticking as fast as the LLM endpoint allows (no sleeps). Its "memory" is the filesystem and a SQLite database. The computer itself is the state.

### Four Cognitive Modes (Characters)

| Character | When | Tools | Sees |
|-----------|------|-------|------|
| **Executive** | Every main loop iteration | `assign_task`, `create_goal`, `update_goal`, `respond`, `request_reflection` | All goals, inbox, last session result, health issues |
| **Worker** | Inside a session, until task is done | `shell`, `read`, `write`, `finish` | Task description + context. Nothing about other goals or users. |
| **Reflector** | Periodic (every ~2000 ticks) or on demand | None | Journals, all goals, failures, knowledge index |
| **Summarizer** | Infrastructure utility | None | Raw text to compress (transcripts, ticks, journals) |

Each character is a **separate LLM call** with a different system prompt, different tools, and different context. They never see each other's context.

### What is a Tick?

The **tick counter** is the system's clock. Two kinds of ticks:
- **Executive tick**: one LLM call → one strategic decision
- **Worker tick**: one tool call (shell/read/write/finish) within a session

Both increment the same global counter.  The counter is persisted to the DB after every tick, so a crash loses at most one tick.

### How Tasks Flow Down

```
User message → inbox table
                ↓
Executive tick: sees inbox + all goals
                ↓
Executive calls assign_task(goal_id, description, criteria)
                ↓
Infrastructure builds task context (checkpoint, knowledge)
                ↓
Worker session: multi-turn conversation with tools
  shell("ls") → result → shell("cat foo") → result → ... → finish("done")
                ↓
SessionResult → DB + checkpoint file
                ↓
Next Executive tick sees the result
                ↓
Executive responds to user (or assigns next task)
```

### Scaling

Nothing grows unboundedly:

| Schedule | Operation |
|----------|-----------|
| Every 50 ticks | Rewrite CHECKPOINT.md (Summarizer) |
| Every 500 ticks | Write journal entry for each active goal (Summarizer) |
| Every 2000 ticks | Run Reflector — update priorities, distil knowledge |
| Every 5000 ticks | Write weekly summary from journals |
| Every 50000 ticks | Archive old ticks from DB → compressed JSONL; compress transcripts |

---

## Workspace Layout

```
workspace/
├── goals/
│   └── goal-001-<slug>/
│       ├── CHECKPOINT.md        # where we left off; read every session
│       ├── PLAN.md
│       ├── STATUS.md
│       ├── journal/             # periodic summaries of ticks
│       ├── sessions/            # raw transcripts (jsonl, gzipped after archival)
│       ├── src/
│       └── data/
├── knowledge/
│   └── <topic>.md               # distilled learnings; searchable
├── meta/
│   ├── PRIORITIES.md            # maintained by Reflector
│   ├── REFLECTIONS.md           # append-only reflection log
│   └── summaries/
│       └── weekly-<tick>.md
├── scratch/
└── archive/
    └── ticks-before-<N>.jsonl   # archived tick log
```

---

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

Or set `api_key_env` in `config.yaml` to use a different env var name.  
Any OpenAI-compatible endpoint works: change `api_base` and `model` in `config.yaml`.

### 3. Configure

Edit `config.yaml` — at minimum set `model` and `api_base`.

### 4. Run

```bash
# Start with no initial goals (Executive will figure out what to do)
python -m src

# Start with an initial goal
python -m src --seed-goal "explore this machine" \
    "Explore the system: OS, installed tools, disk usage, running services. Write findings to knowledge/this-machine.md"

# Verbose logging
python -m src -v

# Custom config
python -m src --config /etc/mazur/config.yaml
```

Stop with `Ctrl+C`. The agent resumes from the last tick on restart.

---

## Configuration Reference

All values in `config.yaml`. Override with `MAZUR_<FIELDNAME>` env vars (e.g. `MAZUR_MODEL=gpt-4o`).

```yaml
model: "gpt-4o"                   # model name
api_base: "https://api.openai.com/v1"
api_key_env: "OPENAI_API_KEY"     # env var holding the key
max_retries: 3

workspace_root: "./workspace"
db_path: "./agent.db"

command_timeout_seconds: 300      # kills hanging shell commands
max_output_bytes: 102400          # truncate stdout > 100KB
max_read_bytes: 102400            # truncate file reads > 100KB

max_actions_per_session: 200      # Worker tool calls before ending session
max_consecutive_errors: 5         # errors in a row → end session
context_compress_threshold_tokens: 60000

max_task_attempts: 3              # retry a task this many times

checkpoint_interval: 50
journal_interval: 500
reflection_interval: 2000
weekly_summary_interval: 5000
archive_interval: 50000

stuck_detection_window: 5
failure_streak_threshold: 10
neglect_threshold_ticks: 5000
```

---

## Gateway Integration

The inbox/outbox is exposed as hooks in `MainLoop`:

```python
from src.loop.main import MainLoop

class GatewayLoop(MainLoop):
    def _load_inbox(self) -> list[dict]:
        # Return unhandled messages from your DB/API
        return db.query("SELECT id, text, received_at FROM inbox WHERE handled = 0")

    def _deliver_outbox(self, entry: dict) -> None:
        # Write response to your DB/API
        db.execute(
            "INSERT INTO outbox (message_id, text) VALUES (?, ?)",
            entry["message_id"], entry["text"]
        )
```

---

## Architecture Notes

### Tool Safety

The Worker's tools are raw system access — no deny-lists, no path jails. The OS user's permissions are the only security boundary. Run the agent as a user with appropriate privileges.

Engineering limits (to protect the infrastructure, not the computer):
- Shell commands time out after `command_timeout_seconds`
- Large output/files are truncated before entering the context window
- Sessions end after `max_actions_per_session` tool calls

### Crash Recovery

State is always recoverable:
1. Tick counter: DB records every tick. Restart resumes from `last_tick + 1`.
2. Goal state: CHECKPOINT.md files survive crashes.
3. Sessions: transcripts are flushed after each tool call. Incomplete sessions are detected by NULL `status` in the sessions table.

### Extending

- **Add Executive tools**: extend `characters/executive.py` TOOL_SCHEMAS + `loop/actions.py`
- **Add Worker tools**: extend `tools.py` WORKER_TOOL_SCHEMAS + `ToolExecutor.execute()`  
- **Custom consolidation**: subclass `Consolidation`
- **Custom health checks**: subclass `HealthChecker`
- **Custom LLM**: set `api_base` to any OpenAI-compatible endpoint (Ollama, vLLM, LiteLLM, etc.)

---

## File Map

```
src/
├── __main__.py          entry point
├── config.py            Config dataclass + YAML/env loader
├── models.py            all domain types (Goal, Task, Session, TickRecord, …)
├── db.py                SQLite: schema, all queries
├── workspace.py         filesystem: goals, journals, knowledge, transcripts
├── llm.py               OpenAI-compatible client with retries
├── tools.py             shell/read/write + tool schemas for LLM
├── health.py            stuck detection, failure streaks, neglect detection
├── consolidation.py     journal, reflection, archival (hierarchical forgetting)
├── characters/
│   ├── executive.py     system prompt + tool schemas + response parser
│   ├── worker.py        system prompt (tools from tools.py)
│   ├── reflector.py     system prompt + output parser
│   └── summarizer.py    prompt templates (journal, compress, weekly, checkpoint)
├── context/
│   ├── executive.py     build Executive briefing
│   ├── worker.py        build Worker task context
│   └── reflector.py     build Reflector context
└── loop/
    ├── main.py          MainLoop — the forever orchestrator
    ├── executive.py     ExecutiveTick — one Executive LLM call
    ├── session.py       WorkerSession — multi-turn tool loop
    └── actions.py       ActionExecutor — carries out Executive decisions
```
