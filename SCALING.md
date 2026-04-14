# Scaling Analysis — auto-mazur

This document explains how the agent handles unbounded growth in disk storage
and LLM context.  It also lists every prompt the agent sends to an LLM and
the token-size budget for each.

---

## 1. Constraints

| Resource          | Constraint |
|-------------------|------------|
| Disk              | Infinite (no limit assumed) |
| LLM context       | `context_length_tokens` (config, minimum 100 000) |
| Tool output       | `max_output_bytes` (default 100 KB) |
| File read (raw)   | `max_read_bytes` (default 100 KB) |
| File read (chars) | `max_read_chars` (default 30 000 chars ≈ 7 500 tokens) |
| Read lines        | `max_read_lines` (default 100 lines, overridden by `lines` param) |
| Session actions   | `max_actions_per_session` (default 200) |
| Executive queries | `max_executive_queries` (default 10 per tick) |
| Compress trigger  | `context_compress_threshold_tokens` (default 0 = auto = 60 % of `context_length_tokens`) |

---

## 2. Context budget — by prompt type

### 2a. Worker session (most critical)

The Worker conversation grows every turn.  It is actively managed.

| Slot | Content | Bound |
|------|---------|-------|
| System prompt | Worker character definition | ~600 tokens (fixed) |
| Task context (user) | Task + criteria + checkpoint + knowledge snippets | ~3 000 tokens (capped at construction) |
| Conversation history | Grows with every tool call + result | Unbounded — compressed |

**Compression:**
When estimated token count crosses `effective_compress_threshold()` (default
60 % × `context_length_tokens` = 76 800 for gpt-4o), the middle of the
conversation is replaced by a single summarised user message.  The system
prompt, task context, and last 6 turns are preserved.

After compression the conversation fits in roughly:
`system(600) + task_context(3000) + compress_summary(~2000) + last_6_turns(~6000) ≈ 12 000 tokens`

This means the session can run indefinitely — compression repeats whenever
the threshold is crossed again.

**Context overflow (rare):**
If the LLM still rejects the request after compression (e.g. a single
read-tool output near `max_read_chars` is very large), the session terminates
with status `context_overflow`.  The Executive then replans: it can assign a
narrower task, lower `max_read_chars`, or lower
`context_compress_threshold_tokens`.

**Estimated initial context at session start:** ~4 000–5 000 tokens  
**Safe operating range:** up to `context_length_tokens − 10 000` tokens  
**Maximum input burst per turn:** `max_read_chars // 4 ≈ 7 500 tokens` (one read tool call)

---

### 2b. Executive briefing

The Executive sees a briefing built fresh every tick.

| Slot | Content | Bound |
|------|---------|-------|
| System prompt | Executive character definition | ~500 tokens (fixed) |
| Tick / inbox | Current tick + unread inbox | ~200 tokens (inbox capped at all pending messages, text truncated) |
| Last session result | Status + one-line summary | ~80 tokens (status line only) |
| Health alerts | Alert list | ~200 tokens (only active alerts) |
| Active goals | Up to 20 goals × (title + description + checkpoint) | 20 × 500 = 10 000 tokens (checkpoint capped at 1 500 chars each) |
| Paused/blocked goals | Up to 20 one-liners | ~400 tokens |
| Terminal goal count | One line | ~20 tokens |
| Recent decisions | Last 10 Executive ticks | ~400 tokens |
| PRIORITIES.md | Strategic rationale | 2 000 chars max ≈ 500 tokens |
| Recent reflections | Last 3 summaries (one-liners) | ~150 tokens |

**Estimated total (steady state, 20 active goals):** ~12 000–15 000 tokens  
**Worst case (20 goals × full checkpoint):** ~14 000 tokens  
This is always well within the 100 k minimum.

**Query phase additions (if Executive calls query tools):**
Each query tool result is capped at 4 000 chars ≈ 1 000 tokens.
With `max_executive_queries = 10`, the query phase adds at most
10 × 1 000 = 10 000 tokens.

**Estimated Executive context worst case:** ~25 000 tokens

---

### 2c. Reflector

The Reflector runs periodically (every `reflection_interval` ticks, default 2 000).

| Slot | Content | Bound |
|------|---------|-------|
| System prompt | Reflector character definition | ~600 tokens (fixed) |
| Trigger header | Tick + reason | ~50 tokens |
| All goals | Up to 50 one-liners | ~1 500 tokens |
| Recent failures | Up to 20 error tick summaries | ~500 tokens |
| PRIORITIES.md | 3 000 chars max ≈ 750 tokens |
| Knowledge index | One-liner per file (unlimited files — listing only) | ~50 tokens × N files (no content) |
| Journal entries | 5 goals × 3 entries × 2 000 chars = 30 000 chars ≈ 7 500 tokens |
| Weekly summaries | 5 × 2 000 chars = 10 000 chars ≈ 2 500 tokens |
| Recent sessions | 10 × one-liner | ~400 tokens |
| Previous reflections | 3 × 1 500 chars = 4 500 chars ≈ 1 125 tokens |

**Estimated Reflector context:** ~15 000 tokens  
**Safe within 100 k minimum:** yes, significant headroom

---

### 2d. Summarizer (checkpoint / journal / weekly)

These are one-shot, low-input prompts.

| Prompt | Input | Output |
|--------|-------|--------|
| `checkpoint_prompt` | Task desc + session summary + previous checkpoint | ~1 500 tokens in, ~300 out |
| `journal_prompt` | N tick records (capped by journal_interval) | ~3 000 tokens in, ~400 out |
| `weekly_prompt` | Up to 5 goals × 3 journals × 2 000 chars | ~12 000 tokens in, ~600 out |
| `compress_prompt` | Middle of Worker conversation (variable) | Up to threshold tokens in, ~500 out |

All fit comfortably inside 100 k.

---

## 3. Disk scaling

Disk grows without bound; this is intentional.

| Data | Storage | Growth rate | Archival |
|------|---------|-------------|---------|
| `agent.db` ticks table | SQLite | ~200 bytes/tick | Archived to JSONL every `archive_interval` ticks (default 50 000) |
| Session transcripts | `.jsonl` per session | ~2 KB/action × 200 actions = ~400 KB/session | Gzip-compressed at `archive_interval` |
| Journals | `.md` per interval | ~1 KB/entry | Never deleted (accumulate, indexed in DB) |
| Reflections | `.md` per `reflection_interval` | ~3 KB each | Never deleted (accumulate) |
| Weekly summaries | `.md` per `weekly_summary_interval` | ~2 KB each | Never deleted |
| Knowledge files | `.md` per topic | Overwritten (not growing) | Stable |
| CHECKPOINT.md | Per goal | Overwritten each checkpoint | Stable |
| Archive JSONL | Per `archive_interval` | ~10 MB per archive file | Permanent storage |

### DB size estimate
- 100 k ticks at 200 bytes each = ~20 MB before first archive
- After archive: DB stays small (only live ticks remain)
- Archive JSONL files accumulate on disk but are never re-read by the agent

---

## 4. Context is always bounded — proof

The agent never enters an LLM call with unbounded context because:

1. **Worker**: conversation is compressed before each LLM call when
   `estimated_tokens ≥ effective_compress_threshold()`.  Single read-tool
   outputs are capped at `max_read_chars` chars.

2. **Executive briefing**: all lists are entry-capped (goals, decisions,
   reflections); all content is char-capped (checkpoints, PRIORITIES.md).

3. **Reflector**: all lists are entry-capped (goals, failures, sessions);
   all file content is char-capped per entry.

4. **Summarizer prompts**: journal inputs are bounded by `journal_interval`
   ticks × summary-per-tick (short strings); weekly inputs are bounded by
   5 × 3 × 2 000 chars.

5. **Minimum context validation**: `load_config()` raises `ValueError` at
   startup if `context_length_tokens < 100 000`.

---

## 5. Configuration reference (scaling-relevant settings)

```yaml
context_length_tokens: 128000        # MINIMUM 100 000; agent refuses to start below this
max_output_bytes: 102400             # Shell output cap (100 KB)
max_read_bytes: 102400               # File read raw cap (100 KB)
max_read_chars: 30000                # File read char cap (~7 500 tokens)
max_read_lines: 100                  # Lines returned when no range specified
context_compress_threshold_tokens: 0 # 0 = auto (60 % of context_length_tokens)
max_executive_queries: 10            # Max read-only queries per Executive tick
max_actions_per_session: 200         # Tool calls per Worker session
checkpoint_interval: 50             # Ticks between checkpoint updates
journal_interval: 500               # Ticks between journal entries
reflection_interval: 2000           # Ticks between Reflector passes
weekly_summary_interval: 5000       # Ticks between weekly summaries
archive_interval: 50000             # Ticks between tick archival + transcript compression
```

### Tuning for smaller models (e.g. 100 k context)
Lower the thresholds to give more safety margin:

```yaml
context_length_tokens: 100000
max_read_chars: 15000               # ~3 750 tokens per read
context_compress_threshold_tokens: 50000  # compress at 50 % instead of 60 %
```
