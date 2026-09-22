# Context Health

Measure the health of an AI coding-agent context. Exposes a 0–100 **Context Health** score.

> **Status:** V0.1 — experimental heuristic, not validated.

## Context Health Score

Context Health is an experimental, deterministic 0–100 score computed from observable session signals.

It is computed as a weighted penalty model:

```
health_score = 100 * (1 - weighted_penalty)
```

where:

```
weighted_penalty =
    0.25 * utilization_penalty +
    0.30 * relevance_penalty +
    0.15 * complexity_penalty +
    0.10 * contradiction_penalty +
    0.10 * correction_penalty +
    0.10 * retry_penalty
```

Penalties:
- `utilization_penalty` = `context_utilization`
- `relevance_penalty` = `1 - relevant_context_ratio`
- `complexity_penalty` = `task_complexity`
- `contradiction_penalty` = `contradictions / (contradictions + 3)`
- `correction_penalty` = `corrections / (corrections + 3)`
- `retry_penalty` = `retries / (retries + 3)`

### Status Bands
- **80.0–100.0**: `healthy`
- **60.0–79.99**: `watch`
- **40.0–59.99**: `warning`
- **0.0–39.99**: `critical`

### Important Disclaimers

Context Health is an **experimental heuristic observability metric**.

The weights and thresholds are initial priors and are **NOT** empirically validated.

The score must **NOT** be interpreted as:
- hallucination probability or detector
- probability that the agent is correct
- probability that the agent will fail
- a machine-learning reliability prediction

## Task Complexity

Task complexity is an experimental deterministic heuristic based on observable session/task signals.

It is **NOT**:
- an assessment of reasoning difficulty
- a prediction of success
- a hallucination probability
- a machine-learned complexity model

The score has not been empirically validated.

## Relevant Context Ratio

Relevant Context Ratio is an experimental metric representing the fraction of currently measured context tokens considered relevant to the task.

For V0.1, relevance must be **explicitly supplied or annotated**. The system does **NOT** currently determine semantic relevance automatically.

It is **NOT**:
- a hallucination prediction
- a reliability prediction
- a validated degradation prediction
- a claim of semantic understanding
- a machine-learned relevance detector

This metric serves as a foundational measurement primitive for future validation and experimentation.

## Contradiction Detection

Contradiction detection is an experimental deterministic heuristic designed to identify explicit instruction and requirement reversals within a conversation.

It detects only a limited set of explicit syntactic instruction reversals (such as "use X" vs "do not use X" or "enable X" vs "disable X").

It is **NOT**:
- a general semantic reasoning engine or arbitrary contradiction detector
- an LLM-based evaluation system
- a hallucination detector
- a reasoning verification tool
- a correctness verification or validated reliability prediction system

It has not been empirically validated as a general contradiction detector.

## Correction Events

Correction event detection is an experimental deterministic heuristic that identifies explicit user correction, rejection, and redirect statements during a conversation.

It detects explicit conversational correction markers (such as "No, use PostgreSQL instead", "That's wrong", or "I meant X, not Y").

It is **NOT**:
- an LLM-based sentiment or user satisfaction classifier
- a general semantic intent parser
- a hallucination detector
- a measure of agent correctness or failure probability

## Tool Retries

Tool retry tracking is an experimental deterministic observation signal that identifies repeated tool invocations during a session.

It detects tool retries when:
- an attempt targets the same tool and operation identifier after a previous failed attempt (e.g. `run_command("pytest")` failing and then being executed again), or
- an attempt is explicitly flagged as a retry (`is_retry=True`).

Ordinary repeated calls with different arguments (such as `read_file("a.py")` followed by `read_file("b.py")`) or independent successful operations are **NOT** classified as retries.

It is **NOT**:
- an assessment of whether the agent was "wrong"
- a measure of context degradation
- an automatic retry execution or intervention system
- an LLM-based or semantic similarity evaluator
- a hallucination detector

It has not been empirically validated. A tool retry count does not itself mean the agent is incorrect.

## Install

```bash
pip install -e ".[dev]"
```

## CLI Usage

```bash
# Display help and commands
context-health --help

# Measure Context Health with observable signals and record telemetry
context-health measure --utilization 0.45 --relevance 0.85 --complexity 0.30 --corrections 1

# View recent local telemetry records
context-health history

# Explain signal penalties and contributions for the latest measurement
context-health explain

# Check version
context-health version
```

### Example Output

```text
Context Health: 83.2  [healthy]  (Experimental heuristic)

Context:
  utilization: 45.0%
  relevant:    85.0%
  complexity:  30.0%

Signals:
  contradictions: 0
  corrections:    1
  tool retries:   0

Recommendation:
  CONTINUE (informational UX guidance)
```

## JSONL Telemetry

Context Health persists local, append-only telemetry records to:

```text
~/.context-health/telemetry.jsonl
```

### Telemetry Schema

Each line is a single JSON object capturing:

- `timestamp`: UTC ISO 8601 string
- `session_id`: Opaque stable session UUID
- `turn_index`: Turn sequence index
- `health_score`: Score in `[0.0, 100.0]`
- `health_status`: `'healthy'`, `'watch'`, `'warning'`, or `'critical'`
- `context_utilization`: Float in `[0.0, 1.0]`
- `relevant_context_ratio`: Float in `[0.0, 1.0]`
- `task_complexity`: Float in `[0.0, 1.0]`
- `contradiction_count`: Integer count
- `correction_event_count`: Integer count
- `tool_call_count`: Integer count
- `tool_retry_count`: Integer count
- `model`: Optional model identifier string
- `recommendation`: Informational UX recommendation (`'CONTINUE'`, `'MONITOR'`, `'EVALUATE'`, `'CONSIDER_FRESH_CONTEXT'`)
- `penalties`: Object mapping individual signals to normalized penalties in `[0.0, 1.0]`
- `contributions`: Object mapping individual signals to weighted penalty points

### Privacy & Local Storage

Telemetry records are strictly **local** and contain **only numeric and categorical measurements**.
They do **NOT** store:
- API keys or secrets
- Authorization tokens
- Raw prompt texts or conversation transcripts
- Raw model responses

## Agent Event Protocol

The agent event protocol provides a provider-neutral boundary for real coding-agent telemetry.  Any agent adapter can emit a small set of frozen event dataclasses, and the `ContextHealthRecorder` maps them to the existing `ContextTracker`.

### Events

| Event | Purpose |
|---|---|
| `SessionStarted` | Marks session start (session ID, context window, optional model) |
| `TurnStarted` | Marks beginning of a user turn |
| `ContextUpdated` | Reports provider-reported token measurements |
| `ToolCallStarted` | Records a tool invocation start |
| `ToolCallFinished` | Records a tool invocation result (success, retry flag, error category) |
| `TurnFinished` | Marks end of a turn with externally supplied measurements |
| `SessionFinished` | Marks session end |

### ContextHealthRecorder

`ContextHealthRecorder` consumes agent events and drives the existing `ContextTracker`.  It enforces lightweight session/turn lifecycle rules and stores externally supplied measurements (`relevant_context_ratio`, `task_complexity`, `correction_detected`).

The recorder does **NOT** infer these signals — it accepts them from the caller when available.

### Tool Error Categories

Tool errors use a closed `ToolErrorCategory` enum (`timeout`, `permission_denied`, `not_found`, `syntax_error`, `rate_limited`, `network_error`, `validation_error`, `unknown`).  No arbitrary strings or raw error text enter the event model.

### Privacy Boundary

Agent events contain **only** structural identifiers and numeric measurements.  They do **NOT** store raw prompts, responses, source code, API keys, or error messages.

## Development

```bash
pytest
```

## License

MIT
