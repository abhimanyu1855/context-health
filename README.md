# Context Health

Measure the health of an AI coding-agent context. Exposes a 0–100 **Context Health** score.

> **Status:** V0.1 — experimental heuristic, not validated.

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

## Usage

```bash
context-health --help
context-health version
```

## Development

```bash
pytest
```

## License

MIT
