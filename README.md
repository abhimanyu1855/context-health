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
