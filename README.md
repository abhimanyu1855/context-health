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
