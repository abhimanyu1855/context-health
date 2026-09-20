# Context Health — Project Contract

## Goal

Build V0.1 of a developer tool that measures the health of an AI coding-agent context and exposes a 0–100 Context Health score.

The initial hypothesis is:

> "For the same context utilization level, sessions with high irrelevant-context density (low relevant-context ratio) degrade faster than sessions with low irrelevant-context density."

## V0.1 Scope

V0.1 ONLY includes:

1. Python CLI
2. Anthropic API wrapper
3. Multi-turn conversation support
4. Context/token tracking
5. Task complexity estimation
6. Estimated relevant-context ratio
7. Basic contradiction detection
8. Correction-event detection
9. Tool retry tracking where applicable
10. Context Health score
11. Terminal display
12. JSONL telemetry
13. Unit tests

## Explicitly NOT in V0.1

Do NOT implement:

- automatic context branching
- automatic context compression
- model routing
- reasoning routing
- multi-provider routing
- dashboard
- web application
- SaaS backend
- database
- RAGAS
- expensive evaluator loops
- ML model
- hallucination probability
- IDE UI integration

## Important Terminology

The system must call the metric:

**"Context Health"**

It must NOT claim:

**"Hallucination probability"**

Context Health is an experimental heuristic until validated using real telemetry.

## Source of Truth

When context becomes large or uncertain, the agent must reconstruct project state from:

1. `PROJECT_CONTRACT.md`
2. `README.md`
3. source code
4. tests
5. git history/diff

The conversation itself is NOT the source of truth.

## Development Rules

- Work in small milestones.
- Run tests after each milestone.
- Do not make unrelated changes.
- Do not silently expand scope.
- Do not delete working functionality without explaining why.
- Keep implementation simple.
- Prefer deterministic signals over additional LLM calls.
- Document experimental assumptions.

## Recovery Protocol

If you believe the current conversation/context is becoming unreliable:

1. Stop implementation.
2. Inspect `PROJECT_CONTRACT.md`.
3. Inspect `README.md`.
4. Inspect `git status`.
5. Inspect `git diff`.
6. Run tests.
7. Summarize current implementation state.
8. Continue only after reconstructing the actual repository state.
