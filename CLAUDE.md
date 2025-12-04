# CLAUDE.md - temporalcv

**Version**: 0.1.0 | **Status**: Phase 0 (Repository Setup)

---

## Project Overview

**temporalcv**: Temporal cross-validation with leakage protection for time-series ML.

**Purpose**: Fill ecosystem gap for rigorous time-series validation — shuffled target tests, gap enforcement, statistical testing (DM, PT), and high-persistence handling.

**Audience**: ML practitioners needing statistical rigor without leaving sklearn ecosystem.

---

## Quick Start

```bash
# Development setup
pip install -e ".[dev]"
pytest tests/ -v
mypy src/temporalcv
```

---

## Core Principles (Priority Order)

1. **NEVER FAIL SILENTLY** - Every error must be explicitly reported
2. **External-First Validation** - Shuffled target → Synthetic AR(1) → Internal tests
3. **Temporal Safety** - Gap enforcement: `gap >= horizon` always
4. **Skepticism of Success** - >20% improvement = HALT and investigate

---

## Seed Policy

All stochastic functions accept `random_state: int | None` parameter:
- **Default**: `random_state=None` (non-deterministic for exploration)
- **Tests**: Always use fixed seed for reproducibility
- **Gates**: Shuffled target uses `random_state` with documented default

---

## Package Structure

```
src/temporalcv/
├── __init__.py
├── py.typed              # PEP 561 type marker
├── gates/                # Validation gates (HALT/PASS/WARN/SKIP)
├── tests/                # Statistical tests (DM, PT)
├── cv/                   # Walk-forward cross-validation
├── metrics/              # MC-SS, move-only MAE
└── benchmarks/           # Dataset loaders
```

---

## Exit Codes (CLI)

```python
EXIT_OK = 0       # All gates passed
EXIT_HALT = 1     # Gate failure - stop pipeline
EXIT_WARN = 2     # Gate warning - continue with caution
EXIT_SKIP = 3     # Gate skipped - insufficient data
EXIT_ERROR = 4    # Unexpected error
```

---

## Development Standards

### Code Style
- **Formatter**: ruff (black-compatible, 100-char lines)
- **Type hints**: Required on all public functions
- **Docstrings**: NumPy style

### Testing
- Real tests only — no stubs, TODOs, or placeholders
- Coverage target: 80%+ for modules
- Anti-pattern tests for leakage detection

### Git
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`
- Pre-commit hooks must pass

---

## Documentation Index

| Document | Purpose | Location |
|----------|---------|----------|
| Planning index | Phase tracking | `docs/plans/INDEX.md` |
| API design | Code examples | `docs/plans/reference/api_design.md` |
| Ecosystem gaps | What we fill | `docs/plans/reference/ecosystem_gaps.md` |

---

## Reference Patterns

This project follows patterns from:
- `~/Claude/lever_of_archimedes/patterns/testing.md` - Validation architecture
- `~/Claude/lever_of_archimedes/patterns/data_leakage_prevention.md` - ML guardrails
- `~/Claude/myga-forecasting-v3/CLAUDE.md` - Domain-specific validation

---

## Suspicious Results Protocol

| Condition | Action |
|-----------|--------|
| >20% improvement over persistence | Run shuffled target test |
| MAE below theoretical minimum | Test on synthetic AR(1) |
| h=1 >> h=2,3,4 | Check horizon gap enforcement |
